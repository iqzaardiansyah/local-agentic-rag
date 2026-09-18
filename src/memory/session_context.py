"""
Build agent input: within-session history + cross-session brief + episodic recall.
Shared by the Streamlit UI and FastAPI API.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.memory import chat_sessions

# Bound session history sent to the model.
DEFAULT_MAX_TURNS = 24
CROSS_SESSION_LIMIT = 3
EPISODIC_TOP_K = 3


def _first_line(text: str, limit: int = 100) -> str:
    line = (text or "").strip().splitlines()[0] if text else ""
    line = re.sub(r"\s+", " ", line).strip()
    return (line[: limit - 1] + "…") if len(line) > limit else line


def get_cross_session_brief(
    exclude_session_id: str,
    query: Optional[str] = None,
    max_sessions: int = CROSS_SESSION_LIMIT,
) -> str:
    """
    Build a short brief of other chat sessions + related episodic memories.
    Returns empty string when there is nothing useful to inject.
    """
    parts: List[str] = []

    sessions = chat_sessions.list_sessions()
    others = [s for s in sessions if s["id"] != exclude_session_id][:max_sessions]
    if others:
        lines = ["Prior chat sessions (context only — do not assume they are still active):"]
        for s in others:
            msgs = chat_sessions.load_messages(s["id"], limit=6)
            last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
            lines.append(
                f"- [{s.get('title') or 'Untitled'}] last topic: {_first_line(last_user) or 'n/a'}"
            )
        parts.append("\n".join(lines))

    if query and query.strip():
        try:
            from src.memory.episodic_memory import recall_memories

            mems = recall_memories(query, top_k=EPISODIC_TOP_K)
            if mems:
                mlines = ["Relevant long-term memories:"]
                for m in mems:
                    mlines.append(f"- [{m.get('category', 'general')}] {m.get('fact', '')}")
                parts.append("\n".join(mlines))
        except Exception:
            pass

    if not parts:
        return ""
    return "\n\n".join(parts)


def messages_from_session(
    session_id: Optional[str],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> List[BaseMessage]:
    """Convert persisted session rows into LangChain chat messages (chronological)."""
    if not session_id:
        return []
    rows = chat_sessions.load_messages(session_id)
    # Drop empty assistant placeholders
    rows = [m for m in rows if (m.get("content") or "").strip() or m["role"] == "user"]
    # Keep the most recent max_turns user+assistant exchanges
    if len(rows) > max_turns * 2:
        rows = rows[-(max_turns * 2) :]

    out: List[BaseMessage] = []
    for m in rows:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if m["role"] == "user":
            out.append(HumanMessage(content=content))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=content))
    return out


def build_agent_messages(
    user_message: str,
    session_id: Optional[str] = None,
    include_cross_session: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> List[BaseMessage]:
    """
    Build the full message list for one agent turn.

    Order:
      1. Optional cross-session + episodic context (as a SystemMessage)
      2. Within-session history (multi-turn)
      3. Current user message
    """
    if not user_message or not user_message.strip():
        raise ValueError("user_message must be non-empty")

    messages: List[BaseMessage] = []

    if include_cross_session and session_id:
        brief = get_cross_session_brief(session_id, query=user_message)
        if brief:
            messages.append(
                SystemMessage(
                    content=(
                        "[SESSION CONTEXT — for continuity only]\n"
                        f"{brief}\n\n"
                        "If this conflicts with the current user request, prefer the current request."
                    )
                )
            )

    history = messages_from_session(session_id, max_turns=max_turns)
    # Avoid duplicating the just-typed user message if it was already saved.
    if history and isinstance(history[-1], HumanMessage):
        if history[-1].content.strip() == user_message.strip():
            messages.extend(history)
            return messages
    messages.extend(history)
    messages.append(HumanMessage(content=user_message.strip()))
    return messages


def persist_turn(
    session_id: Optional[str],
    user_message: str,
    assistant_message: str,
    steps: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    skip_user_if_saved: bool = False,
) -> None:
    """
    Persist one completed user→assistant exchange for a session.

    skip_user_if_saved: set True when the caller already wrote the user row
    (e.g. API streaming that saves the user message up-front).
    """
    if not session_id:
        return
    if not skip_user_if_saved:
        existing = chat_sessions.load_messages(session_id, limit=4)
        last_user = next(
            (m["content"] for m in reversed(existing) if m["role"] == "user"), None
        )
        if last_user is None or last_user.strip() != user_message.strip():
            chat_sessions.save_message(session_id, "user", user_message)
    if assistant_message and assistant_message.strip():
        chat_sessions.save_message(
            session_id,
            "assistant",
            assistant_message,
            steps=steps,
            citations=citations,
        )
