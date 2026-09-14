"""
Export a chat turn (or full session) as a research brief: answer + ranked sources.
Free, local Markdown.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "workspace")


def _last_assistant_block(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m
    return None


def _last_user_block(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for m in reversed(messages):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return m
    return None


def build_research_brief_markdown(
    session_id: str,
    message_id: Optional[int] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a Markdown research brief from a session.

    If message_id is provided and points at an assistant message, brief that turn.
    Otherwise briefs the latest assistant answer + its citations.
    """
    from src.memory import chat_sessions

    messages = chat_sessions.load_messages(session_id)
    if not messages:
        return {"success": False, "message": "Session has no messages.", "session_id": session_id}

    sessions = {s["id"]: s for s in chat_sessions.list_sessions()}
    session_title = title or (sessions.get(session_id) or {}).get("title") or "Chat session"

    answer_msg = None
    if message_id is not None:
        for m in messages:
            if m.get("id") == message_id and m.get("role") == "assistant":
                answer_msg = m
                break
    if answer_msg is None:
        answer_msg = _last_assistant_block(messages)
    if answer_msg is None:
        return {"success": False, "message": "No assistant answer found in session.", "session_id": session_id}

    # Question immediately before this answer
    user_msg = None
    for m in messages:
        if answer_msg.get("id") is not None and m.get("id") is not None:
            if m["id"] < answer_msg["id"] and m.get("role") == "user":
                user_msg = m
        elif m.get("role") == "user":
            user_msg = m  # fallback chronological
    if user_msg is None:
        user_msg = _last_user_block(messages)

    citations = answer_msg.get("citations") or []
    steps = answer_msg.get("steps") or []
    metrics = answer_msg.get("metrics") or {}
    ts = datetime.now().isoformat(timespec="seconds")

    lines: List[str] = [
        f"# Research Brief — {session_title}",
        "",
        f"- **Session:** `{session_id}`",
        f"- **Generated:** {ts}",
    ]
    if metrics:
        bits = []
        if metrics.get("wall_ms") is not None:
            bits.append(f"wall {metrics['wall_ms']} ms")
        if metrics.get("tokens_per_sec") is not None:
            bits.append(f"{metrics['tokens_per_sec']} tok/s")
        if metrics.get("tool_count"):
            bits.append(f"{metrics['tool_count']} tool call(s)")
        if bits:
            lines.append(f"- **Run:** {' · '.join(bits)}")
    lines.append("")

    lines.extend(["## Question", ""])
    lines.append((user_msg.get("content") if user_msg else "*(not found)*") or "")
    lines.append("")

    lines.extend(["## Answer", ""])
    lines.append(answer_msg.get("content") or "")
    lines.append("")

    lines.extend(["## Sources", ""])
    if citations:
        for i, c in enumerate(citations, 1):
            src = c.get("source") or "Unknown"
            kind = c.get("kind") or "document"
            score = c.get("score")
            score_txt = f" — score {score:.3f}" if isinstance(score, (int, float)) else ""
            rel = c.get("rel_path") or c.get("path") or ""
            lines.append(f"{i}. **`{src}`** (`{kind}`){score_txt}")
            if rel:
                lines.append(f"   - Path: `{rel}`")
            if c.get("entities"):
                lines.append(f"   - Entities: {c['entities']}")
            preview = (c.get("preview") or "").replace("\n", " ")
            if preview:
                lines.append(f"   - Preview: {preview}")
            lines.append("")
    else:
        lines.append("*(No structured citations recorded for this answer.)*")
        lines.append("")

    tool_names = [s.get("tool") for s in steps if s.get("type") == "call" and s.get("tool")]
    if tool_names:
        # unique preserve order
        seen = set()
        ordered = []
        for t in tool_names:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        lines.extend(["## Tools Used", ""])
        lines.append(", ".join(f"`{t}`" for t in ordered))
        lines.append("")

    lines.extend(
        [
            "---",
            "*Local Agentic RAG research brief — free, offline export.*",
            "",
        ]
    )

    return {
        "success": True,
        "session_id": session_id,
        "title": session_title,
        "markdown": "\n".join(lines),
        "citation_count": len(citations),
        "question": (user_msg or {}).get("content", ""),
        "answer": answer_msg.get("content", ""),
    }


def write_research_brief(
    session_id: str,
    message_id: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and write the research brief to disk (default workspace/research_brief_<sid8>.md)."""
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    built = build_research_brief_markdown(session_id, message_id=message_id)
    if not built.get("success"):
        return built

    if not output_path:
        safe = "".join(c if c.isalnum() else "_" for c in session_id)[:12]
        output_path = os.path.join(DEFAULT_OUT_DIR, f"research_brief_{safe}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(built["markdown"])

    built["path"] = output_path
    built["bytes"] = len(built["markdown"].encode("utf-8"))
    return built
