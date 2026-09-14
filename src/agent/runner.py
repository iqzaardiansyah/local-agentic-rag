"""
Shared agent turn runner used by Streamlit and the FastAPI SSE endpoint.

Yields typed events so UIs can render tokens, tool calls, and citations
with one code path. Free / local only.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.memory.session_context import build_agent_messages
from src.rag.citations import clear_citations, get_citations


def run_agent_stream(
    user_message: str,
    session_id: Optional[str] = None,
    history: Optional[Sequence[BaseMessage]] = None,
    include_cross_session: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream one agent turn as event dicts:

      {"type": "meta", "session_id": ..., "model": ...}
      {"type": "tool_call", "tool": ..., "args": ...}
      {"type": "token", "token": "..."}
      {"type": "tool_result", "tool": ..., "output": ...}
      {"type": "grade", "label": "...", "detail": "..."}
      {"type": "done", "answer": ..., "steps": [...], "citations": [...]}
      {"type": "error", "error": "..."}

    If `history` is provided it is used as-is (already includes the user message).
    Otherwise messages are built from `session_id` + `user_message`.
    """
    from src.agent.graph import LLM_BASE_URL, LLM_MODEL, app as agent_app

    clear_citations()
    tool_steps: List[Dict[str, Any]] = []
    registered_calls: set = set()
    final_response = ""

    try:
        if history is not None:
            messages: List[BaseMessage] = list(history)
        else:
            messages = build_agent_messages(
                user_message,
                session_id=session_id,
                include_cross_session=include_cross_session,
            )

        yield {
            "type": "meta",
            "session_id": session_id,
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "message_count": len(messages),
        }

        inputs = {"messages": messages}

        for chunk, meta in agent_app.stream(inputs, stream_mode="messages"):
            node_name = (meta or {}).get("langgraph_node", "")

            if node_name == "agent":
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        call_id = tc.get("id") or str(tc)
                        if call_id in registered_calls:
                            continue
                        registered_calls.add(call_id)
                        tool_name = tc.get("name", "tool")
                        tool_args = tc.get("args", {})
                        tool_steps.append(
                            {"type": "call", "tool": tool_name, "args": tool_args}
                        )
                        yield {
                            "type": "tool_call",
                            "tool": tool_name,
                            "args": tool_args,
                        }
                elif getattr(chunk, "content", None):
                    token = chunk.content
                    final_response += token
                    yield {"type": "token", "token": token}

            elif node_name == "action":
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    tool_content = getattr(chunk, "content", "") or ""
                    tool_steps.append(
                        {
                            "type": "result",
                            "tool": tool_name,
                            "output": tool_content[:2000],
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "output": tool_content[:2000],
                    }

            elif node_name == "grade_retrieval":
                if isinstance(chunk, ToolMessage):
                    content = getattr(chunk, "content", "") or ""
                    label, detail = _classify_grade(content)
                    if label:
                        yield {"type": "grade", "label": label, "detail": detail}

        citations = get_citations()
        yield {
            "type": "done",
            "answer": final_response,
            "steps": tool_steps,
            "citations": citations,
        }

    except Exception as e:
        yield {"type": "error", "error": str(e)}


def _classify_grade(content: str) -> tuple:
    if "High Confidence Match" in content:
        return "crag_high", "Verified local documents as relevant & grounded."
    if "Low Local Document Relevance" in content:
        return "crag_low", "Low relevance score. Avoiding hallucination."
    if "Reflexion Auto-Debugger: Execution Failure Detected" in content:
        return "reflexion_fail", "Execution error caught; auto-fix loop engaged."
    if "Reflexion Auto-Debugger: Execution verified" in content:
        return "reflexion_ok", "Code execution verified with zero errors."
    return "", ""


def run_agent_once(
    user_message: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-streaming convenience wrapper. Returns the final done/error event."""
    final: Dict[str, Any] = {
        "type": "done",
        "answer": "",
        "steps": [],
        "citations": [],
    }
    for event in run_agent_stream(user_message, session_id=session_id):
        if event["type"] == "done":
            final = event
        elif event["type"] == "error":
            return event
    return final
