"""
Shared agent turn runner for Streamlit and FastAPI SSE.
Yields token, tool, subagent, and citation events on one code path.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any, Dict, Generator, List, Optional, Sequence

from langchain_core.messages import BaseMessage

from src.memory.session_context import build_agent_messages
from src.rag.citations import clear_citations, get_citations
from src.agent.metrics import TurnMetrics
from src.agent.subagent_events import (
    register_subagent_observer,
    unregister_subagent_observer,
)

_SENTINEL = object()

# Remove tool-call XML that appears as plain text.
_TOOL_XML_BLOCK = re.compile(
    r"<ifm\|tool_calls>[\s\S]*?</ifm\|tool_calls>",
    re.I,
)
_TOOL_XML_SINGLE = re.compile(
    r"<ifm\|tool_call>[\s\S]*?</ifm\|tool_call>",
    re.I,
)
_IFM_TAG = re.compile(r"</?ifm\|[^>]*>", re.I)


def _strip_tool_xml(text: str) -> str:
    if not text or "<ifm|" not in text:
        return text
    text = _TOOL_XML_BLOCK.sub("", text)
    text = _TOOL_XML_SINGLE.sub("", text)
    text = _IFM_TAG.sub("", text)
    return text


def _chunk_content_text(chunk: Any) -> str:
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for part in c:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    return str(c)


def _chunk_reasoning_text(chunk: Any) -> str:
    for attr in ("reasoning_content", "reasoning", "thinking"):
        val = getattr(chunk, attr, None)
        if isinstance(val, str) and val:
            return val
    ak = getattr(chunk, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = ak.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def run_agent_stream(
    user_message: str,
    session_id: Optional[str] = None,
    history: Optional[Sequence[BaseMessage]] = None,
    include_cross_session: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream one agent turn as event dicts:
    meta, heartbeat, tool_call, token, thinking, tool_result, grade,
    subagent_*, done, error.
    """
    import time as _time

    from src.agent.graph import LLM_BASE_URL, LLM_MODEL, app as agent_app

    clear_citations()
    out_q: "queue.Queue[Any]" = queue.Queue()
    tool_steps: List[Dict[str, Any]] = []
    registered_calls: set = set()
    final_response = ""
    metrics = TurnMetrics()
    call_id_to_tool: Dict[str, str] = {}
    started = _time.perf_counter()
    last_visible_at = started
    last_reasoning = ""

    def _observer(event: Dict[str, Any]) -> None:
        out_q.put(dict(event))

    register_subagent_observer(_observer)

    def _heartbeat(phase: str) -> None:
        nonlocal last_visible_at
        now = _time.perf_counter()
        last_visible_at = now
        out_q.put({
            "type": "heartbeat",
            "phase": phase,
            "elapsed_ms": round((now - started) * 1000.0, 1),
        })

    def _worker() -> None:
        nonlocal final_response, last_reasoning
        try:
            if history is not None:
                messages: List[BaseMessage] = list(history)
            else:
                messages = build_agent_messages(
                    user_message,
                    session_id=session_id,
                    include_cross_session=include_cross_session,
                )

            out_q.put({
                "type": "meta",
                "session_id": session_id,
                "model": LLM_MODEL,
                "base_url": LLM_BASE_URL,
                "message_count": len(messages),
            })
            _heartbeat("starting")

            inputs = {"messages": messages}

            for chunk, meta in agent_app.stream(inputs, stream_mode="messages"):
                node_name = (meta or {}).get("langgraph_node", "")

                if node_name == "agent":
                    tool_calls = getattr(chunk, "tool_calls", None) or []
                    if tool_calls:
                        for tc in tool_calls:
                            call_id = tc.get("id") or str(tc)
                            if call_id in registered_calls:
                                continue
                            registered_calls.add(call_id)
                            tool_name = tc.get("name", "tool")
                            tool_args = tc.get("args", {})
                            call_id_to_tool[call_id] = tool_name
                            metrics.mark_tool_call(tool_name, call_id=call_id)
                            tool_steps.append(
                                {"type": "call", "tool": tool_name, "args": tool_args}
                            )
                            _heartbeat("tool_call")
                            out_q.put({
                                "type": "tool_call",
                                "tool": tool_name,
                                "args": tool_args,
                            })

                    reasoning = _chunk_reasoning_text(chunk)
                    if reasoning:
                        if last_reasoning and reasoning.startswith(last_reasoning):
                            delta = reasoning[len(last_reasoning):]
                        else:
                            delta = reasoning
                        last_reasoning = reasoning
                        if delta:
                            metrics.mark_token(delta)
                            out_q.put({"type": "thinking", "token": delta})

                    token = _chunk_content_text(chunk)
                    if token:
                        token = _strip_tool_xml(token)
                        if token:
                            final_response += token
                            metrics.mark_token(token)
                            out_q.put({"type": "token", "token": token})

                elif node_name == "action":
                    from langchain_core.messages import ToolMessage

                    if isinstance(chunk, ToolMessage):
                        tool_name = getattr(chunk, "name", "tool")
                        tool_content = getattr(chunk, "content", "") or ""
                        tool_call_id = getattr(chunk, "tool_call_id", None) or tool_name
                        metrics.mark_tool_result(tool_name, call_id=tool_call_id)
                        tool_steps.append(
                            {
                                "type": "result",
                                "tool": tool_name,
                                "output": tool_content[:2000],
                            }
                        )
                        _heartbeat("tool_result")
                        out_q.put({
                            "type": "tool_result",
                            "tool": tool_name,
                            "output": tool_content[:2000],
                        })

                elif node_name == "grade_retrieval":
                    from langchain_core.messages import ToolMessage

                    if isinstance(chunk, ToolMessage):
                        content = getattr(chunk, "content", "") or ""
                        label, detail = _classify_grade(content)
                        if label:
                            out_q.put({
                                "type": "grade",
                                "label": label,
                                "detail": detail,
                            })

                now = _time.perf_counter()
                if now - last_visible_at > 3.0:
                    phase = "tools" if tool_steps else "thinking"
                    _heartbeat(phase)

            citations = get_citations()
            answer = final_response
            if not answer.strip() and tool_steps:
                called = [s.get("tool") for s in tool_steps if s.get("type") == "call"]
                uniq = list(dict.fromkeys(called))
                answer = (
                    "Finished with tool activity but no text summary. Tools: "
                    + ", ".join(f"`{t}`" for t in uniq[:12])
                    + f" ({len(tool_steps)} steps). See ./workspace for artifacts."
                )
            out_q.put({
                "type": "done",
                "answer": answer,
                "steps": tool_steps,
                "citations": citations,
                "metrics": metrics.snapshot(),
            })

        except Exception as e:
            snap = metrics.snapshot()
            out_q.put({"type": "error", "error": str(e), "metrics": snap})
        finally:
            try:
                unregister_subagent_observer(_observer)
            except Exception:
                pass
            out_q.put(_SENTINEL)

    worker = threading.Thread(target=_worker, name="agent-runner", daemon=True)
    worker.start()

    while True:
        item = out_q.get()
        if item is _SENTINEL:
            break
        yield item

    worker.join(timeout=5)


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
        "metrics": {},
    }
    for event in run_agent_stream(user_message, session_id=session_id):
        if event["type"] == "done":
            final = event
        elif event["type"] == "error":
            return event
    return final
