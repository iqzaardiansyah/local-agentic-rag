"""
Free local run metrics for agent turns (tool latency, tokens/sec, wall time).

Collected by the runner and surfaced in UI + API. No external services.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class TurnMetrics:
    """Accumulate metrics for one agent turn. Not thread-safe across turns."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.first_token_at: Optional[float] = None
        self.token_count = 0
        self.token_chars = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self._open_tools: Dict[str, float] = {}
        self._tool_seq = 0

    def mark_tool_call(self, tool_name: str, call_id: Optional[str] = None) -> str:
        key = call_id or f"{tool_name}#{self._tool_seq}"
        self._tool_seq += 1
        self._open_tools[key] = time.perf_counter()
        return key

    def mark_tool_result(self, tool_name: str, call_id: Optional[str] = None) -> None:
        key = call_id or tool_name
        started = self._open_tools.pop(key, None)
        # Fallback: pop any open call for this tool name
        if started is None:
            for k in list(self._open_tools.keys()):
                if k.startswith(tool_name):
                    started = self._open_tools.pop(k)
                    break
        elapsed_ms = (
            round((time.perf_counter() - started) * 1000.0, 1)
            if started is not None
            else None
        )
        self.tool_calls.append(
            {
                "tool": tool_name,
                "elapsed_ms": elapsed_ms,
                "call_id": key,
            }
        )

    def mark_token(self, token: str) -> None:
        now = time.perf_counter()
        if self.first_token_at is None:
            self.first_token_at = now
        self.token_count += 1
        self.token_chars += len(token or "")

    def snapshot(self) -> Dict[str, Any]:
        ended = time.perf_counter()
        wall_ms = round((ended - self.started_at) * 1000.0, 1)
        ttft_ms = (
            round((self.first_token_at - self.started_at) * 1000.0, 1)
            if self.first_token_at is not None
            else None
        )
        gen_seconds = (
            (ended - self.first_token_at) if self.first_token_at is not None else 0.0
        )
        tokens_per_sec = (
            round(self.token_count / gen_seconds, 2)
            if gen_seconds > 0.05 and self.token_count > 0
            else None
        )
        # Approximate token estimate from chars when stream chunks aren't tokens
        approx_tokens = max(self.token_count, self.token_chars // 4)
        tool_total = sum(
            t["elapsed_ms"] for t in self.tool_calls if t.get("elapsed_ms") is not None
        )
        return {
            "wall_ms": wall_ms,
            "ttft_ms": ttft_ms,
            "token_chunks": self.token_count,
            "token_chars": self.token_chars,
            "approx_tokens": approx_tokens,
            "tokens_per_sec": tokens_per_sec,
            "tool_count": len(self.tool_calls),
            "tool_total_ms": round(tool_total, 1) if tool_total else 0.0,
            "tools": self.tool_calls,
            "open_tool_calls": list(self._open_tools.keys()),
        }


def format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    """One-line human summary for captions / logs."""
    if not metrics:
        return "No metrics"
    parts = [f"wall {metrics.get('wall_ms', '?')} ms"]
    if metrics.get("ttft_ms") is not None:
        parts.append(f"TTFT {metrics['ttft_ms']} ms")
    if metrics.get("tokens_per_sec") is not None:
        parts.append(f"{metrics['tokens_per_sec']} tok/s")
    elif metrics.get("approx_tokens"):
        parts.append(f"~{metrics['approx_tokens']} tokens")
    if metrics.get("tool_count"):
        parts.append(f"{metrics['tool_count']} tools / {metrics.get('tool_total_ms', 0)} ms")
    return " · ".join(parts)
