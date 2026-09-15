"""
Observer bus for parallel subagent progress.

LangGraph only returns from spawn_parallel_subagents after all workers finish;
this side channel publishes live start/done events to the UI/SSE.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

Observer = Callable[[Dict[str, Any]], None]

_lock = threading.RLock()
_observers: List[Observer] = []


def register_subagent_observer(callback: Observer) -> None:
    """Register a callback invoked with subagent progress event dicts."""
    with _lock:
        if callback not in _observers:
            _observers.append(callback)


def unregister_subagent_observer(callback: Observer) -> None:
    with _lock:
        if callback in _observers:
            _observers.remove(callback)


def clear_subagent_observers() -> None:
    with _lock:
        _observers.clear()


def publish_subagent_event(event: Dict[str, Any]) -> None:
    """
    Publish a progress event to all observers. Failures never break the agent.

    Event shapes:
      {"type":"subagent_start","name":...,"role":...,"task":...,"total":N,"index":i}
      {"type":"subagent_done","name":...,"role":...,"status":"success|error",
       "result_preview":...,"elapsed_ms":...}
      {"type":"subagents_all_done","count":N,"success":n,"error":m}
    """
    with _lock:
        observers = list(_observers)
    for cb in observers:
        try:
            cb(event)
        except Exception:
            pass
