"""Heuristic capture of durable user preferences into episodic memory."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.memory.episodic_memory import list_all_memories, save_memory

# Phrases that usually introduce a durable preference / rule.
_PREF_PATTERNS = [
    re.compile(r"\b(?:please\s+)?remember\s+that\b[:\s]+(.{8,200})", re.I),
    re.compile(r"\b(?:please\s+)?always\s+(.{8,180})", re.I),
    re.compile(r"\b(?:please\s+)?never\s+(?:use\s+|do\s+)?(.{8,180})", re.I),
    re.compile(r"\bi\s+prefer\s+(.{6,180})", re.I),
    re.compile(r"\bi\s+always\s+(.{6,180})", re.I),
    re.compile(r"\bi\s+never\s+(?:use\s+|want\s+)?(.{6,180})", re.I),
    re.compile(r"\bmy\s+name\s+is\s+([A-Za-z][\w .'-]{1,60})", re.I),
    re.compile(r"\bi\s+work\s+(?:at|for|on)\s+(.{3,120})", re.I),
    re.compile(r"\bfrom\s+now\s+on\b[:,]?\s*(.{8,180})", re.I),
    re.compile(r"\bdon't\s+(?:use|call|ask)\s+(.{6,160})", re.I, ),
    re.compile(r"\bdo\s+not\s+(?:use|call|ask)\s+(.{6,160})", re.I),
    re.compile(r"\buse\s+([A-Za-z0-9_.-]{2,40})\s+instead\s+of\s+([A-Za-z0-9_.-]{2,40})", re.I),
    re.compile(r"\bcall\s+me\s+([A-Za-z][\w .'-]{1,40})", re.I),
]

_CATEGORY_RULES = [
    (re.compile(r"\bname\b|\bcall\s+me\b", re.I), "user_info"),
    (re.compile(r"\bwork\b|\bcompany\b|\bteam\b|\brole\b", re.I), "user_info"),
    (re.compile(r"\balways\b|\bnever\b|\bdon't\b|\bdo\s+not\b|\bprefer\b", re.I), "preference"),
    (re.compile(r"\bstyle\b|\bformat\b|\bmarkdown\b|\bcode\b|\bpython\b", re.I), "preference"),
    (re.compile(r"\barchitecture\b|\bstack\b|\blibrary\b|\bframework\b", re.I), "architecture"),
    (re.compile(r"\brule\b|\bpolicy\b|\bconvention\b|\bfrom\s+now\s+on\b", re.I), "project_rule"),
]


def _normalize_fact(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.strip(" .;,:")
    if len(text) > 220:
        text = text[:217] + "..."
    return text


def _categorize(fact: str) -> str:
    for pattern, cat in _CATEGORY_RULES:
        if pattern.search(fact):
            return cat
    return "preference"


def _is_duplicate(fact: str, existing: Optional[List[Dict[str, Any]]] = None) -> bool:
    if existing is None:
        try:
            existing = list_all_memories()
        except Exception:
            return False
    key = fact.lower()
    for m in existing or []:
        prev = (m.get("fact") or "").lower()
        if not prev:
            continue
        if key in prev or prev in key:
            return True
        # Token overlap for near-duplicates
        a = set(re.findall(r"\w+", key))
        b = set(re.findall(r"\w+", prev))
        if a and b:
            overlap = len(a & b) / max(len(a), len(b))
            if overlap >= 0.85:
                return True
    return False


def extract_preference_facts(user_message: str) -> List[str]:
    """Extract durable-looking preference facts from a user message."""
    if not user_message or len(user_message.strip()) < 8:
        return []
    text = user_message.strip()
    found: List[str] = []
    seen = set()
    for pattern in _PREF_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex and match.lastindex >= 2:
                fact = _normalize_fact(match.group(0))
            else:
                captured = match.group(1) if match.lastindex else match.group(0)
                # Prefer full match for "use X instead of Y"
                fact = _normalize_fact(match.group(0) if match.lastindex >= 2 else captured)
                if pattern.pattern.startswith("\\buse\\s+"):
                    fact = _normalize_fact(match.group(0))
            if not fact or len(fact) < 8:
                continue
            low = fact.lower()
            if low in seen:
                continue
            seen.add(low)
            found.append(fact)
    return found[:3]


def maybe_autocapture_preferences(
    user_message: str,
    session_id: Optional[str] = None,
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Capture preference facts from a user message into episodic memory.

    Returns list of saved facts: [{"fact", "category", "id"}]. Empty if disabled
    or nothing durable was found / already stored.
    """
    if not enabled:
        return []
    candidates = extract_preference_facts(user_message)
    if not candidates:
        return []

    saved: List[Dict[str, Any]] = []
    try:
        existing = list_all_memories()
    except Exception:
        existing = []

    for fact in candidates:
        if _is_duplicate(fact, existing):
            continue
        category = _categorize(fact)
        meta = {"source": "auto_capture"}
        if session_id:
            meta["session_id"] = session_id
        try:
            mem_id = save_memory(fact, category=category, metadata=meta)
            existing.append({"fact": fact, "category": category})
            saved.append({"fact": fact, "category": category, "id": mem_id})
        except Exception:
            continue
    return saved
