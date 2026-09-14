"""In-process citation store for RAG answers. Free and local."""

from typing import Any, Dict, List, Optional

_citations: List[Dict[str, Any]] = []


def set_citations(items: List[Dict[str, Any]]) -> None:
    global _citations
    _citations = items


def get_citations() -> List[Dict[str, Any]]:
    return list(_citations)


def clear_citations() -> None:
    global _citations
    _citations = []


def format_citation_footer(items: Optional[List[Dict[str, Any]]] = None) -> str:
    items = items if items is not None else _citations
    if not items:
        return ""
    lines = ["", "---", "**Sources used for this answer:**"]
    for i, c in enumerate(items, 1):
        src = c.get("source", "Unknown")
        score = c.get("score")
        preview = (c.get("preview") or "").replace("\n", " ")[:120]
        score_txt = f" · score {score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(f"{i}. `{src}`{score_txt} — {preview}")
    return "\n".join(lines)
