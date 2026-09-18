"""
In-process citation store for RAG answers.

Citations now resolve to real on-disk source files so the UI can open,
preview, and download them. Free and local.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
WORKSPACE_DIR = os.path.join(ROOT_DIR, "workspace")

_citations: List[Dict[str, Any]] = []

_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".7z", ".rar",
    ".xlsx", ".xls", ".docx", ".pptx", ".sqlite", ".db",
    ".bin", ".pt", ".pth", ".onnx", ".mp3", ".mp4", ".wav",
}


def set_citations(items: List[Dict[str, Any]]) -> None:
    global _citations
    filtered = []
    for c in items:
        ec = enrich_citation(c)
        src = (ec.get("source") or "").lower()
        if src.endswith((".db", ".sqlite", ".sqlite3")) or src in {
            "chat_sessions.db",
            "web_search_cache.db",
        }:
            continue
        if ec.get("kind") == "graphrag":
            filtered.append(ec)
            continue
        path = ec.get("path") or ""
        if path and not ec.get("readable") and ec.get("exists"):
            # Binary/on-disk file that cannot be shown — still useful if user wants download
            filtered.append(ec)
            continue
        filtered.append(ec)
    _citations = filtered


def get_citations() -> List[Dict[str, Any]]:
    return list(_citations)


def clear_citations() -> None:
    global _citations
    _citations = []


def resolve_source_path(source_name: str) -> Optional[str]:
    """
    Resolve a citation source name to an absolute path on disk.
    Search order: data/, workspace/, project root, then absolute path.
    """
    if not source_name:
        return None
    name = source_name.strip()
    if not name:
        return None

    candidates: List[str] = []
    if os.path.isabs(name):
        candidates.append(name)
    else:
        # Strip accidental leading ./ or data/
        rel = name.lstrip("./")
        for base in (DATA_DIR, WORKSPACE_DIR, ROOT_DIR):
            candidates.append(os.path.join(base, name))
            candidates.append(os.path.join(base, rel))
            candidates.append(os.path.join(base, os.path.basename(rel)))

    seen = set()
    for c in candidates:
        c = os.path.normpath(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    return None


def enrich_citation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Add path/exists/readable fields used by clickable UI panels."""
    out = dict(item or {})
    source = out.get("source") or out.get("path") or ""
    path = resolve_source_path(source)
    out["path"] = path
    out["exists"] = bool(path)
    if path:
        try:
            size = os.path.getsize(path)
            out["size_bytes"] = size
            ext = os.path.splitext(path)[1].lower()
            out["readable"] = ext not in _BINARY_EXTS and size <= 2_000_000
            out["rel_path"] = os.path.relpath(path, ROOT_DIR).replace("\\", "/")
        except OSError:
            out["exists"] = False
            out["readable"] = False
    else:
        out["readable"] = False
    return out


def read_source_content(path: str, max_chars: int = 20000) -> str:
    """Read a resolved source file as text (truncated)."""
    if not path or not os.path.isfile(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    if ext in _BINARY_EXTS:
        return f"*(binary file — download to open: {os.path.basename(path)})*"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars)
        if os.path.getsize(path) > max_chars:
            text += "\n…[truncated]"
        return text
    except OSError as e:
        return f"*(could not read file: {e})*"


def find_snippet_lines(path: str, preview: str, context: int = 2) -> List[str]:
    """
    Locate the citation preview inside the source file and return
    nearby lines (1-indexed) for a focused open-at-match experience.
    """
    if not path or not preview or not os.path.isfile(path):
        return []
    needle = " ".join(preview.split())
    if len(needle) < 12:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return []
    for i, line in enumerate(lines):
        if needle[:80] in " ".join(line.split()) or needle[:40] in line:
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            out = []
            for j in range(start, end):
                marker = "→" if j == i else " "
                out.append(f"{marker} {j + 1}: {lines[j].rstrip()}")
            return out
    return []


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
        rel = c.get("rel_path") or ""
        path_txt = f" · `{rel}`" if rel else ""
        lines.append(f"{i}. `{src}`{score_txt}{path_txt} — {preview}")
    return "\n".join(lines)
