"""
Fingerprint data/ (name, size, mtime) so the KB can detect stale indexes.
No extra dependency; optional auto-reindex.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FINGERPRINT_PATH = os.path.join(DATA_DIR, ".index_fingerprint.json")

# Files that are runtime/DB artifacts and should not force a reindex.
_SKIP_NAMES = {
    "chat_sessions.db",
    "web_search_cache.db",
    "knowledge_graph.json",
    "eval_set.json",
    ".index_fingerprint.json",
}


def _should_include(name: str) -> bool:
    if name.startswith("."):
        return False
    if name in _SKIP_NAMES:
        return False
    if name.endswith((".db", ".sqlite", ".pyc", ".bin")):
        return False
    return True


def compute_fingerprint(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return a stable fingerprint of indexable files under data/."""
    root = data_dir or DATA_DIR
    entries: List[str] = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if not _should_include(name):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            try:
                st = os.stat(path)
                entries.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                continue
    blob = "\n".join(entries)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {
        "digest": digest,
        "file_count": len(entries),
        "files": [e.split(":", 1)[0] for e in entries],
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_stored_fingerprint(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    fp = path or FINGERPRINT_PATH
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("digest"):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def save_fingerprint(fp: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Dict[str, Any]:
    """Persist the current (or provided) fingerprint after a successful reindex."""
    out_path = path or FINGERPRINT_PATH
    data = fp or compute_fingerprint()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def check_freshness(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare current data/ fingerprint to the last successful index.

    Returns:
      {
        fresh: bool,
        changed: bool,
        current: {...},
        stored: {...} | None,
        reason: str,
      }
    """
    current = compute_fingerprint(data_dir)
    stored = load_stored_fingerprint()
    if stored is None:
        return {
            "fresh": False,
            "changed": True,
            "current": current,
            "stored": None,
            "reason": "no stored index fingerprint (never re-indexed?)",
        }
    if stored.get("digest") == current.get("digest"):
        return {
            "fresh": True,
            "changed": False,
            "current": current,
            "stored": stored,
            "reason": "in sync",
        }
    return {
        "fresh": False,
        "changed": True,
        "current": current,
        "stored": stored,
        "reason": (
            f"data/ changed since last index "
            f"({stored.get('file_count', '?')} → {current.get('file_count', '?')} files)"
        ),
    }


def mark_indexed(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Call after a successful reindex/ingest so freshness checks pass."""
    return save_fingerprint(compute_fingerprint(data_dir))


def auto_reindex_if_stale(force: bool = False) -> Dict[str, Any]:
    """
    If data/ is stale (or force=True), rebuild Chroma+BM25 and save fingerprint.

    Safe to call often; no-op when already fresh.
    """
    status = check_freshness()
    if status["fresh"] and not force:
        return {"reindexed": False, "reason": status["reason"], "freshness": status}

    from src.rag.vectorstore import reindex_all_data

    try:
        chunks = reindex_all_data()
    except Exception as e:
        return {
            "reindexed": False,
            "reason": f"reindex failed: {e}",
            "freshness": status,
            "error": str(e),
        }

    # reindex_all_data already invalidates BM25; record fingerprint
    fp = mark_indexed()
    return {
        "reindexed": True,
        "chunks": chunks,
        "reason": "reindexed because data/ changed" if status["changed"] else "forced reindex",
        "fingerprint": fp,
        "freshness": check_freshness(),
    }


def format_freshness(status: Dict[str, Any]) -> str:
    if status.get("fresh"):
        cur = status.get("current") or {}
        return f"FRESH · {cur.get('file_count', 0)} file(s) indexed"
    return f"STALE · {status.get('reason', 'unknown')}"
