"""SQLite TTL cache for web_search results."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(ROOT_DIR, "data", "web_search_cache.db")

DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours
MAX_ROWS = 2000


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_search_cache (
                query_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                hits INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def _normalize_key(query: str) -> str:
    return " ".join((query or "").lower().split())


def get_cached_search(query: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> Optional[List[Dict[str, Any]]]:
    init_cache()
    key = _normalize_key(query)
    if not key:
        return None
    now = time.time()
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload, created_at FROM web_search_cache WHERE query_key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        if ttl_seconds > 0 and (now - float(row["created_at"])) > ttl_seconds:
            conn.execute("DELETE FROM web_search_cache WHERE query_key = ?", (key,))
            return None
        conn.execute(
            "UPDATE web_search_cache SET hits = hits + 1 WHERE query_key = ?",
            (key,),
        )
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None
        return payload


def put_cached_search(query: str, results: List[Dict[str, Any]]) -> None:
    init_cache()
    key = _normalize_key(query)
    if not key or results is None:
        return
    payload = json.dumps(results, ensure_ascii=False, default=str)
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO web_search_cache (query_key, query, payload, created_at, hits)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT(query_key) DO UPDATE SET
                payload = excluded.payload,
                created_at = excluded.created_at
            """,
            (key, query, payload, now),
        )
        # Evict oldest rows if over cap
        count = conn.execute("SELECT COUNT(*) AS c FROM web_search_cache").fetchone()["c"]
        if count > MAX_ROWS:
            conn.execute(
                """
                DELETE FROM web_search_cache WHERE query_key IN (
                    SELECT query_key FROM web_search_cache
                    ORDER BY created_at ASC LIMIT ?
                )
                """,
                (count - MAX_ROWS,),
            )


def clear_cache() -> int:
    init_cache()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM web_search_cache")
    return cur.rowcount


def cache_stats() -> Dict[str, Any]:
    init_cache()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(hits),0) AS h FROM web_search_cache"
        ).fetchone()
    return {"entries": row["c"], "total_hits": row["h"], "path": DB_PATH}


def format_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No results found."
    formatted = []
    for r in results:
        formatted.append(
            f"Title: {r.get('title', '')}\nLink: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
        )
    return "\n\n".join(formatted)
