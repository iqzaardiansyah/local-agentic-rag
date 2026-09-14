"""SQLite-backed chat session persistence. Free, local, no external services."""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(ROOT_DIR, "data", "chat_sessions.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                steps TEXT,
                citations TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
        )
        # Lightweight migration for DBs created before the citations column existed.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
        if "citations" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN citations TEXT")

        session_cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
        if "pinned" not in session_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_session(title: str = "New Chat") -> Dict[str, Any]:
    init_db()
    session_id = str(uuid.uuid4())
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, pinned) VALUES (?, ?, ?, ?, 0)",
            (session_id, title[:120], ts, ts),
        )
    return {
        "id": session_id,
        "title": title[:120],
        "created_at": ts,
        "updated_at": ts,
        "pinned": 0,
    }


def list_sessions() -> List[Dict[str, Any]]:
    """List sessions: pinned first, then most recently updated."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, COALESCE(pinned, 0) AS pinned "
            "FROM sessions ORDER BY pinned DESC, updated_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["pinned"] = bool(d.get("pinned"))
        out.append(d)
    return out


def set_session_pinned(session_id: str, pinned: bool = True) -> bool:
    """Pin or unpin a session (favorites rise to the top of the picker)."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE sessions SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, session_id),
        )
    return cur.rowcount > 0


def toggle_session_pinned(session_id: str) -> bool:
    """Toggle pin state; returns the new pinned value."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(pinned, 0) AS pinned FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return False
        new_val = 0 if row["pinned"] else 1
        conn.execute(
            "UPDATE sessions SET pinned = ? WHERE id = ?",
            (new_val, session_id),
        )
    return bool(new_val)


def rename_session(session_id: str, title: str) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title[:120], _now(), session_id),
        )
    return cur.rowcount > 0


def delete_session(session_id: str) -> bool:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return cur.rowcount > 0


def clear_messages(session_id: str) -> int:
    """Delete all messages in a session; returns how many rows were removed."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
    return cur.rowcount


def load_messages(session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load messages for a session in chronological order.

    limit: if set, return only the most recent N messages (still chronological).
    Each message includes an integer `id` (DB row id) for fork/truncate ops.
    """
    init_db()
    with _connect() as conn:
        if limit and limit > 0:
            rows = conn.execute(
                "SELECT id, role, content, steps, citations, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, int(limit)),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                "SELECT id, role, content, steps, citations, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
    messages = []
    for r in rows:
        msg = {
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
        }
        if r["steps"]:
            try:
                msg["steps"] = json.loads(r["steps"])
            except json.JSONDecodeError:
                msg["steps"] = []
        if r["citations"]:
            try:
                msg["citations"] = json.loads(r["citations"])
            except json.JSONDecodeError:
                msg["citations"] = []
        messages.append(msg)
    return messages


def truncate_after_message(session_id: str, message_id: int) -> int:
    """
    Delete every message in the session with id >= message_id (inclusive).
    Used for regenerate (drop the assistant reply) and fork-from-point.
    Returns number of deleted rows.
    """
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id >= ?",
            (session_id, int(message_id)),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
    return cur.rowcount


def delete_last_assistant_reply(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Delete the most recent assistant message in a session (for regenerate).
    Returns the deleted message dict, or None if there was no assistant reply.
    """
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, role, content, steps, citations, created_at FROM messages "
            "WHERE session_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
    deleted = {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
    if row["steps"]:
        try:
            deleted["steps"] = json.loads(row["steps"])
        except json.JSONDecodeError:
            deleted["steps"] = []
    return deleted


def last_user_message(session_id: str) -> Optional[str]:
    """Return the content of the most recent user message, if any."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    return row["content"] if row else None


def fork_session(
    source_session_id: str,
    up_to_message_id: Optional[int] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new session containing a copy of messages from source_session_id.

    up_to_message_id: if set, copy only messages with id <= this row id
                      (exclusive fork-at-point: keep history up to that message).
    Title defaults to "<original title> (fork)".
    """
    init_db()
    with _connect() as conn:
        src = conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (source_session_id,)
        ).fetchone()
        if not src:
            raise ValueError(f"Source session not found: {source_session_id}")

        if up_to_message_id is None:
            rows = conn.execute(
                "SELECT role, content, steps, citations, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (source_session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content, steps, citations, created_at FROM messages "
                "WHERE session_id = ? AND id <= ? ORDER BY id ASC",
                (source_session_id, int(up_to_message_id)),
            ).fetchall()

        fork_id = str(uuid.uuid4())
        ts = _now()
        fork_title = (title or f"{src['title']} (fork)")[:120]
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, pinned) VALUES (?, ?, ?, ?, 0)",
            (fork_id, fork_title, ts, ts),
        )
        for r in rows:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, steps, citations, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fork_id,
                    r["role"],
                    r["content"],
                    r["steps"],
                    r["citations"],
                    r["created_at"],
                ),
            )

    return {
        "id": fork_id,
        "title": fork_title,
        "created_at": ts,
        "updated_at": ts,
        "message_count": len(rows),
    }


def save_message(
    session_id: str,
    role: str,
    content: str,
    steps: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    init_db()
    ts = _now()
    steps_json = json.dumps(steps, default=str) if steps else None
    citations_json = json.dumps(citations, default=str) if citations else None
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, steps, citations, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, steps_json, citations_json, ts),
        )
        # Auto-title from first user message
        if role == "user":
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id = ? AND role = 'user'",
                (session_id,),
            ).fetchone()["c"]
            if count == 1:
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (content[:80].replace("\n", " "), ts, session_id),
                )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (ts, session_id)
        )


def export_session_markdown(session_id: str) -> str:
    """Export a full chat session as a Markdown document."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return "# Chat Session\n\n*Session not found.*"

    messages = load_messages(session_id)
    lines = [
        f"# {row['title']}",
        "",
        f"- **Session ID:** `{session_id}`",
        f"- **Created:** {row['created_at']}",
        f"- **Updated:** {row['updated_at']}",
        f"- **Messages:** {len(messages)}",
        "",
        "---",
        "",
    ]
    for m in messages:
        role = "🧑 User" if m["role"] == "user" else "🤖 Assistant"
        lines.append(f"## {role}")
        lines.append("")
        lines.append(m.get("content") or "*(no text)*")
        lines.append("")
        citations = m.get("citations") or []
        if citations:
            lines.append("**Sources:**")
            for c in citations:
                score = c.get("score")
                score_txt = f" (score {score:.3f})" if isinstance(score, (int, float)) else ""
                path = c.get("path")
                path_txt = f" — `{path}`" if path else ""
                lines.append(f"- `{c.get('source', 'Unknown')}`{score_txt}{path_txt}")
            lines.append("")
        steps = m.get("steps") or []
        if steps:
            lines.append("<details>")
            lines.append("<summary>Tool trace</summary>")
            lines.append("")
            for step in steps:
                if step.get("type") == "call":
                    lines.append(f"- Called `{step.get('tool')}`")
                    args = step.get("args")
                    if args:
                        lines.append(f"  ```json\n  {json.dumps(args, indent=2, default=str)}\n  ```")
                elif step.get("type") == "result":
                    out = (step.get("output") or "")[:500]
                    lines.append(f"- Result from `{step.get('tool')}`:")
                    lines.append(f"  ```\n  {out}\n  ```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    lines.append("---")
    lines.append(f"*Exported locally from Local Agentic RAG on {datetime.now().isoformat(timespec='seconds')}*")
    return "\n".join(lines)


def ensure_session(session_id: str, title: str = "API Chat") -> str:
    """
    Ensure a session row exists for the given id (creating it if needed).
    Returns the same session_id. Useful when clients supply their own ids.
    """
    if not session_id:
        return create_session(title)["id"]
    init_db()
    ts = _now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, pinned) VALUES (?, ?, ?, ?, 0)",
                (session_id, title[:120], ts, ts),
            )
    return session_id


def ensure_active_session() -> str:
    """Return the newest session id, creating one if the store is empty."""
    init_db()
    sessions = list_sessions()
    if sessions:
        return sessions[0]["id"]
    return create_session("New Chat")["id"]


def search_messages(
    query: str,
    limit: int = 30,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search chat message content across sessions (case-insensitive substring).

    Returns hits with session title, role, snippet, and created_at.
    Optionally restrict to one session_id.
    """
    init_db()
    if not query or not query.strip():
        return []
    q = f"%{query.strip()}%"
    sql = (
        "SELECT m.session_id AS session_id, s.title AS session_title, "
        "m.role AS role, m.content AS content, m.created_at AS created_at "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE m.content LIKE ? COLLATE NOCASE"
    )
    params: List[Any] = [q]
    if session_id:
        sql += " AND m.session_id = ?"
        params.append(session_id)
    sql += " ORDER BY m.created_at DESC LIMIT ?"
    params.append(max(1, min(int(limit), 100)))

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    hits: List[Dict[str, Any]] = []
    needle = query.strip().lower()
    for r in rows:
        content = r["content"] or ""
        low = content.lower()
        idx = low.find(needle)
        if idx < 0:
            snippet = content[:160]
        else:
            start = max(0, idx - 40)
            end = min(len(content), idx + len(needle) + 80)
            snippet = ("…" if start > 0 else "") + content[start:end] + ("…" if end < len(content) else "")
        snippet = " ".join(snippet.split())
        hits.append({
            "session_id": r["session_id"],
            "session_title": r["session_title"],
            "role": r["role"],
            "snippet": snippet,
            "created_at": r["created_at"],
        })
    return hits
