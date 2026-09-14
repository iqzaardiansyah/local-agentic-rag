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
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_session(title: str = "New Chat") -> Dict[str, Any]:
    init_db()
    session_id = str(uuid.uuid4())
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title[:120], ts, ts),
        )
    return {"id": session_id, "title": title[:120], "created_at": ts, "updated_at": ts}


def list_sessions() -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


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


def load_messages(session_id: str) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, steps, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    messages = []
    for r in rows:
        msg = {
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
        }
        if r["steps"]:
            try:
                msg["steps"] = json.loads(r["steps"])
            except json.JSONDecodeError:
                msg["steps"] = []
        messages.append(msg)
    return messages


def save_message(
    session_id: str,
    role: str,
    content: str,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> None:
    init_db()
    ts = _now()
    steps_json = json.dumps(steps, default=str) if steps else None
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, steps, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, steps_json, ts),
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


def ensure_active_session() -> str:
    """Return the newest session id, creating one if the store is empty."""
    init_db()
    sessions = list_sessions()
    if sessions:
        return sessions[0]["id"]
    return create_session("New Chat")["id"]
