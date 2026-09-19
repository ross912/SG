"""SQLite-backed chat history for the single-user dashboard."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from config import DATA_DIR


CHAT_DB_PATH = DATA_DIR / "intelligence" / "chat_history.sqlite3"
RETENTION_DAYS = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _connect() -> sqlite3.Connection:
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CHAT_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_updated
            ON conversations(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id);
        """
    )
    return connection


def cleanup_expired(now: datetime | None = None) -> int:
    cutoff = (now or _now()) - timedelta(days=RETENTION_DAYS)
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE updated_at < ?", (cutoff.isoformat(),),
        )
        return cursor.rowcount


def create_conversation(first_question: str) -> str:
    cleanup_expired()
    conversation_id = uuid.uuid4().hex
    timestamp = _now().isoformat()
    title = " ".join(first_question.split())[:42] or "新对话"
    with _connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title, timestamp, timestamp),
        )
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    cleanup_expired()
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,),
        ).fetchone()
    return row is not None


def add_message(conversation_id: str, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("unsupported chat role")
    timestamp = _now().isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )
        if cursor.rowcount != 1:
            raise KeyError("conversation not found")
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, timestamp),
        )


def get_messages(conversation_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    cleanup_expired()
    query = (
        "SELECT id, role, content, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id"
    )
    parameters: tuple[Any, ...] = (conversation_id,)
    if limit is not None:
        query = (
            "SELECT id, role, content, created_at FROM ("
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?"
            ") ORDER BY id"
        )
        parameters = (conversation_id, max(1, min(int(limit), 100)))
    with _connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def list_conversations(limit: int = 100) -> list[dict[str, Any]]:
    cleanup_expired()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count,
                   COALESCE((
                       SELECT content FROM messages latest
                       WHERE latest.conversation_id = c.id
                       ORDER BY latest.id DESC LIMIT 1
                   ), '') AS preview
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,),
        )
        return cursor.rowcount == 1
