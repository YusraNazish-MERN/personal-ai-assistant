"""SQLite storage: conversations, messages, long-term memories, memory suggestions."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT 'New chat',
    summary TEXT NOT NULL DEFAULT '',
    summarized_upto INTEGER NOT NULL DEFAULT 0,
    recap TEXT NOT NULL DEFAULT '',
    recap_upto INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    source TEXT NOT NULL DEFAULT 'manual',
    previous_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    memory_id INTEGER,
    text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    evidence TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect():
    """Open the database, commit on success, always close."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- conversations
def create_conversation(title="New chat"):
    now = now_iso()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        return cur.lastrowid


def list_conversations(limit=20):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def latest_conversation_id():
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM conversations ORDER BY updated_at DESC, id DESC LIMIT 1"
        ).fetchone()
    return row["id"] if row else None


def get_conversation(conv_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return dict(row) if row else None


def rename_conversation(conv_id, title):
    with _connect() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))


def delete_conversation(conv_id):
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def set_summary(conv_id, summary, upto_message_id):
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET summary = ?, summarized_upto = ? WHERE id = ?",
            (summary, upto_message_id, conv_id),
        )


def set_recap(conv_id, recap, upto_message_id):
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET recap = ?, recap_upto = ? WHERE id = ?",
            (recap, upto_message_id, conv_id),
        )


def latest_recap_excluding(conv_id):
    """Most recent OTHER conversation that has a recap (used for continuity across chats)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, recap, updated_at FROM conversations "
            "WHERE id != ? AND recap != '' ORDER BY updated_at DESC, id DESC LIMIT 1",
            (conv_id,),
        ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------- messages
def add_message(conv_id, role, content):
    now = now_iso()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        return cur.lastrowid


def get_messages(conv_id, after_id=0):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? AND id > ? ORDER BY id",
            (conv_id, after_id),
        ).fetchall()
    return [dict(r) for r in rows]


def count_messages(conv_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conv_id,)
        ).fetchone()
    return row["n"]


# ----------------------------------------------------------------------- memories
def add_memory(text, category="other", source="manual", when=None):
    when = when or now_iso()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO memories (text, category, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, category, source, when, when),
        )
        return cur.lastrowid


def list_memories():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC, id DESC").fetchall()
    return [dict(r) for r in rows]


def get_memory(memory_id):
    if memory_id is None:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return dict(row) if row else None


def update_memory(memory_id, text=None, category=None):
    old = get_memory(memory_id)
    if not old:
        return False
    new_text = text if text is not None else old["text"]
    new_category = category if category is not None else old["category"]
    previous = old["text"] if new_text != old["text"] else old["previous_text"]
    with _connect() as conn:
        conn.execute(
            "UPDATE memories SET text = ?, category = ?, previous_text = ?, updated_at = ? "
            "WHERE id = ?",
            (new_text, new_category, previous, now_iso(), memory_id),
        )
    return True


def touch_memory(memory_id):
    """Mark a memory as 'still true' (refreshes its date)."""
    with _connect() as conn:
        conn.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (now_iso(), memory_id))


def delete_memory(memory_id):
    with _connect() as conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------- proposals
def _norm(text):
    return " ".join((text or "").lower().split())


def add_proposal(action, text, category, memory_id, evidence, reason):
    """Queue a memory suggestion. Returns its id, or None if an identical one is already queued."""
    with _connect() as conn:
        existing = conn.execute("SELECT * FROM proposals").fetchall()
        for row in existing:
            if (
                row["action"] == action
                and row["memory_id"] == memory_id
                and _norm(row["text"]) == _norm(text)
            ):
                return None
        cur = conn.execute(
            "INSERT INTO proposals (action, memory_id, text, category, evidence, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (action, memory_id, text, category, evidence, reason, now_iso()),
        )
        return cur.lastrowid


def list_proposals():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM proposals ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_proposal(proposal_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    return dict(row) if row else None


def delete_proposal(proposal_id):
    with _connect() as conn:
        conn.execute("DELETE FROM proposals WHERE id = ?", (proposal_id,))


# ------------------------------------------------------------------------- export
def export_all():
    return {
        "exported_at": now_iso(),
        "memories": list_memories(),
        "pending_suggestions": list_proposals(),
    }
