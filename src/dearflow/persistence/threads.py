"""Thread records: one row per conversation."""

from __future__ import annotations

from dataclasses import dataclass

from dearflow.persistence.database import Database, now

IDLE = "idle"
BUSY = "busy"


@dataclass(frozen=True)
class ThreadRecord:
    thread_id: str
    user_id: str
    title: str | None
    status: str
    created_at: str
    updated_at: str


def _record(row) -> ThreadRecord:
    return ThreadRecord(**{key: row[key] for key in row.keys()})


class ThreadStore:
    """Metadata about conversations. The messages live in the checkpointer."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def ensure(self, thread_id: str, user_id: str) -> ThreadRecord:
        """Create the thread on first use. Threads belong to the user who made them."""
        stamp = now()
        self._db.execute(
            "INSERT OR IGNORE INTO threads (thread_id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, user_id, stamp, stamp),
        )
        return self.get(thread_id)  # type: ignore[return-value]

    def get(self, thread_id: str) -> ThreadRecord | None:
        rows = self._db.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,))
        return _record(rows[0]) if rows else None

    def list(self, user_id: str, limit: int = 50) -> list[ThreadRecord]:
        """A user's threads, most recently active first."""
        rows = self._db.execute(
            "SELECT * FROM threads WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)
        )
        return [_record(r) for r in rows]

    def set_title(self, thread_id: str, title: str) -> None:
        self._db.execute("UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?", (title, now(), thread_id))

    def set_status(self, thread_id: str, status: str) -> None:
        self._db.execute("UPDATE threads SET status = ?, updated_at = ? WHERE thread_id = ?", (status, now(), thread_id))

    def claim(self, thread_id: str) -> bool:
        """Mark the thread busy. False if it already was: only one run per thread."""
        rows = self._db.execute(
            "UPDATE threads SET status = ?, updated_at = ? WHERE thread_id = ? AND status = ? RETURNING thread_id",
            (BUSY, now(), thread_id, IDLE),
        )
        return bool(rows)

    def delete(self, thread_id: str) -> None:
        self._db.execute("DELETE FROM runs WHERE thread_id = ?", (thread_id,))
        self._db.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
