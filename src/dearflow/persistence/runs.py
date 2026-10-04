"""Run records: one row per invocation of the agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from dearflow.persistence.database import Database, now

RUNNING = "running"
SUCCESS = "success"
ERROR = "error"
INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    thread_id: str
    user_id: str
    status: str
    model_name: str | None
    error: str | None
    first_human_message: str | None
    last_ai_message: str | None
    message_count: int
    created_at: str
    updated_at: str


def _record(row) -> RunRecord:
    return RunRecord(**{key: row[key] for key in row.keys()})


class RunStore:
    """What happened, per run: status, error, and a preview for list pages."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def start(self, thread_id: str, user_id: str, model_name: str | None = None) -> str:
        run_id = uuid.uuid4().hex
        stamp = now()
        self._db.execute(
            "INSERT INTO runs (run_id, thread_id, user_id, status, model_name, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, thread_id, user_id, RUNNING, model_name, stamp, stamp),
        )
        return run_id

    def finish(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        first_human_message: str | None = None,
        last_ai_message: str | None = None,
        message_count: int = 0,
    ) -> None:
        self._db.execute(
            "UPDATE runs SET status = ?, error = ?, first_human_message = ?, last_ai_message = ?,"
            " message_count = ?, updated_at = ? WHERE run_id = ?",
            (status, error, first_human_message, last_ai_message, message_count, now(), run_id),
        )

    def get(self, run_id: str) -> RunRecord | None:
        rows = self._db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return _record(rows[0]) if rows else None

    def list(self, thread_id: str, limit: int = 50) -> list[RunRecord]:
        """Newest first."""
        rows = self._db.execute(
            "SELECT * FROM runs WHERE thread_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?", (thread_id, limit)
        )
        return [_record(r) for r in rows]

    def interrupt_unfinished(self) -> list[str]:
        """Mark runs left `running` by a crash as interrupted. Returns their thread ids."""
        rows = self._db.execute(
            "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE status = ? RETURNING thread_id",
            (INTERRUPTED, "process stopped before the run finished", now(), RUNNING),
        )
        return sorted({r["thread_id"] for r in rows})
