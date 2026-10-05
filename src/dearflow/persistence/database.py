"""One SQLite file for thread and run records."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MEMORY = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id  TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT,
    status     TEXT NOT NULL DEFAULT 'idle',      -- idle | busy
    metadata   TEXT NOT NULL DEFAULT '{}',        -- JSON object from the client
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_threads_user ON threads (user_id, updated_at);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    thread_id           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'running',  -- running | success | error | interrupted
    model_name          TEXT,
    error               TEXT,
    first_human_message TEXT,
    last_ai_message     TEXT,
    message_count       INTEGER NOT NULL DEFAULT 0,
    stop_reason         TEXT,                             -- token_capped | NULL
    total_input_tokens  INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    llm_call_count      INTEGER NOT NULL DEFAULT 0,
    lead_agent_tokens   INTEGER NOT NULL DEFAULT 0,
    subagent_tokens     INTEGER NOT NULL DEFAULT 0,
    middleware_tokens   INTEGER NOT NULL DEFAULT 0,
    token_usage_by_model TEXT NOT NULL DEFAULT '{}',  -- JSON: model -> tokens
    idempotency_key     TEXT,
    request_hash        TEXT,
    multitask_strategy  TEXT NOT NULL DEFAULT 'reject',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_runs_thread ON runs (thread_id, created_at);

CREATE TABLE IF NOT EXISTS run_events (
    run_id     TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    event      TEXT NOT NULL,
    data       TEXT NOT NULL,     -- JSON
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Database:
    """A thin, thread-safe wrapper over one sqlite3 connection.

    DeerFlow uses SQLAlchemy with SQLite or Postgres. Two small tables don't
    need an ORM: plain SQL keeps the whole data model readable in `_SCHEMA`.
    """

    def __init__(self, path: str | Path = MEMORY) -> None:
        self.path = str(path)
        if self.path != MEMORY:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != MEMORY:
            self._conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
        self._lock = threading.RLock()
        self._conn.executescript(_SCHEMA)
        self._add_missing_columns()

    # Columns added after the first release: (table, column, definition). Old databases get them on open.
    _LATER_COLUMNS = (
        ("threads", "metadata", "TEXT NOT NULL DEFAULT '{}'"),
        ("runs", "stop_reason", "TEXT"),
        ("runs", "token_usage_by_model", "TEXT NOT NULL DEFAULT '{}'"),
        ("runs", "idempotency_key", "TEXT"),
        ("runs", "request_hash", "TEXT"),
        ("runs", "multitask_strategy", "TEXT NOT NULL DEFAULT 'reject'"),
        *(
            ("runs", name, "INTEGER NOT NULL DEFAULT 0")
            for name in (
                "total_input_tokens",
                "total_output_tokens",
                "total_tokens",
                "llm_call_count",
                "lead_agent_tokens",
                "subagent_tokens",
                "middleware_tokens",
            )
        ),
    )

    def _add_missing_columns(self) -> None:
        """Tiny migration: add any column a database created by an older version lacks."""
        for table, column, definition in self._LATER_COLUMNS:
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_idempotency_key "
            "ON runs (idempotency_key) WHERE idempotency_key IS NOT NULL"
        )
        self._conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        """Run one statement, commit, and return any rows."""
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            rows = cursor.fetchall()  # read first: RETURNING rows must be consumed before commit
            self._conn.commit()
            return rows

    def close(self) -> None:
        with self._lock:
            self._conn.close()
