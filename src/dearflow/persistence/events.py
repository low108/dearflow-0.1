"""The run log: what a run emitted, in order."""

from __future__ import annotations

import json
from typing import Any

from dearflow.persistence.database import Database, now


class RunEventStore:
    """Append-only events per run, so a finished run can be replayed or audited.

    Token chunks (`messages`) and full-state snapshots (`values`) are NOT stored:
    they are large and the checkpointer already has the final state. `updates`,
    `custom`, errors and the start/end markers are enough to see what happened.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def append(self, run_id: str, seq: int, event: str, data: Any) -> None:
        self._db.execute(
            "INSERT INTO run_events (run_id, seq, event, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, event, json.dumps(data, ensure_ascii=False, default=str), now()),
        )

    def list(self, run_id: str, after_seq: int = 0) -> list[tuple[int, str, Any]]:
        """`(seq, event, data)` for a run, oldest first. `after_seq` resumes a reader."""
        rows = self._db.execute(
            "SELECT seq, event, data FROM run_events WHERE run_id = ? AND seq > ? ORDER BY seq", (run_id, after_seq)
        )
        return [(r["seq"], r["event"], json.loads(r["data"])) for r in rows]
