"""Run records: one row per invocation of the agent."""

from __future__ import annotations

import json
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
    stop_reason: str | None
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    llm_call_count: int
    lead_agent_tokens: int
    subagent_tokens: int
    middleware_tokens: int
    token_usage_by_model: dict[str, int]
    idempotency_key: str | None
    request_hash: str | None
    multitask_strategy: str
    created_at: str
    updated_at: str


def _record(row) -> RunRecord:
    values = {key: row[key] for key in row.keys()}
    values["token_usage_by_model"] = json.loads(values["token_usage_by_model"] or "{}")
    return RunRecord(**values)


class RunStore:
    """What happened, per run: status, error, and a preview for list pages."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def start(
        self,
        thread_id: str,
        user_id: str,
        model_name: str | None = None,
        *,
        idempotency_key: str | None = None,
        request_hash: str | None = None,
        multitask_strategy: str = "reject",
    ) -> str:
        run_id = uuid.uuid4().hex
        stamp = now()
        self._db.execute(
            "INSERT INTO runs (run_id, thread_id, user_id, status, model_name, idempotency_key, request_hash,"
            " multitask_strategy, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                thread_id,
                user_id,
                RUNNING,
                model_name,
                idempotency_key,
                request_hash,
                multitask_strategy,
                stamp,
                stamp,
            ),
        )
        return run_id

    def get_by_idempotency_key(self, idempotency_key: str) -> RunRecord | None:
        rows = self._db.execute("SELECT * FROM runs WHERE idempotency_key = ?", (idempotency_key,))
        return _record(rows[0]) if rows else None

    def finish(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        first_human_message: str | None = None,
        last_ai_message: str | None = None,
        message_count: int = 0,
        usage: dict | None = None,
    ) -> None:
        """Close a run. `usage` is `RunUsage.snapshot()`; its `stop_reason` is saved with the run."""
        usage = usage or {}
        self._db.execute(
            "UPDATE runs SET status = ?, error = ?, first_human_message = ?, last_ai_message = ?,"
            " message_count = ?, stop_reason = ?, total_input_tokens = ?, total_output_tokens = ?,"
            " total_tokens = ?, llm_call_count = ?, lead_agent_tokens = ?, subagent_tokens = ?,"
            " middleware_tokens = ?, token_usage_by_model = ?, updated_at = ? WHERE run_id = ?",
            (
                status, error, first_human_message, last_ai_message, message_count,
                usage.get("stop_reason"),
                usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("total_tokens", 0),
                usage.get("llm_calls", 0), usage.get("lead_agent_tokens", 0),
                usage.get("subagent_tokens", 0), usage.get("middleware_tokens", 0),
                json.dumps(usage.get("by_model", {})),
                now(), run_id,
            ),
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

    def aggregate_tokens(self, thread_id: str) -> dict:
        """Token totals across all finished runs of a thread (DeerFlow's `token-usage` shape)."""
        runs = [r for r in self.list(thread_id, limit=100_000) if r.status != RUNNING]
        by_model: dict[str, dict[str, int]] = {}
        for run in runs:
            for model, tokens in run.token_usage_by_model.items():
                entry = by_model.setdefault(model, {"tokens": 0, "runs": 0})
                entry["tokens"] += tokens
                entry["runs"] += 1
        return {
            "total_tokens": sum(r.total_tokens for r in runs),
            "total_input_tokens": sum(r.total_input_tokens for r in runs),
            "total_output_tokens": sum(r.total_output_tokens for r in runs),
            "total_runs": len(runs),
            "by_model": by_model,
            "by_caller": {
                "lead_agent": sum(r.lead_agent_tokens for r in runs),
                "subagent": sum(r.subagent_tokens for r in runs),
                "middleware": sum(r.middleware_tokens for r in runs),
            },
        }

    def interrupt_unfinished(self) -> list[str]:
        """Mark runs left `running` by a crash as interrupted. Returns their thread ids."""
        rows = self._db.execute(
            "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE status = ? RETURNING thread_id",
            (INTERRUPTED, "process stopped before the run finished", now(), RUNNING),
        )
        return sorted({r["thread_id"] for r in rows})
