"""Request and response shapes. Field names follow DeerFlow's gateway (LangGraph-compatible)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

from dearflow.persistence import RunRecord, ThreadRecord
from dearflow.streaming import UnsupportedStreamModeError, normalize_modes

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"  # same rule as DeerFlow
ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN, description="1-64 letters, digits, '-' or '_'")]

# `context` keys the client may set per run; anything else is ignored.
CONTEXT_KEYS = ("model_name", "is_plan_mode", "subagent_enabled")


class RunCreateRequest(BaseModel):
    """What `POST /runs`, `/runs/stream` and `/runs/wait` accept.

    Like DeerFlow, unknown fields are errors, and options we don't support are
    rejected by name instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    assistant_id: str | None = None
    input: dict[str, Any] = Field(description='{"messages": [{"role": "user", "content": "..."}]}')
    metadata: dict[str, Any] | None = None
    context: dict[str, Any] | None = Field(default=None, description=f"per-run overrides: {', '.join(CONTEXT_KEYS)}")
    stream_mode: list[str] | str | None = None
    on_disconnect: Literal["cancel", "continue"] = "cancel"
    multitask_strategy: str = "reject"

    @field_validator("multitask_strategy")
    @classmethod
    def _only_reject(cls, value: str) -> str:
        if value not in ("reject", "rollback"):
            raise PydanticCustomError(
                "unsupported_run_option",
                "Run option 'multitask_strategy' only supports 'reject' or 'rollback'",
            )
        return value

    @field_validator("stream_mode")
    @classmethod
    def _known_modes(cls, value: list[str] | str | None) -> list[str] | str | None:
        try:
            normalize_modes(value)
        except UnsupportedStreamModeError as exc:
            raise PydanticCustomError("unsupported_stream_mode", "Unsupported stream mode(s): {modes}", {"modes": ", ".join(exc.modes)}) from exc
        return value

    @field_validator("input")
    @classmethod
    def _has_a_user_message(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not user_text(value):
            raise PydanticCustomError("invalid_input", "input.messages must end with a user message that has text")
        return value

    @property
    def message(self) -> str:
        return user_text(self.input)

    @property
    def overrides(self) -> dict[str, Any]:
        return {k: v for k, v in (self.context or {}).items() if k in CONTEXT_KEYS}


def user_text(payload: dict[str, Any]) -> str:
    """Text of the last message in `input.messages` if it is a user message, else ''."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return ""
    last = messages[-1]
    if last.get("role", last.get("type")) not in ("user", "human"):
        return ""
    content = last.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):  # [{"type": "text", "text": "..."}]
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    return ""


class RunResponse(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str | None = None
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    multitask_strategy: str = "reject"
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    llm_call_count: int = 0
    lead_agent_tokens: int = 0
    subagent_tokens: int = 0
    middleware_tokens: int = 0
    stop_reason: str | None = None  # token_capped | interrupted | error
    error: str | None = None

    @classmethod
    def from_record(cls, record: RunRecord) -> RunResponse:
        return cls(
            run_id=record.run_id,
            thread_id=record.thread_id,
            status=record.status,
            multitask_strategy=record.multitask_strategy,
            created_at=record.created_at,
            updated_at=record.updated_at,
            message_count=record.message_count,
            total_input_tokens=record.total_input_tokens,
            total_output_tokens=record.total_output_tokens,
            total_tokens=record.total_tokens,
            llm_call_count=record.llm_call_count,
            lead_agent_tokens=record.lead_agent_tokens,
            subagent_tokens=record.subagent_tokens,
            middleware_tokens=record.middleware_tokens,
            stop_reason=record.stop_reason or (record.status if record.status in ("interrupted", "error") else None),
            error=record.error,
        )


class ThreadCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN)
    assistant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] = Field(default_factory=dict, description="exact-match filter")
    status: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class ThreadResponse(BaseModel):
    thread_id: str
    status: str = "idle"
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, Any] = Field(default_factory=dict)
    interrupts: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_record(cls, record: ThreadRecord, values: dict[str, Any] | None = None) -> ThreadResponse:
        shown = dict(values or {})
        if record.title:
            shown.setdefault("title", record.title)
        return cls(
            thread_id=record.thread_id,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata=record.metadata,
            values=shown,
        )


__all__ = ["RunCreateRequest", "RunResponse", "ThreadCreateRequest", "ThreadId", "ThreadResponse", "ThreadSearchRequest"]
