"""Counts tokens for one run."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

LEAD = "lead_agent"
SUBAGENT = "subagent"
MIDDLEWARE = "middleware"  # model calls made by middleware: title, summarization...
CALLER_KEY = "dearflow_caller"  # metadata tag; the subagent executor sets it to SUBAGENT
TOKEN_CAPPED = "token_capped"


@dataclass
class RunUsage:
    """Everything the token features need to know about ONE run.

    Thread-safe: parallel subagents add to it at the same time.
    """

    run_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    by_caller: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stop_reason: str | None = None  # TOKEN_CAPPED when the budget stopped the run
    warned: bool = False
    pending_warnings: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, caller: str, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.by_caller[caller] += input_tokens + output_tokens
            self.by_model[model] += input_tokens + output_tokens

    def snapshot(self) -> dict[str, Any]:
        """A plain dict, for events and HTTP responses."""
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "llm_calls": self.llm_calls,
                "lead_agent_tokens": self.by_caller[LEAD],
                "subagent_tokens": self.by_caller[SUBAGENT],
                "middleware_tokens": self.by_caller[MIDDLEWARE],
                "by_model": dict(self.by_model),
                "stop_reason": self.stop_reason,
            }


class UsageRegistry:
    """Finds a run's `RunUsage` by run id.

    The service opens an entry when a run starts; the budget middleware, which
    only knows the run id from its config, looks it up. (DeerFlow keys its
    budget state by run id inside the middleware for the same reason.)
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunUsage] = {}
        self._lock = threading.Lock()

    def open(self, run_id: str) -> RunUsage:
        with self._lock:
            return self._runs.setdefault(run_id, RunUsage(run_id))

    def get(self, run_id: str | None) -> RunUsage | None:
        with self._lock:
            return self._runs.get(run_id) if run_id else None

    def close(self, run_id: str) -> RunUsage | None:
        with self._lock:
            return self._runs.pop(run_id, None)


class UsageCallback(BaseCallbackHandler):
    """Adds every finished model call to a `RunUsage`.

    One instance on the run's config sees ALL model calls of the run: the lead
    agent, subagents (callbacks flow into nested agents) and middleware.
    Who made the call comes from metadata: subagents are tagged by the executor,
    calls from the `model` graph node are the lead agent's, the rest are middleware.
    """

    def __init__(self, usage: RunUsage) -> None:
        super().__init__()
        self._usage = usage
        self._started: dict[UUID, tuple[str, str]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        metadata = metadata or {}
        if metadata.get(CALLER_KEY):
            caller = metadata[CALLER_KEY]
        else:
            caller = LEAD if metadata.get("langgraph_node") == "model" else MIDDLEWARE
        self._started[run_id] = (caller, metadata.get("ls_model_name") or "unknown")

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        caller, model = self._started.pop(run_id, (MIDDLEWARE, "unknown"))
        input_tokens = output_tokens = 0
        for generation in response.generations:
            for item in generation:
                usage = getattr(getattr(item, "message", None), "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0) or 0
                output_tokens += usage.get("output_tokens", 0) or 0
                model = getattr(item.message, "response_metadata", {}).get("model_name") or model
        self._usage.add(caller, model, input_tokens, output_tokens)
        if input_tokens or output_tokens:
            logger.info("LLM token usage (%s): input=%s output=%s", caller, input_tokens, output_tokens)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._started.pop(run_id, None)
