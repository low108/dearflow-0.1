"""A token limit per run, enforced in code."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config

from dearflow.config import TokenBudgetConfig
from dearflow.middlewares.messages import without_tool_calls
from dearflow.usage import TOKEN_CAPPED, RunUsage, UsageRegistry

logger = logging.getLogger(__name__)

_WARNING = (
    "[TOKEN BUDGET WARNING] You have used {used:,} of your {budget:,} {reason} token budget ({percent:.0f}%). "
    "Wrap up your current work and produce a final answer. Avoid starting new tool calls unless they are essential."
)
_EXCEEDED = (
    "[TOKEN BUDGET EXCEEDED] The {reason} token usage ({used:,}) has exceeded the safety limit ({budget:,}). "
    "Producing final answer with results collected so far."
)


class TokenBudgetMiddleware(AgentMiddleware):
    """Two steps, like DeerFlow's budget:

    1. Soft limit (`warn_threshold`): tell the model to wrap up. The warning is a
       message added to the NEXT model request only. It is not stored in the
       conversation, and it goes after the tool results, so each tool call still
       sits right before its result.
    2. Hard limit (`hard_stop_threshold`): remove the tool calls from the model's
       response. The loop then ends by itself with a final answer; nothing is raised.

    Counting is per run and covers everyone in it: the lead agent, every subagent
    and middleware model calls (see `UsageCallback`). The same middleware instance
    is also given to subagents, so a subagent stops when the whole run is over budget.
    """

    def __init__(self, config: TokenBudgetConfig, registry: UsageRegistry) -> None:
        super().__init__()
        self._config = config
        self._registry = registry

    # --- after each model response: measure, warn or stop -----------------------

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        usage = self._usage()
        messages = state["messages"]
        last = messages[-1] if messages else None
        if usage is None or not isinstance(last, AIMessage) or usage.total_tokens <= 0:
            return None

        reason, used, limit, fraction = self._highest_fraction(usage)
        if fraction >= self._config.hard_stop_threshold:
            usage.stop_reason = TOKEN_CAPPED
            logger.warning("Token budget hard stop for run %s: %s limit reached", usage.run_id, reason)
            note = _EXCEEDED.format(reason=reason, used=used, budget=limit)
            return {"messages": [without_tool_calls(last, note)]}  # same id: replaces the message

        if fraction >= self._config.warn_threshold and not usage.warned:
            usage.warned = True
            usage.pending_warnings.append(_WARNING.format(reason=reason, used=used, budget=limit, percent=fraction * 100))
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    # --- before the next model call: deliver a queued warning ----------------------

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        return handler(self._with_warning(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelCallResult:
        return await handler(self._with_warning(request))

    def _with_warning(self, request: ModelRequest) -> ModelRequest:
        usage = self._usage()
        if usage is None or not usage.pending_warnings:
            return request
        text, usage.pending_warnings = "\n\n".join(usage.pending_warnings), []
        return request.override(messages=[*request.messages, HumanMessage(content=text, name="budget_warning")])

    # --- helpers --------------------------------------------------------------------

    def _usage(self) -> RunUsage | None:
        """This run's counters. None outside `RunService`, where nothing counts tokens."""
        try:
            run_id = (get_config().get("configurable") or {}).get("run_id")
        except RuntimeError:
            return None
        return self._registry.get(run_id)

    def _highest_fraction(self, usage: RunUsage) -> tuple[str, int, int, float]:
        """Which limit (total / input / output) is closest to its cap: (name, used, cap, used/cap)."""
        limits = [("total", usage.total_tokens, self._config.max_tokens)]
        if self._config.max_input_tokens:
            limits.append(("input", usage.input_tokens, self._config.max_input_tokens))
        if self._config.max_output_tokens:
            limits.append(("output", usage.output_tokens, self._config.max_output_tokens))
        return max(((name, used, cap, used / cap) for name, used, cap in limits), key=lambda item: item[3])
