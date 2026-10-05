"""Detect repeated tool calls before they exhaust LangGraph's recursion limit."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config

from dearflow.config import LoopDetectionConfig
from dearflow.middlewares.messages import is_real_user_message, without_tool_calls
from dearflow.usage import LOOP_CAPPED, UsageRegistry

_WARNING = (
    "[LOOP DETECTED] You are repeating the same tool calls. Stop repeating the call, "
    "use the results already collected, and produce a final answer. If the task cannot "
    "be completed, explain the blocker."
)
_FORCED_STOP = (
    "[FORCED STOP] Repeated tool calls exceeded the safety limit. "
    "No further tool call was executed; answer using the results already collected."
)


def _call_signature(message: AIMessage) -> str | None:
    if not message.tool_calls:
        return None
    calls = []
    for call in message.tool_calls:
        if not isinstance(call, dict):
            continue
        calls.append(
            {
                "name": call.get("name", ""),
                "args": call.get("args", {}),
            }
        )
    if not calls:
        return None
    return json.dumps(
        sorted(calls, key=lambda item: json.dumps(item, sort_keys=True, default=str)),
        sort_keys=True,
        default=str,
    )


def _repetition_count(messages: list, *, window_size: int) -> int:
    """Count the identical trailing tool-call sets in the current user turn."""
    last_user = -1
    for index, message in enumerate(messages):
        if is_real_user_message(message):
            last_user = index
    signatures = [
        signature
        for message in messages[last_user + 1 :]
        if isinstance(message, AIMessage)
        if (signature := _call_signature(message)) is not None
    ][-window_size:]
    if not signatures:
        return 0
    latest = signatures[-1]
    count = 0
    for signature in reversed(signatures):
        if signature != latest:
            break
        count += 1
    return count


class LoopDetectionMiddleware(AgentMiddleware):
    """Warn the model when its latest tool-call set is repeating."""

    def __init__(self, config: LoopDetectionConfig, usage: UsageRegistry | None = None) -> None:
        super().__init__()
        self._config = config
        self._usage = usage

    def _with_warning(self, request: ModelRequest) -> ModelRequest:
        count = _repetition_count(list(request.messages), window_size=self._config.window_size)
        if count != self._config.warn_threshold:
            return request
        warning = HumanMessage(
            content=_WARNING,
            name="loop_warning",
            additional_kwargs={"hide_from_ui": True, "dearflow_injected": "loop_warning"},
        )
        return request.override(messages=[*request.messages, warning])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._with_warning(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._with_warning(request))

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return None
        count = _repetition_count(messages, window_size=self._config.window_size)
        if count >= self._config.hard_limit:
            if self._usage is not None:
                try:
                    run_id = (get_config().get("configurable") or {}).get("run_id")
                except RuntimeError:
                    run_id = None
                usage = self._usage.get(run_id)
                if usage is not None:
                    usage.stop_reason = LOOP_CAPPED
            return {"messages": [without_tool_calls(last, _FORCED_STOP)]}
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
