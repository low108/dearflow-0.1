"""Keeps personal data away from the model."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection
from dataclasses import replace as dataclass_replace

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from dearflow.middlewares.messages import is_real_user_message
from dearflow.pii import Redactor

logger = logging.getLogger(__name__)

MCP_SOURCE = "mcp"
SOURCE_KEY = "dearflow_source"  # metadata tag set on tools that come from MCP servers


class PiiRedactionMiddleware(AgentMiddleware):
    """Rewrites personal data to placeholders at the two places outside text enters.

    1. **What the user types** (`wrap_model_call`). Only the copy of the messages sent to the
       model is changed. The conversation in the database keeps the original, so the user
       still sees what they wrote. Every model call redacts the whole history again; that is
       cheap and gives the same result each time because placeholders depend only on the value.
    2. **What outside tools return** (`wrap_tool_call`): MCP tools and any tool listed in
       `pii_redaction.tool_names`. The model only ever sees the redacted result. Results of local
       tools are left alone: that data came from the user's own setup, not from the internet.

    If redaction itself breaks, the original goes through and a warning is logged. A broken
    pattern must not stop the agent (the same trade-off DeerFlow makes).
    """

    def __init__(self, redactor: Redactor, tool_names: Collection[str] = ()) -> None:
        super().__init__()
        self._redactor = redactor
        self._tool_names = frozenset(tool_names)

    # --- user messages, at the model call -------------------------------------------

    def _redact_request(self, request: ModelRequest) -> ModelRequest:
        messages = list(request.messages)
        changed = False
        for index, message in enumerate(messages):
            if not is_real_user_message(message):
                continue  # hidden middleware messages (date, memory, warnings) are not user text
            content, was_changed = self._redactor.redact_content(message.content)
            if was_changed:
                messages[index] = message.model_copy(update={"content": content})
                changed = True
        return request.override(messages=messages) if changed else request

    def _safe_redact_request(self, request: ModelRequest) -> ModelRequest:
        try:
            return self._redact_request(request)
        except GraphBubbleUp:
            raise
        except Exception:
            logger.warning("PII redaction failed; sending the original request", exc_info=True)
            return request

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        return handler(self._safe_redact_request(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelCallResult:
        return await handler(self._safe_redact_request(request))

    # --- tool results, at the tool call ---------------------------------------------

    def _from_outside(self, request: ToolCallRequest) -> bool:
        if request.tool_call.get("name") in self._tool_names:
            return True
        tool = getattr(request, "tool", None)
        return (getattr(tool, "metadata", None) or {}).get(SOURCE_KEY) == MCP_SOURCE

    def _redact_result(self, result: ToolMessage | Command) -> ToolMessage | Command:
        if isinstance(result, ToolMessage):
            return self._redact_tool_message(result)
        update = getattr(result, "update", None)  # a Command carries its ToolMessages inside
        messages = update.get("messages") if isinstance(update, dict) else None
        if isinstance(messages, list):
            rewritten = [self._redact_tool_message(m) if isinstance(m, ToolMessage) else m for m in messages]
            if rewritten != messages:
                return dataclass_replace(result, update={**update, "messages": rewritten})
        return result

    def _redact_tool_message(self, message: ToolMessage) -> ToolMessage:
        content, changed = self._redactor.redact_content(message.content)
        return message.model_copy(update={"content": content}) if changed else message

    def _safe_redact_result(self, request: ToolCallRequest, result: ToolMessage | Command) -> ToolMessage | Command:
        if not self._from_outside(request):
            return result
        try:
            return self._redact_result(result)
        except Exception:
            logger.warning("PII redaction failed on a tool result; leaving it unchanged", exc_info=True)
            return result

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        return self._safe_redact_result(request, handler(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return self._safe_redact_result(request, await handler(request))
