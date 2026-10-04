"""Stamps a receipt on every tool result and shows the receipt list to the model."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from dearflow.middlewares.messages import injected_message, is_real_user_message
from dearflow.receipts import LEDGER_KEY, RECEIPT_KEY, Receipt, extract_receipts, make_receipt, render_ledger

logger = logging.getLogger(__name__)

LEDGER_CONTEXT = "receipt_ledger"  # kind of the hidden message that carries the ledger text
DELEGATION_TOOL = "task"


class ToolReceiptMiddleware(AgentMiddleware):
    """Zero-model provenance for every tool call.

    - `wrap_tool_call` stamps a receipt into the result message. The key is owned by this
      code: it is always overwritten, so a tool cannot forge its own "evidence".
    - `wrap_model_call` adds the receipt list to the END of the request (a hidden message, never
      stored in the conversation) and stamps that same list on the model's reply. Later, a
      citation in that reply is checked against this stamped list.

    Place it OUTERMOST in the chain. Inner layers can turn an exception into an error message
    or replace a result; a receipt layer inside them would miss those calls.

    `render_mode`: "always" shows the list on every model call (subagents: they write the
    citations, so they must see it). "delegation_only" shows it only while the model is reading
    a subagent's result in the current turn (lead agent: avoids a token cost on ordinary turns).
    """

    def __init__(self, *, render_mode: str = "always") -> None:
        super().__init__()
        if render_mode not in ("always", "delegation_only"):
            raise ValueError(f"Unknown render_mode: {render_mode}")
        self._render_mode = render_mode

    # --- tool results: stamp -------------------------------------------------------------

    @staticmethod
    def _stamp_message(message: ToolMessage, request: ToolCallRequest) -> None:
        try:
            message.additional_kwargs = {**message.additional_kwargs, RECEIPT_KEY: make_receipt(request.tool_call, message)}
        except Exception:
            # Never block a tool call, but say so: a silent gap makes citations lie.
            logger.warning("Failed to stamp tool receipt", exc_info=True)

    def _stamp(self, result: ToolMessage | Command, request: ToolCallRequest) -> ToolMessage | Command:
        if isinstance(result, ToolMessage):
            self._stamp_message(result, request)
        else:  # a Command carries its ToolMessage inside `update`
            update = getattr(result, "update", None)
            messages = update.get("messages", []) if isinstance(update, dict) else []
            call_id = str(request.tool_call.get("id") or "")
            for message in [messages] if isinstance(messages, ToolMessage) else messages:
                if isinstance(message, ToolMessage) and str(message.tool_call_id) == call_id:
                    self._stamp_message(message, request)
        return result

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        return self._stamp(handler(request), request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return self._stamp(await handler(request), request)

    # --- model calls: show, then stamp the reply ---------------------------------------------

    def _should_render(self, request: ModelRequest) -> bool:
        if self._render_mode == "always":
            return True
        messages = list(request.messages)
        turn_start = max((i for i, m in enumerate(messages) if is_real_user_message(m)), default=-1) + 1
        return any(isinstance(m, ToolMessage) and m.name == DELEGATION_TOOL for m in messages[turn_start:])

    def _prepare(self, request: ModelRequest) -> tuple[ModelRequest, list[Receipt] | None]:
        if not self._should_render(request):
            return request, None
        text, shown = render_ledger(extract_receipts(list(request.messages)))
        if not text:
            return request, shown
        hidden = injected_message(text, LEDGER_CONTEXT)
        return request.override(messages=[*request.messages, hidden]), shown

    @staticmethod
    def _stamp_shown_list(result: ModelCallResult, shown: list[Receipt] | None) -> ModelCallResult:
        if shown is None:
            return result
        response: Any = result if isinstance(result, AIMessage) else getattr(result, "model_response", result)
        messages = [result] if isinstance(result, AIMessage) else getattr(response, "result", [])
        for message in messages:
            if isinstance(message, AIMessage):
                # Always overwritten: provider output must not be able to forge the list its citations are checked against.
                message.additional_kwargs = {**message.additional_kwargs, LEDGER_KEY: [dict(r) for r in shown]}
        return result

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        prepared, shown = self._prepare(request)
        return self._stamp_shown_list(handler(prepared), shown)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelCallResult:
        prepared, shown = self._prepare(request)
        return self._stamp_shown_list(await handler(prepared), shown)
