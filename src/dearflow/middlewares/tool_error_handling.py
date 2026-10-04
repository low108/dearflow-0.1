from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command


class ToolErrorHandlingMiddleware(AgentMiddleware):
    """Convert ordinary tool failures into model-visible error messages."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._error_message(request, exc)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._error_message(request, exc)

    @staticmethod
    def _error_message(request: ToolCallRequest, exc: Exception) -> ToolMessage:
        tool_name = str(request.tool_call.get("name") or "unknown")
        tool_call_id = str(request.tool_call.get("id") or "")
        return ToolMessage(
            content=f"Tool {tool_name!r} failed: {exc}",
            tool_call_id=tool_call_id,
            status="error",
        )
