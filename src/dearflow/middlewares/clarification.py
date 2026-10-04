"""Stops the run when the model asks the user a question."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from dearflow.tools.clarification import ASK_CLARIFICATION

_ICONS = {
    "missing_info": "❓",
    "ambiguous_requirement": "🤔",
    "approach_choice": "🔀",
    "risk_confirmation": "⚠️",
    "suggestion": "💡",
}


class ClarificationMiddleware(AgentMiddleware):
    """Turns an `ask_clarification` tool call into "show the question, end the run".

    Two hooks cooperate:

    1. `after_model` runs right after the model answers. If the model called
       `ask_clarification` together with other tools, the other calls are
       removed, so nothing with side effects runs before the user replies.
    2. `wrap_tool_call` intercepts the clarification call itself and returns
       the formatted question as the tool result. Because the tool is
       `return_direct`, the graph then ends instead of calling the model
       again. The user's next message continues the same thread as a new run.
    """

    # --- hook 1: drop sibling tool calls -----------------------------------

    @hook_config(can_jump_to=[])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1] if state["messages"] else None
        if not isinstance(last, AIMessage) or len(last.tool_calls) < 2:
            return None
        clarifications = [c for c in last.tool_calls if c["name"] == ASK_CLARIFICATION]
        if not clarifications:
            return None
        # Same message id => LangGraph replaces the old message with this one.
        return {"messages": [last.model_copy(update={"tool_calls": clarifications[:1]})]}

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    # --- hook 2: end the run with the question -----------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != ASK_CLARIFICATION:
            return handler(request)
        return self._end_with_question(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != ASK_CLARIFICATION:
            return await handler(request)
        return self._end_with_question(request)

    def _end_with_question(self, request: ToolCallRequest) -> ToolMessage:
        call = request.tool_call
        return ToolMessage(
            content=self.format_question(call.get("args") or {}),
            tool_call_id=call["id"],
            name=ASK_CLARIFICATION,
        )

    @staticmethod
    def format_question(args: dict[str, Any]) -> str:
        """Render the tool arguments as text for the user."""
        icon = _ICONS.get(args.get("clarification_type", ""), "❓")
        lines = []
        if args.get("context"):
            lines.append(f"{icon} {args['context']}")
            lines.append(str(args.get("question", "")))
        else:
            lines.append(f"{icon} {args.get('question', '')}")
        for number, option in enumerate(args.get("options") or [], start=1):
            lines.append(f"  {number}. {option}")
        return "\n".join(lines)
