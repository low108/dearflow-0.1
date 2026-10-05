"""Plan mode: a todo list the agent must finish."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import TodoListMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage

from dearflow.middlewares.messages import TODO_CONTEXT, TODO_REMINDER, injected_kind, injected_message


class TodoMiddleware(TodoListMiddleware):
    """LangChain's todo middleware plus a "don't quit early" guard.

    The base class gives us the `write_todos` tool, the `todos` state and the
    todo prompt. This subclass adds one behaviour: if the model tries to give a
    final answer (no tool calls) while todos are still open, we add a reminder
    message and send the model back for another turn, at most `max_reminders`
    times per user message.
    """

    def __init__(self, *, max_reminders: int = 2) -> None:
        super().__init__()
        self._max_reminders = max_reminders

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Restore checkpointed todos when their original tool call left context."""
        todos = state.get("todos") or []
        messages = state.get("messages") or []
        if not todos or self._todos_are_visible(messages) or self._context_reminder_is_visible(messages):
            return None
        items = "\n".join(f"- [{todo.get('status', 'pending')}] {todo.get('content', '')}" for todo in todos)
        return {
            "messages": [
                injected_message(
                    "<system_reminder>\n"
                    "Your earlier todo list is no longer visible, but it is still active:\n"
                    f"{items}\n"
                    "Continue tracking it and call write_todos whenever an item changes.\n"
                    "</system_reminder>",
                    TODO_CONTEXT,
                )
            ]
        }

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # First let the base class reject duplicate parallel write_todos calls.
        base_result = super().after_model(state, runtime)
        if base_result is not None:
            return base_result

        messages = state["messages"]
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None  # the model is still working, not finishing

        unfinished = [t for t in state.get("todos") or [] if t.get("status") != "completed"]
        if not unfinished or self._reminders_sent(messages) >= self._max_reminders:
            return None

        return {
            "messages": [self._reminder(unfinished)],
            "jump_to": "model",
        }

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    @staticmethod
    def _reminder(unfinished: list[dict[str, Any]]) -> HumanMessage:
        items = "\n".join(f"- [{t.get('status')}] {t.get('content')}" for t in unfinished)
        return injected_message(
            "<system_reminder>\nYou still have unfinished todos:\n"
            f"{items}\nFinish them, or update the list with write_todos if the plan changed, "
            "before giving your final answer.\n</system_reminder>",
            TODO_REMINDER,
        )

    @staticmethod
    def _reminders_sent(messages: list) -> int:
        """Count reminders since the last real user message (= this run)."""
        count = 0
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            if injected_kind(message) != TODO_REMINDER:
                break
            count += 1
        return count

    @staticmethod
    def _todos_are_visible(messages: list) -> bool:
        return any(
            isinstance(message, AIMessage)
            and any(call.get("name") == "write_todos" for call in message.tool_calls or [])
            for message in messages
        )

    @staticmethod
    def _context_reminder_is_visible(messages: list) -> bool:
        return any(
            isinstance(message, HumanMessage) and injected_kind(message) == TODO_CONTEXT
            for message in messages
        )
