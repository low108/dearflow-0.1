"""Enforces delegation limits in code, not just in the prompt."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from dearflow.middlewares.messages import is_real_user_message
from dearflow.subagents.spec import TASK

_LIMIT_NOTE = (
    "[SUBAGENT LIMIT REACHED] Some task calls were dropped because the delegation "
    "limit was hit. Use the results you already have, or do the remaining work directly."
)


class SubagentLimitMiddleware(AgentMiddleware):
    """Trims excess `task` calls from the model's response.

    Two caps, both checked in `after_model`:
    - `max_concurrent`: task calls allowed in ONE response (they run in parallel)
    - `max_total`: task calls allowed since the user's last message

    A prompt can ask the model to behave; this guarantees it. Dropped calls are
    removed from the AI message itself, so no tool call is left unanswered.
    """

    def __init__(self, max_concurrent: int = 3, max_total: int = 6) -> None:
        super().__init__()
        self.max_concurrent = max(1, max_concurrent)
        self.max_total = max(1, max_total)

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return None

        task_positions = [i for i, call in enumerate(last.tool_calls) if call["name"] == TASK]
        allowed = min(self.max_concurrent, max(0, self.max_total - self._used_this_run(messages[:-1])))
        if len(task_positions) <= allowed:
            return None

        dropped = set(task_positions[allowed:])
        kept = [call for i, call in enumerate(last.tool_calls) if i not in dropped]
        content = last.content if isinstance(last.content, str) else "".join(str(part) for part in last.content)
        trimmed = last.model_copy(
            update={"tool_calls": kept, "content": f"{content}\n\n{_LIMIT_NOTE}".strip()}
        )
        return {"messages": [trimmed]}  # same id: replaces the original message

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    @staticmethod
    def _used_this_run(earlier: list) -> int:
        """Task calls made since the last real user message."""
        used = 0
        for message in reversed(earlier):
            if is_real_user_message(message):
                break
            if isinstance(message, AIMessage):
                used += sum(1 for call in message.tool_calls if call["name"] == TASK)
        return used
