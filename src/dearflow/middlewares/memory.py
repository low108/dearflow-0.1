"""Connects the chat to long-term memory."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.config import get_config

from dearflow.memory import Exchange, MemoryManager
from dearflow.middlewares.messages import (
    MEMORY_CONTEXT,
    injected_kind,
    injected_message,
    insert_before_last_user,
    is_real_user_message,
    text_of,
)

DEFAULT_USER = "default"


def current_user_id() -> str:
    """The user for this run: `configurable.user_id`, else "default"."""
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:  # called outside a graph run
        return DEFAULT_USER
    return str(configurable.get("user_id") or DEFAULT_USER)


class MemoryMiddleware(AgentMiddleware):
    """Read before the run, write after it.

    - `before_agent`: puts what we know about the user in a hidden `<memory>`
      message in front of their message. Like the date, it goes into the
      conversation, not the system prompt, so the prompt stays cacheable. A new
      message is added only when the text changed since the last one.
    - `after_agent`: hands the latest exchange to `MemoryManager`, which extracts
      facts in the background. Only real user text and the assistant's final
      reply are used; tool calls and hidden messages are ignored.
    """

    def __init__(
        self, manager: MemoryManager, *, inject: bool = True, redact: Callable[[str], str] | None = None
    ) -> None:
        super().__init__()
        self._manager = manager
        self._inject = inject
        self._redact = redact  # the queue is saved to disk, so personal data must not get in

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self._inject:
            return None
        messages = state["messages"]
        last = messages[-1] if messages else None
        if last is None or last.id is None or not is_real_user_message(last):
            return None

        context = self._manager.context(current_user_id())
        digest = hashlib.sha1(context.encode()).hexdigest()[:12]
        if not context or self._last_digest(messages) == digest:
            return None

        block = injected_message(f"<memory>\n{context}\n</memory>", MEMORY_CONTEXT, memory_digest=digest)
        return {"messages": insert_before_last_user(last, block)}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        exchange = self._latest_exchange(state["messages"])
        if exchange is not None:
            if self._redact is not None:
                exchange = Exchange(self._redact(exchange.user), self._redact(exchange.assistant))
            self._manager.enqueue(current_user_id(), [exchange])
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    @staticmethod
    def _last_digest(messages: list) -> str | None:
        for message in reversed(messages):
            if injected_kind(message) == MEMORY_CONTEXT:
                return message.additional_kwargs.get("memory_digest")
        return None

    @staticmethod
    def _latest_exchange(messages: list) -> Exchange | None:
        """The last real user message and the assistant's final text after it."""
        for index in range(len(messages) - 1, -1, -1):
            if is_real_user_message(messages[index]):
                user = text_of(messages[index]).strip()
                break
        else:
            return None
        if not user:
            return None
        replies = [
            text_of(m).strip()
            for m in messages[index + 1 :]
            if isinstance(m, AIMessage) and not m.tool_calls and text_of(m).strip()
        ]
        return Exchange(user=user, assistant=replies[-1] if replies else "")
