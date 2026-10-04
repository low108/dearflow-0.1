"""Helpers for messages that middleware injects into the conversation.

Several middlewares add hidden messages (date reminder, todo reminders). They
all mark those messages the same way, so any middleware can tell a *real* user
message from an injected one.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage

from dearflow.text import text_of  # noqa: F401  (re-exported for middleware)

_INJECTED_KEY = "dearflow_injected"

DATE_REMINDER = "date"
TODO_REMINDER = "todo_reminder"
TODO_CONTEXT = "todo_context"
MEMORY_CONTEXT = "memory"


def injected_message(content: str, kind: str, **extra: Any) -> HumanMessage:
    """A hidden message added by middleware (not typed by the user)."""
    return HumanMessage(
        content=content,
        additional_kwargs={_INJECTED_KEY: kind, "hide_from_ui": True, **extra},
    )


def injected_kind(message: AnyMessage) -> str | None:
    """Return the injection kind, or None for an ordinary message."""
    return message.additional_kwargs.get(_INJECTED_KEY)


def is_real_user_message(message: AnyMessage) -> bool:
    return isinstance(message, HumanMessage) and injected_kind(message) is None


def insert_before_last_user(last: HumanMessage, injected: HumanMessage) -> list[AnyMessage]:
    """State update that puts `injected` right before the user's message.

    State updates can only append, so: remove the user message, add the hidden
    message, then add the user message back under a new id. The id suffix is
    added once, so several middlewares can stack reminders without it growing.
    """
    suffix = "__user"
    new_id = last.id if last.id.endswith(suffix) else f"{last.id}{suffix}"
    return [RemoveMessage(id=last.id), injected, last.model_copy(update={"id": new_id})]
