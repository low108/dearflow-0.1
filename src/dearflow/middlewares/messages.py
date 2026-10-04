"""Helpers for messages that middleware injects into the conversation.

Several middlewares add hidden messages (date reminder, todo reminders). They
all mark those messages the same way, so any middleware can tell a *real* user
message from an injected one.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage

_INJECTED_KEY = "dearflow_injected"

DATE_REMINDER = "date"
TODO_REMINDER = "todo_reminder"
TODO_CONTEXT = "todo_context"


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


def text_of(message: AnyMessage) -> str:
    """Plain text of a message, whatever shape its content has."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
    return "".join(parts)
