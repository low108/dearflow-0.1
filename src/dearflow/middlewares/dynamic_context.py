"""Tells the model today's date without touching the system prompt."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents.middleware import AgentMiddleware

from dearflow.middlewares.messages import (
    DATE_REMINDER,
    injected_kind,
    injected_message,
    insert_before_last_user,
    is_real_user_message,
)

Clock = Callable[[], datetime]


def make_clock(timezone: str | None) -> Clock:
    """A clock in the given IANA timezone, or the machine's local zone."""
    if timezone:
        zone = ZoneInfo(timezone)
        return lambda: datetime.now(zone)
    return lambda: datetime.now().astimezone()


class DynamicContextMiddleware(AgentMiddleware):
    """Injects `<current_date>` as a hidden message before the user's message.

    Why not put the date in the system prompt? The system prompt is identical
    on every call, which lets model providers cache it. Volatile facts such as
    the date go into the conversation instead.

    Runs once per user turn (`before_agent`). It adds a reminder only when the
    conversation has none yet, or the last one is for an earlier day, so a long
    chat gets about one reminder per day.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        super().__init__()
        self._clock = clock or make_clock(None)

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if last is None or last.id is None or not is_real_user_message(last):
            return None

        today = self._clock().strftime("%Y-%m-%d, %A")
        if self._last_reminder_date(messages) == today:
            return None

        reminder = injected_message(
            f"<system-reminder>\n<current_date>{today}</current_date>\n</system-reminder>",
            DATE_REMINDER,
            reminder_date=today,
        )
        return {"messages": insert_before_last_user(last, reminder)}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    @staticmethod
    def _last_reminder_date(messages: list) -> str | None:
        for message in reversed(messages):
            if injected_kind(message) == DATE_REMINDER:
                return message.additional_kwargs.get("reminder_date")
        return None
