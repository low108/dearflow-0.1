"""Gives each conversation a short title."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from dearflow.middlewares.messages import is_real_user_message, text_of

logger = logging.getLogger(__name__)

_PROMPT = (
    "Generate a concise title (max {max_words} words) for this conversation.\n"
    "User: {user}\nAssistant: {assistant}\n"
    "Return ONLY the title, no quotes, no explanation."
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_SNIPPET_CHARS = 500


class TitleMiddleware(AgentMiddleware):
    """Writes `state["title"]` once, after the first exchange.

    The agent never reads the title; it exists so a UI (or the CLI) can label
    the conversation in a list. It runs in `after_agent`, when the whole run is
    over, which also covers a first run that ended with a clarification
    question instead of an answer.

    With `model=None` the title is just the user's first message, shortened
    (no extra model call). With a model, the model writes it, and any failure
    falls back to the shortened message.
    """

    def __init__(self, *, model: BaseChatModel | None = None, max_words: int = 6, max_chars: int = 60) -> None:
        super().__init__()
        self._model = model
        self._max_words = max_words
        self._max_chars = max_chars

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        exchange = self._first_exchange(state)
        if exchange is None:
            return None
        user, assistant = exchange
        title = self._fallback(user)
        if self._model is not None:
            try:
                reply = self._model.invoke(self._prompt(user, assistant))
                title = self._clean(text_of(reply)) or title
            except Exception:  # a title is never worth failing the run
                logger.warning("Title generation failed; using fallback", exc_info=True)
        return {"title": title}

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        exchange = self._first_exchange(state)
        if exchange is None:
            return None
        user, assistant = exchange
        title = self._fallback(user)
        if self._model is not None:
            try:
                reply = await self._model.ainvoke(self._prompt(user, assistant))
                title = self._clean(text_of(reply)) or title
            except Exception:
                logger.warning("Title generation failed; using fallback", exc_info=True)
        return {"title": title}

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _first_exchange(state: Any) -> tuple[str, str] | None:
        """(user text, assistant text) if a title is due, else None."""
        if state.get("title"):
            return None
        messages = state["messages"]
        users = [m for m in messages if is_real_user_message(m)]
        if len(users) != 1:
            return None
        # The assistant side is the last text the agent produced. A clarification
        # question arrives as a ToolMessage, so those count too.
        replies = [m for m in messages if isinstance(m, (AIMessage, ToolMessage)) and text_of(m).strip()]
        if not replies:
            return None
        return text_of(users[0]), text_of(replies[-1])

    def _prompt(self, user: str, assistant: str) -> str:
        return _PROMPT.format(
            max_words=self._max_words,
            user=user[:_SNIPPET_CHARS],
            assistant=assistant[:_SNIPPET_CHARS],
        )

    def _clean(self, raw: str) -> str:
        text = _THINK_BLOCK.sub("", raw).strip().strip("\"'`")
        return self._shorten(text)

    def _fallback(self, user: str) -> str:
        return self._shorten(user) or "New conversation"

    def _shorten(self, text: str) -> str:
        text = " ".join(text.split())
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars - 3].rstrip() + "..."
