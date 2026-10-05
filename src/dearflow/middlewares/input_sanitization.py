"""Frame untrusted user text before it enters a model request."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

from dearflow.middlewares.messages import is_real_user_message

USER_INPUT_BEGIN = "--- BEGIN USER INPUT ---"
USER_INPUT_END = "--- END USER INPUT ---"

_BLOCKED_TAGS = frozenset(
    {
        "analysis",
        "available_skills",
        "citations",
        "current_date",
        "ignore",
        "important",
        "instruction",
        "memory",
        "override",
        "prompt",
        "role",
        "skill_system",
        "system",
        "system-reminder",
        "system_reminder",
        "think",
        "todo_list_system",
    }
)
_BLOCKED_TAG_PATTERN = re.compile(
    r"<\s*/?\s*(?:" + "|".join(re.escape(tag) for tag in sorted(_BLOCKED_TAGS)) + r")\b[^>]*>?",
    re.IGNORECASE,
)
_BOUNDARY_PATTERN = re.compile(re.escape(USER_INPUT_BEGIN) + "|" + re.escape(USER_INPUT_END))


def frame_untrusted_text(text: str) -> str:
    """Escape reserved tags and place genuine user text inside clear boundaries."""
    if not text.strip():
        return text

    def escape_tag(match: re.Match) -> str:
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    text = _BLOCKED_TAG_PATTERN.sub(escape_tag, text)
    if text.startswith(USER_INPUT_BEGIN) and text.endswith(USER_INPUT_END):
        inner = text[len(USER_INPUT_BEGIN) : -len(USER_INPUT_END)]
        inner = _neutralize_boundaries(inner)
        return f"{USER_INPUT_BEGIN}{inner}{USER_INPUT_END}"
    text = _neutralize_boundaries(text)
    return f"{USER_INPUT_BEGIN}\n{text}\n{USER_INPUT_END}"


def _neutralize_boundaries(text: str) -> str:
    return _BOUNDARY_PATTERN.sub(
        lambda match: "[BEGIN USER INPUT]" if match.group(0) == USER_INPUT_BEGIN else "[END USER INPUT]",
        text,
    )


def _frame_content(content: Any) -> tuple[Any, bool]:
    if isinstance(content, str):
        framed = frame_untrusted_text(content)
        return framed, framed != content
    if not isinstance(content, list):
        return content, False

    framed_blocks: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, str):
            framed = frame_untrusted_text(block)
            framed_blocks.append(framed)
            changed = changed or framed != block
        elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            framed = frame_untrusted_text(block["text"])
            framed_blocks.append({**block, "text": framed})
            changed = changed or framed != block["text"]
        else:
            framed_blocks.append(block)
    return framed_blocks, changed


class InputSanitizationMiddleware(AgentMiddleware):
    """Sanitize every genuine user turn in the model-bound copy of history."""

    @staticmethod
    def sanitize(request: ModelRequest) -> ModelRequest:
        messages = list(request.messages)
        changed = False
        for index, message in enumerate(messages):
            if not is_real_user_message(message):
                continue
            content, content_changed = _frame_content(message.content)
            if content_changed:
                messages[index] = message.model_copy(update={"content": content})
                changed = True
        return request.override(messages=messages) if changed else request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self.sanitize(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self.sanitize(request))
