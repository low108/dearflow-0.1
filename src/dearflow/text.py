"""Small message helpers with no dependencies on the rest of the package."""

from __future__ import annotations

from langchain_core.messages import AnyMessage


def text_of(message: AnyMessage) -> str:
    """Plain text of a message, whatever shape its content has."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
    return "".join(parts)
