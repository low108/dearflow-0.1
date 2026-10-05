"""Conversation value passed from memory middleware to the extractor."""

from __future__ import annotations


class Exchange:
    """One immutable user message and its final assistant reply."""

    __slots__ = ("_assistant", "_user")

    def __init__(self, user: str, assistant: str = "") -> None:
        object.__setattr__(self, "_user", user)
        object.__setattr__(self, "_assistant", assistant)

    @property
    def user(self) -> str:
        return self._user

    @property
    def assistant(self) -> str:
        return self._assistant

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Exchange):
            return NotImplemented
        return (self.user, self.assistant) == (other.user, other.assistant)

    def __hash__(self) -> int:
        return hash((self.user, self.assistant))

    def __repr__(self) -> str:
        return f"Exchange(user={self.user!r}, assistant={self.assistant!r})"
