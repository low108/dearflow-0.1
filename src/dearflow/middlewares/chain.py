"""Ordered collection of agent middleware."""

from __future__ import annotations

from collections.abc import Iterable

from langchain.agents.middleware import AgentMiddleware

from dearflow.middlewares.tool_error_handling import ToolErrorHandlingMiddleware


class MiddlewareChain:
    """Holds middleware in execution order.

    Order matters: middleware wrap each other like onion layers, so the first
    item is the outermost. Keeping the order in one class means a reader can
    see the whole request pipeline in `default()`.
    """

    def __init__(self, middlewares: Iterable[AgentMiddleware] = ()) -> None:
        self._middlewares: list[AgentMiddleware] = list(middlewares)

    @classmethod
    def default(cls) -> MiddlewareChain:
        """The standard DearFlow pipeline. Later slices append to this list."""
        return cls([ToolErrorHandlingMiddleware()])

    def add(self, middleware: AgentMiddleware) -> MiddlewareChain:
        self._middlewares.append(middleware)
        return self

    def build(self) -> list[AgentMiddleware]:
        return list(self._middlewares)
