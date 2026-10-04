"""Ordered collection of agent middleware."""

from __future__ import annotations

from collections.abc import Iterable

from langchain.agents.middleware import AgentMiddleware

from dearflow.middlewares.clarification import ClarificationMiddleware
from dearflow.middlewares.todo import TodoMiddleware
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
    def default(cls, *, plan_mode: bool = False) -> MiddlewareChain:
        """The standard DearFlow pipeline. Later slices add entries here.

        Clarification is last on purpose: `after_model` hooks run in reverse
        order, so it sees the model's answer first and can strip sibling tool
        calls before any other middleware looks at them.
        """
        chain = cls([ToolErrorHandlingMiddleware()])
        if plan_mode:
            chain.add(TodoMiddleware())
        chain.add(ClarificationMiddleware())
        return chain

    def add(self, middleware: AgentMiddleware) -> MiddlewareChain:
        self._middlewares.append(middleware)
        return self

    def build(self) -> list[AgentMiddleware]:
        return list(self._middlewares)
