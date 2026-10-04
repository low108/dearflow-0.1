"""Ordered collection of agent middleware."""

from __future__ import annotations

from collections.abc import Iterable

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel

from dearflow.config import ContextConfig
from dearflow.middlewares.clarification import ClarificationMiddleware
from dearflow.middlewares.dynamic_context import Clock, DynamicContextMiddleware, make_clock
from dearflow.middlewares.title import TitleMiddleware
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
    def default(
        cls,
        *,
        plan_mode: bool = False,
        context: ContextConfig | None = None,
        model: BaseChatModel | None = None,
        summary_model: BaseChatModel | None = None,
        clock: Clock | None = None,
    ) -> MiddlewareChain:
        """The standard DearFlow pipeline. Later slices add entries here.

        Order, outermost first:
            tool errors -> date reminder -> summarization -> title -> todos -> clarification

        Clarification is last on purpose: `after_model` hooks run in reverse
        order, so it sees the model's answer first and can strip sibling tool
        calls before any other middleware looks at them.

        Context middleware is only added when `context` is given (the factory
        always gives it), so a bare `default()` stays small for tests.
        """
        chain = cls([ToolErrorHandlingMiddleware()])
        if context is not None:
            if context.date_reminder:
                chain.add(DynamicContextMiddleware(clock or make_clock(context.timezone)))
            summarization = context.summarization
            if summarization.enabled and (summary_model or model) is not None:
                chain.add(
                    SummarizationMiddleware(
                        summary_model or model,
                        trigger=("tokens", summarization.trigger_tokens),
                        keep=("messages", summarization.keep_messages),
                    )
                )
            if context.title.enabled:
                chain.add(
                    TitleMiddleware(
                        model=model if context.title.use_model else None,
                        max_words=context.title.max_words,
                    )
                )
        if plan_mode:
            chain.add(TodoMiddleware())
        chain.add(ClarificationMiddleware())
        return chain

    def add(self, middleware: AgentMiddleware) -> MiddlewareChain:
        self._middlewares.append(middleware)
        return self

    def build(self) -> list[AgentMiddleware]:
        return list(self._middlewares)
