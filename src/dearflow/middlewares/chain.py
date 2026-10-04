"""Ordered collection of agent middleware."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel

from dearflow.config import ContextConfig, SubagentsConfig, TokenBudgetConfig
from dearflow.middlewares.clarification import ClarificationMiddleware
from dearflow.middlewares.dynamic_context import Clock, DynamicContextMiddleware, make_clock
from dearflow.memory import MemoryManager
from dearflow.middlewares.memory import MemoryMiddleware
from dearflow.middlewares.pii_redaction import PiiRedactionMiddleware
from dearflow.middlewares.redacting_summarization import RedactingSummarizationMiddleware
from dearflow.middlewares.subagent_limit import SubagentLimitMiddleware
from dearflow.middlewares.token_budget import TokenBudgetMiddleware
from dearflow.middlewares.title import TitleMiddleware
from dearflow.middlewares.todo import TodoMiddleware
from dearflow.middlewares.tool_error_handling import ToolErrorHandlingMiddleware
from dearflow.pii import Redactor
from dearflow.usage import UsageRegistry


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
        subagents: SubagentsConfig | None = None,
        memory: MemoryManager | None = None,
        inject_memory: bool = True,
        token_budget: TokenBudgetConfig | None = None,
        usage: UsageRegistry | None = None,
        redactor: Redactor | None = None,
        pii_tool_names: Sequence[str] = (),
        clock: Clock | None = None,
    ) -> MiddlewareChain:
        """The standard DearFlow pipeline. Later slices add entries here.

        Order, outermost first:
            tool errors -> date reminder -> memory -> summarization -> title -> subagent limit -> token budget -> todos -> PII redaction -> clarification

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
            if memory is not None:
                chain.add(MemoryMiddleware(memory, inject=inject_memory, redact=redactor.redact if redactor else None))
            summarization = context.summarization
            if summarization.enabled and (summary_model or model) is not None:
                settings = {
                    "trigger": ("tokens", summarization.trigger_tokens),
                    "keep": ("messages", summarization.keep_messages),
                }
                chosen = summary_model or model
                chain.add(
                    RedactingSummarizationMiddleware(chosen, redactor=redactor, **settings)
                    if redactor
                    else SummarizationMiddleware(chosen, **settings)
                )
            if context.title.enabled:
                chain.add(
                    TitleMiddleware(
                        model=model if context.title.use_model else None,
                        max_words=context.title.max_words,
                        redact=redactor.redact if redactor else None,
                    )
                )
        if subagents is not None:
            chain.add(SubagentLimitMiddleware(subagents.max_concurrent, subagents.max_total))
        if token_budget is not None and token_budget.enabled and usage is not None:
            chain.add(TokenBudgetMiddleware(token_budget, usage))
        if plan_mode:
            chain.add(TodoMiddleware())
        if redactor is not None:
            chain.add(PiiRedactionMiddleware(redactor, pii_tool_names))
        chain.add(ClarificationMiddleware())
        return chain

    def add(self, middleware: AgentMiddleware) -> MiddlewareChain:
        self._middlewares.append(middleware)
        return self

    def build(self) -> list[AgentMiddleware]:
        return list(self._middlewares)
