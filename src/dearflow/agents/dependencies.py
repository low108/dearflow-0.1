from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver


class AgentDependencies:
    """Runtime adapters used while constructing the lead agent."""

    __slots__ = ("create_model", "tools", "middleware", "checkpointer")

    def __init__(
        self,
        create_model: Callable[[str], BaseChatModel],
        tools: Sequence[BaseTool] = (),
        middleware: Sequence[AgentMiddleware] = (),
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.create_model = create_model
        self.tools = tools
        self.middleware = middleware
        self.checkpointer = checkpointer
