from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from dearflow.agents.dependencies import AgentDependencies
from dearflow.config import AppConfig
from dearflow.tools import get_available_tools


def build_agent_dependencies(
    app_config: AppConfig,
    *,
    create_model: Callable[[str], BaseChatModel],
    checkpointer: BaseCheckpointSaver | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    tool_groups: Iterable[str] | None = None,
) -> AgentDependencies:
    """Build the model, tool, middleware, and persistence adapters for assembly."""

    return AgentDependencies(
        create_model=create_model,
        tools=get_available_tools(app_config, groups=tool_groups),
        middleware=middleware,
        checkpointer=checkpointer,
    )
