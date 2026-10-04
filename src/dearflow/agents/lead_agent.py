"""The lead agent: the one place where every part is wired together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from dearflow.config import AppConfig
from dearflow.middlewares import MiddlewareChain
from dearflow.models import ModelFactory
from dearflow.tools import ToolRegistry


class ThreadState(AgentState):
    """Conversation state shared by the graph and middleware.

    `messages` comes from AgentState. The optional keys are reserved for later
    slices (todos, title, summaries, ...), so the schema is stable from day one.
    """

    title: NotRequired[str]
    todos: NotRequired[list[dict[str, Any]]]
    artifacts: NotRequired[list[str]]
    summary_text: NotRequired[str]
    delegations: NotRequired[list[dict[str, Any]]]
    skill_context: NotRequired[list[dict[str, Any]]]
    uploaded_files: NotRequired[list[dict[str, Any]]]


@dataclass(frozen=True)
class LeadAgent:
    """A built agent: the runnable graph plus the model it resolved to."""

    graph: CompiledStateGraph
    model_name: str


class LeadAgentFactory:
    """Builds a `LeadAgent` from its collaborators.

    Flow of `build()`:
        run config -> pick model name -> ModelFactory -> model
                   -> ToolRegistry    -> tools
                   -> MiddlewareChain -> middleware
                   -> create_agent(...) -> compiled graph
    """

    def __init__(
        self,
        app_config: AppConfig,
        *,
        model_factory: ModelFactory | None = None,
        tool_registry: ToolRegistry | None = None,
        middleware_chain: MiddlewareChain | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._app_config = app_config
        self._model_factory = model_factory or ModelFactory()
        self._tool_registry = tool_registry or ToolRegistry(app_config)
        self._middleware_chain = middleware_chain or MiddlewareChain.default()
        self._checkpointer = checkpointer

    def build(self, run_config: RunnableConfig | None = None) -> LeadAgent:
        """Create the agent. `configurable.model_name` overrides the default model."""
        configurable = (run_config or {}).get("configurable") or {}
        requested = configurable.get("model_name")
        model_name = self._app_config.resolve_model(requested if isinstance(requested, str) else None)

        graph = create_agent(
            model=self._model_factory.create(model_name),
            tools=self._tool_registry.get_tools(),
            middleware=self._middleware_chain.build(),
            system_prompt=self._app_config.system_prompt,
            state_schema=ThreadState,
            context_schema=dict,
            checkpointer=self._checkpointer,
        )
        return LeadAgent(graph=graph, model_name=model_name)
