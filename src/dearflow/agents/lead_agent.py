from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain_core.runnables import RunnableConfig

from dearflow.agents.assembly import LeadAgentAssembly
from dearflow.agents.dependencies import AgentDependencies
from dearflow.config.app_config import AppConfig


class ThreadState(AgentState):
    """Conversation state shared by the graph and future middleware."""

    title: NotRequired[str]
    todos: NotRequired[list[dict[str, Any]]]
    artifacts: NotRequired[list[str]]
    summary_text: NotRequired[str]
    delegations: NotRequired[list[dict[str, Any]]]
    skill_context: NotRequired[list[dict[str, Any]]]
    uploaded_files: NotRequired[list[dict[str, Any]]]


def assemble_lead_agent(
    config: RunnableConfig,
    *,
    app_config: AppConfig,
    dependencies: AgentDependencies,
) -> LeadAgentAssembly:
    """Construct the configured lead-agent graph """

    runtime_config = config.get("configurable", {}) or {}
    requested_model = runtime_config.get("model_name")
    model_name = app_config.resolve_model(requested_model if isinstance(requested_model, str) else None)
    model = dependencies.create_model(model_name)

    graph = create_agent(
        model=model,
        tools=list(dependencies.tools),
        middleware=list(dependencies.middleware),
        system_prompt=app_config.system_prompt,
        state_schema=ThreadState,
        context_schema=dict,
        checkpointer=dependencies.checkpointer,
    )

    return LeadAgentAssembly(
        graph=graph,
        effective_model=model_name,
    )
