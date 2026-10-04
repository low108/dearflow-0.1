"""The lead agent: the one place where every part is wired together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from dearflow.config import AppConfig
from dearflow.memory import FactExtractor, MemoryManager, MemoryStore
from dearflow.middlewares import MiddlewareChain
from dearflow.models import ModelFactory
from dearflow.prompts import PromptBuilder
from dearflow.skills import SkillCatalog, create_load_skill_tool
from dearflow.subagents import SubagentExecutor, SubagentRegistry
from dearflow.tools import ToolRegistry, ask_clarification, create_task_tool


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
                   -> SkillCatalog    -> PromptBuilder -> system prompt
                   -> ToolRegistry    -> tools (+ load_skill when skills exist)
                   -> MiddlewareChain -> middleware (context + clarification; todo guard in plan mode)
                   -> optional: SubagentExecutor -> `task` tool + limit middleware + prompt section
                   -> create_agent(...) -> compiled graph
    """

    def __init__(
        self,
        app_config: AppConfig,
        *,
        model_factory: ModelFactory | None = None,
        skill_catalog: SkillCatalog | None = None,
        tool_registry: ToolRegistry | None = None,
        middleware_chain: MiddlewareChain | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._app_config = app_config
        self._model_factory = model_factory or ModelFactory()
        self._skills = skill_catalog if skill_catalog is not None else self._discover_skills()
        self._tool_registry = tool_registry or ToolRegistry(app_config, extra_tools=self._builtin_tools())
        self._middleware_chain = middleware_chain  # None = build the default per run
        self._checkpointer = checkpointer
        self._memory: MemoryManager | None = None  # built on first use

    def build(self, run_config: RunnableConfig | None = None) -> LeadAgent:
        """Create the agent. `configurable.model_name` overrides the default model."""
        configurable = (run_config or {}).get("configurable") or {}
        requested = configurable.get("model_name")
        model_name = self._app_config.resolve_model(requested if isinstance(requested, str) else None)

        plan_mode = bool(configurable.get("is_plan_mode", self._app_config.plan_mode))
        subagents_on = bool(configurable.get("subagent_enabled", self._app_config.subagents.enabled))
        model = self._model_factory.create(model_name)
        tools = self._tool_registry.get_tools()
        registry = None
        if subagents_on:
            registry = SubagentRegistry.from_config(self._app_config.subagents)
            executor = SubagentExecutor(tools, default_model=model, model_resolver=self._create_model)
            tools = [*tools, create_task_tool(executor, registry)]

        subagent_config = self._app_config.subagents
        chain = self._middleware_chain or MiddlewareChain.default(
            plan_mode=plan_mode,
            context=self._app_config.context,
            model=model,
            summary_model=self._summary_model(),
            subagents=subagent_config if subagents_on else None,
            memory=self.memory,
            inject_memory=self._app_config.memory.injection_enabled,
        )

        prompt = PromptBuilder(
            self._app_config.system_prompt,
            self._skills,
            subagents=registry,
            max_concurrent=subagent_config.max_concurrent,
            max_total=subagent_config.max_total,
        ).build()

        graph = create_agent(
            model=model,
            tools=tools,
            middleware=chain.build(),
            system_prompt=prompt,
            state_schema=ThreadState,
            context_schema=dict,
            checkpointer=self._checkpointer,
        )
        return LeadAgent(graph=graph, model_name=model_name)

    @property
    def memory(self) -> MemoryManager | None:
        """The shared memory manager (None when memory is off). One per factory,
        so every agent it builds feeds the same update queue."""
        config = self._app_config.memory
        if not config.enabled:
            return None
        if self._memory is None:
            model = self._create_model(config.model or self._app_config.default_model)
            self._memory = MemoryManager(
                MemoryStore(config.storage_path, max_facts=config.max_facts),
                FactExtractor(model, confidence_threshold=config.confidence_threshold),
                debounce_seconds=config.debounce_seconds,
                max_injection_tokens=config.max_injection_tokens,
            )
        return self._memory

    def flush_memory(self) -> None:
        """Save pending memory updates now (call before exiting)."""
        if self._memory is not None:
            self._memory.flush()

    def _create_model(self, alias: str):
        return self._model_factory.create(self._app_config.resolve_model(alias))

    def _summary_model(self):
        """A separate (usually cheaper) model for summarization, if configured."""
        alias = self._app_config.context.summarization.model
        if alias is None:
            return None
        return self._model_factory.create(self._app_config.resolve_model(alias))

    def _discover_skills(self) -> SkillCatalog:
        skills_config = self._app_config.skills
        return SkillCatalog.discover(skills_config.path) if skills_config.enabled else SkillCatalog()

    def _skill_tools(self) -> list:
        return [create_load_skill_tool(self._skills)] if self._skills else []

    def _builtin_tools(self) -> list:
        """Tools every agent gets, added by code rather than listed in config."""
        return [ask_clarification, *self._skill_tools()]
