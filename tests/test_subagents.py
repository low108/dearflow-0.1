import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig, SubagentsConfig
from dearflow.middlewares import MiddlewareChain, SubagentLimitMiddleware
from dearflow.prompts import PromptBuilder
from dearflow.subagents import (
    GENERAL_PURPOSE,
    SubagentExecutor,
    SubagentRegistry,
    SubagentStatus,
    UnknownSubagentError,
)
from dearflow.subagents.executor import NO_RESPONSE, filter_tools
from dearflow.tools import create_task_tool


def call(name, id, **args):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def task_call(id="t1", prompt="look it up"):
    return call("task", id, prompt=prompt, subagent_type="general-purpose")


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"answer to {q}"


# --- registry / spec ----------------------------------------------------------


def test_registry_has_builtin_and_custom_types():
    config = SubagentsConfig.model_validate(
        {
            "max_turns": 7,
            "custom_agents": {
                "reviewer": {"description": "Reviews text.", "system_prompt": "Review.", "tools": ["lookup"]}
            },
        }
    )
    registry = SubagentRegistry.from_config(config)

    assert registry.names() == ["general-purpose", "reviewer"]
    reviewer = registry.get("reviewer")
    assert reviewer.tools == ("lookup",) and reviewer.max_turns == 7
    assert "task" in reviewer.disallowed_tools  # no nesting, even for custom types
    assert "reviewer: Reviews text." in registry.describe()


def test_unknown_type_error_lists_valid_types():
    with pytest.raises(UnknownSubagentError, match="general-purpose"):
        SubagentRegistry().get("nope")


def test_config_rejects_typos_in_custom_agents():
    with pytest.raises(ValueError):
        SubagentsConfig.model_validate({"custom_agents": {"x": {"description": "d", "system_prompt": "p", "tols": []}}})


def test_filter_tools_denylist_beats_allowlist():
    tools = [lookup, create_task_tool_stub()]
    assert [t.name for t in filter_tools(tools, None, ("task",))] == ["lookup"]
    assert [t.name for t in filter_tools(tools, ("task",), ("task",))] == []


def create_task_tool_stub():
    @tool("task")
    def task_stub(x: str) -> str:
        """stub"""
        return x

    return task_stub


# --- executor -----------------------------------------------------------------


def executor_for(responses, tools=(lookup,)):
    model = ToolCapableFakeModel(responses=responses)
    return SubagentExecutor(tools, default_model=model, model_resolver=lambda alias: model)


def test_executor_returns_last_ai_text_after_using_tools():
    executor = executor_for([AIMessage(content="", tool_calls=[call("lookup", "c1", q="x")]), AIMessage(content="It is 42")])
    result = executor.run(GENERAL_PURPOSE, "what is x?")

    assert result.status is SubagentStatus.COMPLETED
    assert result.render() == "Task Succeeded. Result: It is 42"


def test_executor_stops_at_turn_limit_and_keeps_partial_work():
    looping = [AIMessage(content="still working", tool_calls=[call("lookup", f"c{i}", q="x")]) for i in range(20)]
    executor = executor_for(looping)
    spec = GENERAL_PURPOSE.__class__(**{**GENERAL_PURPOSE.__dict__, "max_turns": 2})
    result = executor.run(spec, "go")

    assert result.status is SubagentStatus.MAX_TURNS
    assert "still working" in result.text


def test_executor_reports_failure_instead_of_raising():
    executor = executor_for([])  # the fake model runs out of responses
    result = executor.run(GENERAL_PURPOSE, "go")

    assert result.status is SubagentStatus.FAILED
    assert result.render().startswith("Task failed. Error:")


def test_no_response_sentinel():
    from dearflow.subagents.executor import final_text

    assert final_text({"messages": [HumanMessage("hi")]}) == NO_RESPONSE


def test_executor_async_path():
    executor = executor_for([AIMessage(content="async done")])
    result = asyncio.run(executor.arun(GENERAL_PURPOSE, "go"))

    assert result.render() == "Task Succeeded. Result: async done"


# --- limit middleware ---------------------------------------------------------


def ai_with(*ids):
    return AIMessage(content="", id="ai-last", tool_calls=[task_call(i) for i in ids])


def test_limit_caps_calls_per_response():
    update = SubagentLimitMiddleware(max_concurrent=2, max_total=10).after_model(
        {"messages": [HumanMessage("go"), ai_with("a", "b", "c", "d")]}, None
    )

    trimmed = update["messages"][0]
    assert [c["id"] for c in trimmed.tool_calls] == ["a", "b"]
    assert trimmed.id == "ai-last"  # replaces the original
    assert "SUBAGENT LIMIT REACHED" in trimmed.content


def test_limit_caps_total_per_user_message_and_resets_on_next_one():
    middleware = SubagentLimitMiddleware(max_concurrent=3, max_total=3)
    earlier = [
        HumanMessage("go"),
        ai_with("a", "b"),
        ToolMessage("r", tool_call_id="a"),
        ToolMessage("r", tool_call_id="b"),
    ]

    update = middleware.after_model({"messages": [*earlier, ai_with("c", "d")]}, None)
    assert [c["id"] for c in update["messages"][0].tool_calls] == ["c"]  # 2 used, 1 left

    fresh = middleware.after_model({"messages": [*earlier, AIMessage(content="ok"), HumanMessage("more"), ai_with("c", "d")]}, None)
    assert fresh is None  # a new user message gets a new budget


def test_limit_ignores_other_tools_and_keeps_them():
    message = AIMessage(content="", id="x", tool_calls=[call("lookup", "l1", q="a"), task_call("a"), task_call("b")])
    update = SubagentLimitMiddleware(max_concurrent=1).after_model({"messages": [HumanMessage("go"), message]}, None)

    assert [c["id"] for c in update["messages"][0].tool_calls] == ["l1", "a"]


# --- wiring -------------------------------------------------------------------


def build(responses, **config):
    model = ToolCapableFakeModel(responses=responses)
    app = AppConfig(default_model="fake", subagents={"enabled": True, **config})
    return LeadAgentFactory(app, model_factory=FakeModelFactory(model)).build()


def test_lead_agent_delegates_through_the_task_tool():
    agent = build(
        [
            AIMessage(content="", tool_calls=[task_call()]),  # lead: delegate
            AIMessage(content="found: 42"),  # subagent: its report
            AIMessage(content="The answer is 42"),  # lead: final answer
        ]
    )
    result = agent.graph.invoke({"messages": [{"role": "user", "content": "find it"}]})

    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_result.content == "Task Succeeded. Result: found: 42"
    assert result["messages"][-1].content == "The answer is 42"
    # the subagent's own messages never enter the lead's history
    assert not any(isinstance(m, AIMessage) and m.content == "found: 42" for m in result["messages"])


def test_unknown_subagent_type_is_reported_to_the_model():
    bad = call("task", "t1", prompt="x", subagent_type="wizard")
    agent = build([AIMessage(content="", tool_calls=[bad]), AIMessage(content="ok")])
    result = agent.graph.invoke({"messages": [{"role": "user", "content": "go"}]})

    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Unknown subagent type 'wizard'" in tool_result.content


def test_subagents_off_by_default_and_switchable_per_run():
    model = ToolCapableFakeModel(responses=[])
    factory = LeadAgentFactory(AppConfig(default_model="fake"), model_factory=FakeModelFactory(model))

    off = factory.build().graph
    on = factory.build({"configurable": {"subagent_enabled": True}}).graph

    assert "task" not in off.get_graph().nodes["tools"].data.tools_by_name
    assert "task" in on.get_graph().nodes["tools"].data.tools_by_name


def test_prompt_mentions_delegation_only_when_enabled():
    assert "subagent_system" not in PromptBuilder("Base.").build()
    prompt = PromptBuilder("Base.", subagents=SubagentRegistry(), max_concurrent=2, max_total=4).build()

    assert "<subagent_system>" in prompt and "general-purpose" in prompt and "At most 2" in prompt


def test_default_chain_adds_limit_only_with_subagents():
    def names(chain):
        return [type(m).__name__ for m in chain.build()]

    assert "SubagentLimitMiddleware" not in names(MiddlewareChain.default())
    assert "SubagentLimitMiddleware" in names(MiddlewareChain.default(subagents=SubagentsConfig()))


def test_two_task_calls_in_one_response_both_run():
    agent = build(
        [
            AIMessage(content="", tool_calls=[task_call("a", "job A"), task_call("b", "job B")]),
            AIMessage(content="report-1"),
            AIMessage(content="report-2"),
            AIMessage(content="combined"),
        ]
    )
    result = agent.graph.invoke({"messages": [{"role": "user", "content": "do both"}]})

    results = {m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert results == {"Task Succeeded. Result: report-1", "Task Succeeded. Result: report-2"}
    assert result["messages"][-1].content == "combined"


def test_cli_flag_turns_subagents_on(tmp_path):
    from dearflow.cli import main

    path = tmp_path / "config.yaml"
    path.write_text('default_model: fast\nmodels:\n  fast: {model: "p:m"}\n', encoding="utf-8")
    sessions: list[dict] = []

    main(["--config", str(path), "--subagents"], chat_runner=lambda **kw: sessions.append(kw))
    main(["--config", str(path)], chat_runner=lambda **kw: sessions.append(kw))

    assert [s["app_config"].subagents.enabled for s in sessions] == [True, False]
