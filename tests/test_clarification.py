from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.middlewares import ClarificationMiddleware
from dearflow.prompts import PromptBuilder


def call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_format_question_uses_icon_context_and_numbered_options():
    text = ClarificationMiddleware.format_question(
        {
            "question": "Which one?",
            "clarification_type": "approach_choice",
            "context": "There are two ways.",
            "options": ["fast", "safe"],
        }
    )

    assert text == "🔀 There are two ways.\nWhich one?\n  1. fast\n  2. safe"
    assert ClarificationMiddleware.format_question({"question": "Name?", "clarification_type": "missing_info"}) == "❓ Name?"


def test_after_model_keeps_only_the_clarification_call():
    ask = call("ask_clarification", {"question": "q", "clarification_type": "missing_info"}, "a")
    other = call("calculator", {}, "b")
    message = AIMessage(content="", tool_calls=[other, ask], id="m1")

    update = ClarificationMiddleware().after_model({"messages": [message]}, None)

    assert update["messages"][0].id == "m1"
    assert [c["id"] for c in update["messages"][0].tool_calls] == ["a"]


def test_after_model_ignores_other_tool_calls():
    message = AIMessage(content="", tool_calls=[call("a", {}, "1"), call("b", {}, "2")])
    assert ClarificationMiddleware().after_model({"messages": [message]}, None) is None


def test_agent_ends_the_run_with_the_question_and_resumes_on_reply():
    ask = call(
        "ask_clarification",
        {"question": "Which city?", "clarification_type": "missing_info", "options": ["KL", "Penang"]},
        "c1",
    )
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[ask]),
            AIMessage(content="Weather in KL: sunny."),
        ]
    )
    from langgraph.checkpoint.memory import InMemorySaver

    agent = LeadAgentFactory(
        AppConfig(default_model="m"), model_factory=FakeModelFactory(model), checkpointer=InMemorySaver()
    ).build()
    thread = {"configurable": {"thread_id": "t"}}

    first = agent.graph.invoke({"messages": [HumanMessage("weather?")]}, thread)

    assert isinstance(first["messages"][-1], ToolMessage)
    assert first["messages"][-1].content == "❓ Which city?\n  1. KL\n  2. Penang"

    second = agent.graph.invoke({"messages": [HumanMessage("KL")]}, thread)
    assert second["messages"][-1].content == "Weather in KL: sunny."


def test_prompt_contains_the_clarification_rules():
    assert "ask_clarification" in PromptBuilder("Base.").build()
