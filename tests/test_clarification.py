from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.middlewares import ClarificationMiddleware
from dearflow.persistence import Persistence
from dearflow.prompts import PromptBuilder
from dearflow.service import RunService


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


def test_malformed_clarification_still_drops_executable_sibling_tools():
    message = AIMessage(
        content="",
        tool_calls=[call("write_file", {"path": "x"}, "safe-to-drop")],
        invalid_tool_calls=[
            {
                "name": "ask_clarification",
                "args": "{not-json",
                "id": "broken-question",
                "error": "invalid JSON",
                "type": "invalid_tool_call",
            }
        ],
        id="model-1",
    )

    update = ClarificationMiddleware().after_model({"messages": [message]}, None)

    assert update["messages"][0].id == "model-1"
    assert update["messages"][0].tool_calls == []
    assert update["messages"][0].invalid_tool_calls[0]["name"] == "ask_clarification"


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


def test_agent_returns_a_structured_multi_field_clarification_form():
    ask = call(
        "ask_clarification",
        {
            "question": "Configure the report",
            "clarification_type": "missing_info",
            "fields": [
                {"name": "title", "label": "Report title", "type": "text", "required": True},
                {
                    "name": "format",
                    "label": "Output format",
                    "type": "select",
                    "options": ["PDF", "Markdown"],
                },
            ],
        },
        "form-1",
    )
    model = ToolCapableFakeModel(responses=[AIMessage(content="", tool_calls=[ask])])
    agent = LeadAgentFactory(AppConfig(default_model="m"), model_factory=FakeModelFactory(model)).build()

    result = agent.graph.invoke({"messages": [HumanMessage("make a report")]})

    message = result["messages"][-1]
    assert "1. Report title (required)" in message.content
    assert "2. Output format — options: PDF / Markdown" in message.content
    assert message.artifact["human_input"] == {
        "version": 2,
        "kind": "human_input_request",
        "source": "ask_clarification",
        "request_id": "clarification:form-1",
        "tool_call_id": "form-1",
        "clarification_type": "missing_info",
        "question": "Configure the report",
        "input_mode": "form",
        "fields": [
            {"name": "title", "label": "Report title", "type": "text", "required": True},
            {
                "name": "format",
                "label": "Output format",
                "type": "select",
                "required": False,
                "options": [
                    {"id": "format-option-1", "label": "PDF", "value": "PDF"},
                    {"id": "format-option-2", "label": "Markdown", "value": "Markdown"},
                ],
            },
        ],
    }


def test_non_interactive_run_suppresses_clarification_and_continues_the_model_loop():
    ask = call(
        "ask_clarification",
        {"question": "May I delete it?", "clarification_type": "risk_confirmation"},
        "unattended-1",
    )
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[ask]),
            AIMessage(content="BLOCKED: deletion requires explicit authorization."),
        ]
    )
    agent = LeadAgentFactory(AppConfig(default_model="m"), model_factory=FakeModelFactory(model)).build()

    result = agent.graph.invoke(
        {"messages": [HumanMessage("clean up the old data")]},
        context={"non_interactive": True},
    )

    guidance = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert "human is not present" in guidance.content
    assert result["messages"][-1].content == "BLOCKED: deletion requires explicit authorization."


def test_run_service_exposes_non_interactive_mode_for_internal_workers():
    ask = call(
        "ask_clarification",
        {"question": "Which option?", "clarification_type": "approach_choice"},
        "scheduled-1",
    )
    model = ToolCapableFakeModel(
        responses=[AIMessage(content="", tool_calls=[ask]), AIMessage(content="Used the safest default.")]
    )
    persistence = Persistence.in_memory()
    agent = LeadAgentFactory(
        AppConfig(default_model="m"),
        model_factory=FakeModelFactory(model),
        checkpointer=persistence.checkpointer,
    ).build()
    service = RunService(agent, persistence)

    result = service.invoke("run unattended", thread_id="scheduled", non_interactive=True)

    assert result.text == "Used the safest default."


def test_prompt_contains_the_clarification_rules():
    assert "ask_clarification" in PromptBuilder("Base.").build()
