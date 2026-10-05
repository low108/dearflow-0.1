from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.persistence import Persistence
from dearflow.service import RunService
from dearflow.tools import ToolRegistry


class RecordingModel(ToolCapableFakeModel):
    requests: list = []

    def _generate(self, messages, *args, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def call(value: str = "same", call_id: str = "c1") -> dict:
    return {
        "name": "lookup",
        "args": {"value": value},
        "id": call_id,
        "type": "tool_call",
    }


@tool
def lookup(value: str) -> str:
    """Return a value for loop-detection tests."""
    LOOKUPS.append(value)
    return value


LOOKUPS: list[str] = []


def build(responses, *, loop_detection):
    model = RecordingModel(responses=responses, requests=[])
    config = AppConfig(default_model="fake", loop_detection=loop_detection)
    agent = LeadAgentFactory(
        config,
        model_factory=FakeModelFactory(model),
        tool_registry=ToolRegistry(config, extra_tools=[lookup]),
    ).build()
    return agent, model


def test_repeated_tool_calls_warn_the_next_model_without_persisting_the_warning():
    LOOKUPS.clear()
    agent, model = build(
        [
            AIMessage(content="", tool_calls=[call(call_id="c1")]),
            AIMessage(content="", tool_calls=[call(call_id="c2")]),
            AIMessage(content="I will stop repeating."),
        ],
        loop_detection={"warn_threshold": 2, "hard_limit": 4},
    )

    result = agent.graph.invoke({"messages": [HumanMessage("find it")]})

    warning = [
        message
        for message in model.requests[2]
        if isinstance(message, HumanMessage) and "LOOP DETECTED" in str(message.content)
    ]
    assert len(warning) == 1
    assert not any(
        isinstance(message, HumanMessage) and "LOOP DETECTED" in str(message.content)
        for message in result["messages"]
    )


def test_hard_limit_removes_the_repeated_call_before_the_tool_runs_again():
    LOOKUPS.clear()
    agent, _ = build(
        [
            AIMessage(content="", tool_calls=[call(call_id="c1")]),
            AIMessage(content="", tool_calls=[call(call_id="c2")]),
            AIMessage(content="", tool_calls=[call(call_id="c3")]),
            AIMessage(content="this response should not be needed"),
        ],
        loop_detection={"warn_threshold": 2, "hard_limit": 3},
    )

    result = agent.graph.invoke({"messages": [HumanMessage("find it")]})

    assert LOOKUPS == ["same", "same"]
    assert result["messages"][-1].tool_calls == []
    assert "FORCED STOP" in str(result["messages"][-1].content)


def test_run_record_exposes_that_loop_detection_capped_the_run():
    LOOKUPS.clear()
    agent, _ = build(
        [
            AIMessage(content="", tool_calls=[call(call_id="c1")]),
            AIMessage(content="", tool_calls=[call(call_id="c2")]),
            AIMessage(content="", tool_calls=[call(call_id="c3")]),
        ],
        loop_detection={"warn_threshold": 2, "hard_limit": 3},
    )
    persistence = Persistence.in_memory()
    service = RunService(agent, persistence)

    result = service.invoke("find it", thread_id="looped")

    assert result.usage["stop_reason"] == "loop_capped"
    assert persistence.runs.get(result.run_id).stop_reason == "loop_capped"


def test_same_tool_with_changing_arguments_counts_as_progress():
    LOOKUPS.clear()
    agent, _ = build(
        [
            AIMessage(content="", tool_calls=[call("page-1", "c1")]),
            AIMessage(content="", tool_calls=[call("page-2", "c2")]),
            AIMessage(content="", tool_calls=[call("page-3", "c3")]),
            AIMessage(content="finished"),
        ],
        loop_detection={"warn_threshold": 2, "hard_limit": 3},
    )

    result = agent.graph.invoke({"messages": [HumanMessage("read every page")]})

    assert LOOKUPS == ["page-1", "page-2", "page-3"]
    assert result["messages"][-1].content == "finished"
