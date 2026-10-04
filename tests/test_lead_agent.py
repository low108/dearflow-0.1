from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgent, LeadAgentFactory
from dearflow.config import AppConfig, ModelConfig
from dearflow.tools import ToolRegistry


def _reply(text):
    return FakeMessagesListChatModel(responses=[AIMessage(content=text)])


def test_build_returns_an_invokable_agent():
    factory = LeadAgentFactory(AppConfig(default_model="m"), model_factory=FakeModelFactory(_reply("Hi")))

    agent = factory.build()

    assert isinstance(agent, LeadAgent)
    assert agent.model_name == "m"
    result = agent.graph.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    assert result["messages"][-1].content == "Hi"
    assert "sandbox" not in result


def test_runtime_model_override_and_alias_resolution():
    models = FakeModelFactory(_reply("x"))
    config = AppConfig(
        default_model="fast",
        models={"fast": ModelConfig(model="p:fast"), "reasoning": ModelConfig(model="p:slow")},
    )
    factory = LeadAgentFactory(config, model_factory=models)

    assert factory.build().model_name == "p:fast"
    assert factory.build({"configurable": {"model_name": "reasoning"}}).model_name == "p:slow"
    assert models.requested == ["p:fast", "p:slow"]


def test_agent_runs_tools_and_returns_to_the_model():
    @tool
    def double(value: int) -> int:
        """Double an integer."""
        return value * 2

    call = {"name": "double", "args": {"value": 21}, "id": "c1", "type": "tool_call"}
    model = ToolCapableFakeModel(
        responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="It is 42.")]
    )
    config = AppConfig(default_model="m")
    agent = LeadAgentFactory(
        config,
        model_factory=FakeModelFactory(model),
        tool_registry=ToolRegistry(config, extra_tools=[double]),
    ).build()

    result = agent.graph.invoke({"messages": [{"role": "user", "content": "Double 21"}]})

    assert any(isinstance(m, ToolMessage) and m.content == "42" for m in result["messages"])
    assert result["messages"][-1].content == "It is 42."
