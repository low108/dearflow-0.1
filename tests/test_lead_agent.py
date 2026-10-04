from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from dearflow.agents import AgentDependencies, LeadAgentAssembly, assemble_lead_agent
from dearflow.config import AppConfig, ModelConfig


class ToolCapableFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools, *, tool_choice=None, **_kwargs):
        return self


def test_assemble_lead_agent_returns_an_invokable_graph_without_sandbox_state():
    model = FakeMessagesListChatModel(responses=[AIMessage(content="Hello from DearFlow")])
    dependencies = AgentDependencies(create_model=lambda _name: model)

    assembly = assemble_lead_agent(
        {"configurable": {"thread_id": "thread-1"}},
        app_config=AppConfig(default_model="test-model"),
        dependencies=dependencies,
    )

    assert isinstance(assembly, LeadAgentAssembly)
    assert assembly.effective_model == "test-model"

    result = assembly.graph.invoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert result["messages"][-1].content == "Hello from DearFlow"
    assert "sandbox" not in result


def test_runtime_model_override_is_used_for_construction():
    requested_models: list[str] = []

    def create_model(name: str):
        requested_models.append(name)
        return FakeMessagesListChatModel(responses=[AIMessage(content="Selected")])

    assembly = assemble_lead_agent(
        {"configurable": {"model_name": "requested-model"}},
        app_config=AppConfig(default_model="default-model"),
        dependencies=AgentDependencies(create_model=create_model),
    )

    assert requested_models == ["requested-model"]
    assert assembly.effective_model == "requested-model"


def test_runtime_model_alias_resolves_to_its_configured_model():
    requested_models: list[str] = []

    assembly = assemble_lead_agent(
        {"configurable": {"model_name": "reasoning"}},
        app_config=AppConfig(
            default_model="fast",
            models={
                "fast": ModelConfig("fast", "openai:ilmu-mini-v3.3"),
                "reasoning": ModelConfig("reasoning", "openai:ilmu-v3.1"),
            },
        ),
        dependencies=AgentDependencies(
            create_model=lambda name: requested_models.append(name)
            or FakeMessagesListChatModel(responses=[AIMessage(content="Selected")])
        ),
    )

    assert requested_models == ["openai:ilmu-v3.1"]
    assert assembly.effective_model == "openai:ilmu-v3.1"


def test_assembled_agent_executes_tools_and_returns_to_the_model():
    @tool
    def double(value: int) -> int:
        """Double an integer."""
        return value * 2

    model = ToolCapableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "double",
                        "args": {"value": 21},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The answer is 42."),
        ]
    )
    assembly = assemble_lead_agent(
        {},
        app_config=AppConfig(default_model="test-model"),
        dependencies=AgentDependencies(create_model=lambda _name: model, tools=(double,)),
    )

    result = assembly.graph.invoke({"messages": [{"role": "user", "content": "Double 21"}]})

    assert any(isinstance(message, ToolMessage) and message.content == "42" for message in result["messages"])
    assert result["messages"][-1].content == "The answer is 42."
