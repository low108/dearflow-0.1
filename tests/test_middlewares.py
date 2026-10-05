from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.middlewares import InputSanitizationMiddleware, MiddlewareChain, ToolErrorHandlingMiddleware
from dearflow.tools import ToolRegistry


def test_chain_preserves_order_and_default_starts_at_the_input_boundary():
    first, second = ToolErrorHandlingMiddleware(), ToolErrorHandlingMiddleware()
    assert MiddlewareChain().add(first).add(second).build() == [first, second]
    default = MiddlewareChain.default().build()
    assert isinstance(default[0], InputSanitizationMiddleware)
    assert any(isinstance(item, ToolErrorHandlingMiddleware) for item in default)


def test_tool_errors_are_returned_to_the_model():
    @tool
    def explode() -> str:
        """Raise a predictable failure."""
        raise RuntimeError("boom")

    call = {"name": "explode", "args": {}, "id": "c1", "type": "tool_call"}
    model = ToolCapableFakeModel(
        responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="Recovered.")]
    )
    config = AppConfig(default_model="m")
    agent = LeadAgentFactory(
        config,
        model_factory=FakeModelFactory(model),
        tool_registry=ToolRegistry(config, extra_tools=[explode]),
    ).build()

    result = agent.graph.invoke({"messages": [{"role": "user", "content": "go"}]})

    errors = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert errors[0].status == "error"
    assert errors[0].content == "Tool 'explode' failed: boom"
    assert result["messages"][-1].content == "Recovered."
