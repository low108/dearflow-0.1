from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.middlewares import MiddlewareChain, ToolErrorHandlingMiddleware
from dearflow.tools import ToolRegistry


def test_chain_preserves_order_and_default_has_error_handling():
    first, second = ToolErrorHandlingMiddleware(), ToolErrorHandlingMiddleware()
    assert MiddlewareChain().add(first).add(second).build() == [first, second]
    assert isinstance(MiddlewareChain.default().build()[0], ToolErrorHandlingMiddleware)


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
