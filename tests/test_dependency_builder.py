from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from dearflow.agents import build_agent_dependencies
from dearflow.config import AppConfig, ToolConfig


def test_build_agent_dependencies_includes_configured_tools():
    config = AppConfig(
        default_model="test-model",
        tools=[
            ToolConfig(
                "calculator",
                "math",
                "dearflow.tools.builtins:calculator",
            )
        ],
    )

    dependencies = build_agent_dependencies(
        config,
        create_model=lambda _name: FakeMessagesListChatModel(responses=[AIMessage(content="Done")]),
    )

    assert [tool.name for tool in dependencies.tools] == ["calculator"]
