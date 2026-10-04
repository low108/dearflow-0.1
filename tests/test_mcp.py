import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig, McpConfig, McpServerConfig, expand_env
from dearflow.mcp import McpToolLoader, build_connection

SERVER = str(Path(__file__).parent / "support" / "mcp_demo_server.py")


def demo(**extra) -> dict:
    return {"command": sys.executable, "args": [SERVER], **extra}


def mcp_config(**servers) -> McpConfig:
    return McpConfig.model_validate({"enabled": True, "load_timeout_seconds": 30, "servers": servers})


# --- config -------------------------------------------------------------------


def test_expand_env(monkeypatch):
    monkeypatch.setenv("TOKEN", "abc")
    assert expand_env("Bearer $TOKEN and ${TOKEN}") == "Bearer abc and abc"
    with pytest.raises(ValueError, match="NOPE"):
        expand_env("$NOPE")


def test_server_config_requires_its_transport_fields():
    with pytest.raises(ValueError, match="command"):
        McpServerConfig()
    with pytest.raises(ValueError, match="url"):
        McpServerConfig(type="http")
    with pytest.raises(ValueError):
        McpServerConfig(command="x", comand="typo")


def test_build_connection_fills_env_and_picks_transport(monkeypatch):
    monkeypatch.setenv("API_KEY", "k1")
    stdio = build_connection(McpServerConfig(command="npx", args=["-y", "srv"], env={"KEY": "$API_KEY"}))
    http = build_connection(McpServerConfig(type="http", url="http://x/mcp", headers={"Authorization": "Bearer $API_KEY"}))
    sse = build_connection(McpServerConfig(type="sse", url="http://x/sse"))

    assert stdio == {"transport": "stdio", "command": "npx", "args": ["-y", "srv"], "env": {"KEY": "k1"}}
    assert http["transport"] == "streamable_http" and http["headers"] == {"Authorization": "Bearer k1"}
    assert sse["transport"] == "sse"


# --- loader (against a real stdio MCP server) ---------------------------------


def test_loader_discovers_prefixed_tools_and_calls_them_sync_and_async():
    import asyncio

    tools = {t.name: t for t in McpToolLoader(mcp_config(demo=demo())).tools()}

    assert set(tools) == {"demo_add", "demo_secret"}
    assert "Add two numbers" in tools["demo_add"].description
    assert "4" in str(tools["demo_add"].invoke({"a": 1, "b": 3}))  # sync path
    assert "9" in str(asyncio.run(tools["demo_add"].ainvoke({"a": 4, "b": 5})))  # async path


def test_prefix_can_be_turned_off_and_env_reaches_the_server(monkeypatch):
    monkeypatch.setenv("REAL_SECRET", "s3cret")
    config = mcp_config(demo=demo(tool_name_prefix=False, env={"DEMO_SECRET": "$REAL_SECRET"}))
    tools = {t.name: t for t in McpToolLoader(config).tools()}

    assert set(tools) == {"add", "secret"}
    assert "s3cret" in str(tools["secret"].invoke({}))


def test_broken_server_is_skipped_and_others_still_load(caplog):
    config = mcp_config(
        broken={"command": "definitely-not-a-real-command-xyz"},
        unset={"command": sys.executable, "env": {"K": "$UNSET_VAR_FOR_TEST"}},
        off={**demo(), "enabled": False},
        good=demo(),
    )
    names = {t.name for t in McpToolLoader(config).tools()}

    assert names == {"good_add", "good_secret"}
    assert "broken" in caplog.text and "unset" in caplog.text


def test_loader_caches_and_respects_enabled_flag():
    off = McpToolLoader(McpConfig(servers={"demo": McpServerConfig(**demo())}))  # enabled defaults to False
    assert off.tools() == []

    loader = McpToolLoader(mcp_config(demo=demo()))
    assert loader.tools()[0] is loader.tools()[0]  # second call reuses the first load


# --- in the agent -------------------------------------------------------------


def test_agent_can_call_an_mcp_tool_and_subagents_inherit_it():
    call = {"name": "demo_add", "args": {"a": 2, "b": 40}, "id": "m1", "type": "tool_call"}
    model = ToolCapableFakeModel(responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="It is 42")])
    app = AppConfig(default_model="fake", mcp=mcp_config(demo=demo()))
    factory = LeadAgentFactory(app, model_factory=FakeModelFactory(model))

    result = factory.build().graph.invoke({"messages": [{"role": "user", "content": "2+40?"}]})

    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "42" in str(tool_result.content) and tool_result.status != "error"
    assert "demo_add" in model.bound_tool_names


def test_subagents_inherit_mcp_tools():
    delegate = {"name": "task", "args": {"prompt": "add 2 and 40", "subagent_type": "general-purpose"}, "id": "t1", "type": "tool_call"}
    add = {"name": "demo_add", "args": {"a": 2, "b": 40}, "id": "m1", "type": "tool_call"}
    seen_tool_results: list[str] = []

    class Recording(ToolCapableFakeModel):
        def _generate(self, messages, *args, **kwargs):
            seen_tool_results.extend(str(m.content) for m in messages if isinstance(m, ToolMessage))
            return super()._generate(messages, *args, **kwargs)

    model = Recording(
        responses=[
            AIMessage(content="", tool_calls=[delegate]),  # lead delegates
            AIMessage(content="", tool_calls=[add]),  # subagent calls the MCP tool
            AIMessage(content="the sum is 42"),  # subagent report
            AIMessage(content="done"),  # lead
        ]
    )
    app = AppConfig(default_model="fake", mcp=mcp_config(demo=demo()), subagents={"enabled": True})
    result = LeadAgentFactory(app, model_factory=FakeModelFactory(model)).build().graph.invoke(
        {"messages": [{"role": "user", "content": "go"}]}
    )

    report = next(m for m in result["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "t1")
    assert report.content == "Task Succeeded. Result: the sum is 42"
    assert any("42" in r and "Task Succeeded" not in r for r in seen_tool_results)  # the subagent really got 42 back
