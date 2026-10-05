"""Exercise trace correlation through the real service, graph, tools and subagents."""

import asyncio
from copy import deepcopy
from uuid import UUID

import pytest
import requests
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client, tracing_context

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig, ObservabilityConfig
from dearflow.middlewares import ClarificationMiddleware, InputSanitizationMiddleware
from dearflow.observability import instrument_middlewares
from dearflow.persistence import Persistence
from dearflow.service import RunService
from dearflow.tools import ToolRegistry


class TraceRecorder(BaseCallbackHandler):
    """Use LangChain's real callback propagation without a network exporter."""

    def __init__(self):
        self.runs = {}

    def _start(self, kind, run_id, parent_run_id, metadata, name, inputs):
        self.runs[run_id] = {
            "kind": kind,
            "parent": parent_run_id,
            "metadata": dict(metadata or {}),
            "name": name,
            "inputs": inputs,
        }

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, name=None, **kwargs):
        self._start("chain", run_id, parent_run_id, metadata, name, inputs)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, name=None, **kwargs):
        self._start("model", run_id, parent_run_id, metadata, name, messages)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, metadata=None, name=None, **kwargs):
        self._start("tool", run_id, parent_run_id, metadata, name or serialized["name"], input_str)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.runs[run_id]["outputs"] = outputs

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.runs[run_id]["outputs"] = output

    def on_llm_end(self, response, *, run_id, **kwargs):
        self.runs[run_id]["outputs"] = response

    def ancestors(self, run_id):
        parents = []
        parent = self.runs[run_id]["parent"]
        while parent is not None:
            assert parent not in parents, "tracing hierarchy contains a cycle"
            parents.append(parent)
            parent = self.runs[parent]["parent"]
        return parents


@pytest.fixture(autouse=True)
def disable_network_export(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@tool
def trace_lookup(query: str) -> str:
    """Return a deterministic search result for trace integration tests."""
    return f"found {query}"


def reply(text="", *, tool_call=None, tokens=(1, 1)):
    return AIMessage(
        content=text,
        tool_calls=[tool_call] if tool_call else [],
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)},
    )


def service_for(responses):
    persistence = Persistence.in_memory()
    app = AppConfig(
        default_model="fake",
        context={"date_reminder": False, "summarization": {"enabled": False}, "title": {"enabled": False}},
        subagents={"enabled": True},
        observability={"middleware_snapshots": True, "capture_content": True, "max_snapshot_chars": 20_000},
    )
    model = ToolCapableFakeModel(responses=responses)
    agent = LeadAgentFactory(
        app,
        model_factory=FakeModelFactory(model),
        tool_registry=ToolRegistry(app, extra_tools=[trace_lookup]),
        checkpointer=persistence.checkpointer,
    ).build()
    recorder = TraceRecorder()
    # The service adds its UsageCallback at invocation time; both must survive.
    agent.graph = agent.graph.with_config(callbacks=[recorder])
    return RunService(agent, persistence), persistence, recorder


@pytest.mark.parametrize("streaming", [False, True])
def test_service_trace_preserves_run_identity_subagent_hierarchy_and_usage(streaming):
    task_call = {
        "name": "task",
        "id": "delegate-1",
        "args": {"prompt": "Find the answer", "subagent_type": "general-purpose"},
    }
    lookup_call = {"name": "trace_lookup", "id": "lookup-1", "args": {"query": "the answer"}}
    service, persistence, recorder = service_for(
        [
            reply(tool_call=task_call, tokens=(11, 2)),
            reply(tool_call=lookup_call, tokens=(7, 1)),
            reply("The answer is 42", tokens=(5, 3)),
            reply("Done: 42", tokens=(13, 4)),
        ]
    )

    if streaming:
        events = list(service.stream("Research this", thread_id="trace-thread", modes=["values", "custom"]))
        run_id = events[0].data["run_id"]
        usage = events[-1].data["usage"]
        assert events[-1].data["status"] == "success"
        assert {event.data["type"] for event in events if event.event == "custom"} >= {"subagent.start", "subagent.end"}
    else:
        result = service.invoke("Research this", thread_id="trace-thread")
        run_id, usage = result.run_id, result.usage
        assert result.text == "Done: 42"

    root_id = UUID(run_id)
    root = recorder.runs[root_id]
    assert root["name"] == "DearFlow"
    assert root["parent"] is None
    assert root["metadata"]["thread_id"] == "trace-thread"
    assert root["metadata"]["dearflow_run_id"] == run_id
    assert usage["total_tokens"] == 46
    assert usage["lead_agent_tokens"] == 30
    assert usage["subagent_tokens"] == 16
    assert usage["llm_calls"] == 4
    assert persistence.runs.get(run_id).total_tokens == 46

    subagent_id = next(rid for rid, run in recorder.runs.items() if run["name"] == "subagent:general-purpose")
    task_id = next(rid for rid, run in recorder.runs.items() if run["kind"] == "tool" and run["name"] == "task")
    lookup_id = next(rid for rid, run in recorder.runs.items() if run["kind"] == "tool" and run["name"] == "trace_lookup")
    assert task_id in recorder.ancestors(subagent_id)
    assert subagent_id in recorder.ancestors(lookup_id)

    snapshots = {rid: run for rid, run in recorder.runs.items() if (run["name"] or "").startswith("middleware.")}
    assert snapshots
    assert any("InputSanitizationMiddleware" in run["name"] and subagent_id in recorder.ancestors(rid) for rid, run in snapshots.items())
    assert any("ToolErrorHandlingMiddleware" in run["name"] and subagent_id in recorder.ancestors(rid) for rid, run in snapshots.items())
    assert any(
        "InputSanitizationMiddleware" in run["name"] and subagent_id not in recorder.ancestors(rid) for rid, run in snapshots.items()
    )

    models = [run for run in recorder.runs.values() if run["kind"] == "model"]
    assert len(models) == 4
    assert len([run for run in models if run["metadata"].get("dearflow_caller") == "subagent"]) == 2
    for rid, run in recorder.runs.items():
        assert run["metadata"]["thread_id"] == "trace-thread"
        assert run["metadata"]["dearflow_run_id"] == run_id
        if rid != root_id:
            assert root_id in recorder.ancestors(rid)


def test_separate_turns_get_distinct_durable_trace_roots_on_the_same_thread():
    service, persistence, recorder = service_for([reply("First", tokens=(3, 2)), reply("Second", tokens=(7, 4))])
    first = service.invoke("one", thread_id="conversation")
    second = service.invoke("two", thread_id="conversation")

    assert first.run_id != second.run_id
    assert first.usage["total_tokens"] == 5
    assert second.usage["total_tokens"] == 11
    roots = {rid: run for rid, run in recorder.runs.items() if run["parent"] is None}
    assert set(roots) == {UUID(first.run_id), UUID(second.run_id)}
    assert {run["metadata"]["thread_id"] for run in roots.values()} == {"conversation"}
    assert {run["metadata"]["dearflow_run_id"] for run in roots.values()} == {first.run_id, second.run_id}
    assert [run.total_tokens for run in persistence.runs.list("conversation")] == [11, 5]


class OfflineLangSmithClient(Client):
    """Capture the SDK export boundary while using real RunTree and tracer code."""

    def __init__(self, session):
        super().__init__(
            api_url="http://langsmith.invalid",
            api_key="offline-test-key",
            session=session,
            auto_batch_tracing=False,
            info={},
        )
        self.records = {}

    def create_run(self, name, inputs, run_type, **kwargs):
        self.records[str(kwargs["id"])] = deepcopy({"name": name, "inputs": inputs, "run_type": run_type, **kwargs})

    def update_run(self, run_id, **kwargs):
        self.records.setdefault(str(run_id), {}).update(deepcopy({key: value for key, value in kwargs.items() if value is not None}))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_langsmith_sdk_exports_snapshot_children_and_suppresses_native_hook_payloads(asynchronous, monkeypatch):
    session = requests.Session()

    def deny_network(*args, **kwargs):
        pytest.fail("offline tracing test attempted a network request")

    monkeypatch.setattr(session, "send", deny_network)
    client = OfflineLangSmithClient(session)
    tracer = LangChainTracer(client=client, project_name="dearflow-offline-test")
    graph = create_agent(
        ToolCapableFakeModel(responses=[reply("done", tokens=(3, 2))]),
        middleware=instrument_middlewares(
            [InputSanitizationMiddleware(), ClarificationMiddleware()],
            ObservabilityConfig(middleware_snapshots=True, capture_content=True),
        ),
    )
    config = {"callbacks": [tracer], "metadata": {"thread_id": "offline-thread", "dearflow_run_id": "offline-run"}}
    inputs = {"messages": [{"role": "user", "content": "hello"}]}
    try:
        # Explicit context enables the SDK while the process environment stays
        # disabled. The injected client catches both native and custom spans.
        with tracing_context(enabled=True, client=client, project_name="dearflow-offline-test"):
            result = asyncio.run(graph.ainvoke(inputs, config)) if asynchronous else graph.invoke(inputs, config)

        assert result["messages"][0].content == "hello"
        assert result["messages"][-1].content == "done"
        hook = "awrap_model_call" if asynchronous else "wrap_model_call"
        native_id, native = next((rid, run) for rid, run in client.records.items() if run["name"] == f"InputSanitizationMiddleware.{hook}")
        custom_id, custom = next(
            (rid, run) for rid, run in client.records.items() if run["name"] == f"middleware.InputSanitizationMiddleware.{hook}"
        )
        assert str(custom["parent_run_id"]) == native_id
        assert native["inputs"] == native["outputs"] == {"details": "See middleware snapshot child span"}
        assert custom["inputs"]["before"]["value"]["messages"][0]["content"] == "hello"
        forwarded = custom["outputs"]["calls"][0]["request"]["value"]["messages"][0]["content"]
        assert forwarded == "--- BEGIN USER INPUT ---\nhello\n--- END USER INPUT ---"
        assert custom["outputs"]["result"]["value"]["result"][0]["content"] == "done"
        assert custom["extra"]["metadata"]["thread_id"] == "offline-thread"
        assert custom["extra"]["metadata"]["dearflow_run_id"] == "offline-run"
        assert custom.get("error") is None
        assert custom["end_time"] >= custom["start_time"]

        model = next(run for run in client.records.values() if run["run_type"] == "llm")
        assert str(model["parent_run_id"]) == custom_id
        assert model["extra"]["metadata"]["usage_metadata"]["total_tokens"] == 5
        native_node = next(run for run in client.records.values() if run["name"] == "ClarificationMiddleware.after_model")
        assert native_node["inputs"] == native_node["outputs"] == {"details": "See middleware snapshot child span"}
    finally:
        client.close()
        session.close()
