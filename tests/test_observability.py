import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import ModelResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from conftest import ToolCapableFakeModel
from dearflow.config import ObservabilityConfig
from dearflow.middlewares.input_sanitization import InputSanitizationMiddleware
from dearflow.observability import instrument_middlewares


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.spans = {}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, name=None, **kwargs):
        self.spans[run_id] = {"name": name, "inputs": inputs, "parent": parent_run_id, "metadata": metadata or {}}

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.spans[run_id]["outputs"] = outputs

    def on_chain_error(self, error, *, run_id, **kwargs):
        self.spans[run_id]["error"] = str(error)

    def snapshots(self):
        return [s for s in self.spans.values() if (s["name"] or "").startswith("middleware.")]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_snapshots_capture_forwarded_request_without_mutating_saved_history(asynchronous):
    recorder = Recorder()
    model = ToolCapableFakeModel(responses=[AIMessage(content="hello")])
    graph = create_agent(
        model,
        middleware=instrument_middlewares(
            [InputSanitizationMiddleware()], ObservabilityConfig(middleware_snapshots=True, capture_content=True)
        ),
    )
    inputs = {"messages": [HumanMessage(content="hello")]}
    config = {"callbacks": [recorder], "metadata": {"thread_id": "thread-1", "dearflow_run_id": "run-1"}}
    result = asyncio.run(graph.ainvoke(inputs, config)) if asynchronous else graph.invoke(inputs, config)
    span = recorder.snapshots()[0]
    assert span["parent"] in recorder.spans
    assert span["metadata"]["thread_id"] == "thread-1"
    assert span["inputs"]["before"]["value"]["messages"][0]["content"] == "hello"
    forwarded = span["outputs"]["calls"][0]["request"]["value"]["messages"][0]["content"]
    assert forwarded == "--- BEGIN USER INPUT ---\nhello\n--- END USER INPUT ---"
    assert result["messages"][0].content == "hello"
    assert result["messages"][-1].content == "hello"


def test_disabled_instrumentation_preserves_instances_and_emits_no_snapshot_spans():
    middleware = InputSanitizationMiddleware()
    assert instrument_middlewares([middleware], ObservabilityConfig()) == [middleware]


def test_observability_config_is_opt_in_and_rejects_unbounded_payload_settings():
    assert ObservabilityConfig().middleware_snapshots is False
    assert ObservabilityConfig().capture_content is False
    with pytest.raises(ValueError):
        ObservabilityConfig(max_snapshot_chars=0)
    with pytest.raises(ValueError):
        ObservabilityConfig(max_snapshot_chars=1_000_001)


def test_snapshot_failure_omits_content_without_interrupting_the_agent():
    class BrokenRedactor:
        def redact(self, text):
            raise ValueError("cannot redact private content")

    recorder = Recorder()
    graph = create_agent(
        ToolCapableFakeModel(responses=[AIMessage(content="done")]),
        middleware=instrument_middlewares(
            [InputSanitizationMiddleware()],
            ObservabilityConfig(middleware_snapshots=True, capture_content=True),
            redactor=BrokenRedactor(),
        ),
    )
    result = graph.invoke({"messages": [HumanMessage(content="private content")]}, {"callbacks": [recorder]})
    assert result["messages"][-1].content == "done"
    assert recorder.snapshots()[0]["inputs"]["before"] == {"unavailable": True, "error_type": "ValueError"}


class RetryAndRewrite(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        handler(request)
        result = handler(request.override(messages=[*request.messages, HumanMessage(content="retry")]))
        return ModelResponse(result=[result.result[0].model_copy(update={"content": "edited answer"})])

    async def awrap_model_call(self, request, handler):
        await handler(request)
        result = await handler(request.override(messages=[*request.messages, HumanMessage(content="retry")]))
        return ModelResponse(result=[result.result[0].model_copy(update={"content": "edited answer"})])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_repeated_handler_calls_and_response_edits_are_distinguished(asynchronous):
    recorder = Recorder()
    graph = create_agent(
        ToolCapableFakeModel(responses=[AIMessage(content="first"), AIMessage(content="second")]),
        middleware=instrument_middlewares([RetryAndRewrite()], ObservabilityConfig(middleware_snapshots=True, capture_content=True)),
    )
    inputs, config = {"messages": [HumanMessage(content="question")]}, {"callbacks": [recorder]}
    result = asyncio.run(graph.ainvoke(inputs, config)) if asynchronous else graph.invoke(inputs, config)
    span = recorder.snapshots()[0]["outputs"]
    assert span["handler_calls"] == 2
    assert span["calls"][0]["response"]["value"]["result"][0]["content"] == "first"
    assert span["calls"][1]["request"]["value"]["messages"][-1]["content"] == "retry"
    assert span["calls"][1]["response"]["value"]["result"][0]["content"] == "second"
    assert span["result"]["value"]["result"][0]["content"] == "edited answer"
    assert result["messages"][-1].content == "edited answer"


class StopEarly(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        return {"messages": [AIMessage(content="no model needed")], "jump_to": "end"}


def test_node_updates_and_jump_routing_survive_instrumentation():
    recorder = Recorder()
    graph = create_agent(
        ToolCapableFakeModel(responses=[]),
        middleware=instrument_middlewares([StopEarly()], ObservabilityConfig(middleware_snapshots=True, capture_content=True)),
    )
    result = graph.invoke({"messages": [HumanMessage(content="question")]}, {"callbacks": [recorder]})
    span = recorder.snapshots()[0]
    assert result["messages"][-1].content == "no model needed"
    assert span["outputs"]["state_update"]["value"]["jump_to"] == "end"
    assert span["inputs"]["before"]["value"]["messages"][0]["content"] == "question"


def test_tool_inventory_arguments_and_response_changes_survive_instrumentation():
    seen = []

    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        seen.append(value)
        return value

    class ToolEdits(AgentMiddleware):
        tools = [echo]

        def wrap_tool_call(self, request, handler):
            result = handler(request.override(tool_call={**request.tool_call, "args": {"value": "rewritten"}}))
            return result.model_copy(update={"content": "edited result"})

    recorder = Recorder()
    graph = create_agent(
        ToolCapableFakeModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "echo", "args": {"value": "original"}, "id": "echo-1"}]),
                AIMessage(content="done"),
            ]
        ),
        middleware=instrument_middlewares([ToolEdits()], ObservabilityConfig(middleware_snapshots=True, capture_content=True)),
    )
    result = graph.invoke({"messages": [HumanMessage(content="go")]}, {"callbacks": [recorder]})
    span = recorder.snapshots()[0]
    assert seen == ["rewritten"]
    assert span["inputs"]["before"]["value"]["tool_call"]["args"] == {"value": "original"}
    assert span["outputs"]["calls"][0]["response"]["value"]["content"] == "rewritten"
    assert span["outputs"]["result"]["value"]["content"] == "edited result"
    assert result["messages"][-2].content == "edited result"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_hook_exception_is_not_swallowed_or_retried_and_snapshot_error_is_sanitized(asynchronous):
    seen = []
    failure = ValueError("private failure payload")

    class Broken(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            seen.append("called")
            raise failure

        async def awrap_model_call(self, request, handler):
            return self.wrap_model_call(request, handler)

    recorder = Recorder()
    graph = create_agent(
        ToolCapableFakeModel(responses=[]),
        middleware=instrument_middlewares([Broken()], ObservabilityConfig(middleware_snapshots=True)),
    )
    with pytest.raises(ValueError) as caught:
        if asynchronous:
            asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="go")]}, {"callbacks": [recorder]}))
        else:
            graph.invoke({"messages": [HumanMessage(content="go")]}, {"callbacks": [recorder]})
    assert caught.value is failure
    assert seen == ["called"]
    assert recorder.snapshots()[0]["error"] == "ValueError"


def test_cancellation_propagates_without_retry():
    seen = []
    entered = asyncio.Event()

    class Cancelled(AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            seen.append("called")
            entered.set()
            await asyncio.Future()

    graph = create_agent(
        ToolCapableFakeModel(responses=[]),
        middleware=instrument_middlewares([Cancelled()], ObservabilityConfig(middleware_snapshots=True)),
    )

    async def cancel():
        task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content="go")]}))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert seen == ["called"]


@pytest.mark.parametrize("when", ["start", "end"])
def test_failing_trace_sink_never_repeats_hook_side_effects(when):
    seen = []

    class CountCalls(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            seen.append("called")
            return AIMessage(content="cached answer")

    class BrokenSink(Recorder):
        raise_error = True

        def on_chain_start(self, serialized, inputs, **kwargs):
            if when == "start" and (kwargs.get("name") or "").startswith("middleware."):
                raise RuntimeError("sink unavailable")
            super().on_chain_start(serialized, inputs, **kwargs)

        def on_chain_end(self, outputs, **kwargs):
            if when == "end" and (self.spans[kwargs["run_id"]]["name"] or "").startswith("middleware."):
                raise RuntimeError("sink unavailable")
            super().on_chain_end(outputs, **kwargs)

    graph = create_agent(
        ToolCapableFakeModel(responses=[]),
        middleware=instrument_middlewares([CountCalls()], ObservabilityConfig(middleware_snapshots=True)),
    )
    result = graph.invoke({"messages": [HumanMessage(content="go")]}, {"callbacks": [BrokenSink()]})
    assert result["messages"][-1].content == "cached answer"
    assert seen == ["called"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_receipt_stamping_is_visible_at_tool_response_boundary(asynchronous):
    from dearflow.middlewares.tool_receipt import ToolReceiptMiddleware
    from dearflow.receipts import RECEIPT_KEY

    @tool
    def arithmetic() -> str:
        """Return an example calculation."""
        return "391"

    recorder = Recorder()
    graph = create_agent(
        ToolCapableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "arithmetic", "args": {}, "id": "arithmetic-1"}
                    ],
                ),
                AIMessage(content="391"),
            ]
        ),
        tools=[arithmetic],
        middleware=instrument_middlewares(
            [ToolReceiptMiddleware()],
            ObservabilityConfig(middleware_snapshots=True, capture_content=True),
        ),
    )
    inputs, config = (
        {"messages": [HumanMessage(content="Calculate 17 times 23")]},
        {"callbacks": [recorder]},
    )
    asyncio.run(graph.ainvoke(inputs, config)) if asynchronous else graph.invoke(
        inputs, config
    )
    span = next(s for s in recorder.snapshots() if s["name"].endswith("wrap_tool_call"))
    before = span["outputs"]["calls"][0]["response"]["value"]
    after = span["outputs"]["result"]["value"]
    assert "additional_kwargs" not in before
    assert after["additional_kwargs"][RECEIPT_KEY]["status"] == "success"
    assert after["additional_kwargs"][RECEIPT_KEY]["tool_name"] == "arithmetic"
    assert before["content"] == after["content"] == "391"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_pii_redaction_change_has_safe_snapshot_evidence(asynchronous):
    from dearflow.middlewares.pii_redaction import PiiRedactionMiddleware
    from dearflow.pii import DETECTORS, Redactor

    redactor = Redactor(DETECTORS, "snapshot-test-secret-long-enough")
    recorder = Recorder()
    graph = create_agent(
        ToolCapableFakeModel(responses=[AIMessage(content="Acknowledged")]),
        middleware=instrument_middlewares(
            [PiiRedactionMiddleware(redactor)],
            ObservabilityConfig(middleware_snapshots=True, capture_content=True),
            redactor=redactor,
        ),
    )
    inputs, config = (
        {"messages": [HumanMessage(content="Contact demo@example.com")]},
        {"callbacks": [recorder]},
    )
    result = (
        asyncio.run(graph.ainvoke(inputs, config))
        if asynchronous
        else graph.invoke(inputs, config)
    )
    span = recorder.snapshots()[0]
    before = span["inputs"]["before"]
    forwarded = span["outputs"]["calls"][0]["request"]
    assert before["redaction_applied"] is True
    assert forwarded["redaction_applied"] is False
    assert (
        before["value"]["messages"][0]["content"]
        == forwarded["value"]["messages"][0]["content"]
    )
    assert "[EMAIL_" in forwarded["value"]["messages"][0]["content"]
    assert result["messages"][0].content == "Contact demo@example.com"
