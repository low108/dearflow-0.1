import copy
import json
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from dearflow.observability.snapshots import Snapshotter
from dearflow.pii import DETECTORS, Redactor


def test_content_snapshot_is_redacted_and_never_mutates_input():
    message = HumanMessage(content="Email alice@example.com with sk-abcdefghijklmnopqrstuvwx")
    state = {"messages": [message], "password": "plain-secret", "nested": {"apiKey": "another-secret"}}
    original = copy.deepcopy(state)
    snapshotter = Snapshotter(capture_content=True, redactor=Redactor(DETECTORS, "test-secret-long-enough"))
    snapshot = snapshotter(state)
    encoded = json.dumps(snapshot)
    assert snapshot["value"]["messages"][0]["content"].startswith("Email [EMAIL_")
    assert "alice@example.com" not in encoded
    assert "sk-abcdefghijklmnopqrstuvwx" not in encoded
    assert "plain-secret" not in encoded and "another-secret" not in encoded
    assert state == original
    message.content = "changed"
    assert "changed" not in json.dumps(snapshot)
    assert snapshot["content_captured"] is True
    assert snapshot["truncated"] is False


def test_metadata_snapshot_excludes_message_arguments_and_state_text():
    snapshot = Snapshotter()(
        {
            "messages": [
                AIMessage(content="private answer", tool_calls=[{"id": "call-1", "name": "search", "args": {"query": "private query"}}])
            ],
            "state_text": "private state",
        }
    )
    encoded = json.dumps(snapshot)
    assert all(secret not in encoded for secret in ("private answer", "private query", "private state"))
    assert snapshot["content_captured"] is False
    assert snapshot["value"]["messages"][0]["type"] == "ai"


def test_model_request_has_only_explicit_safe_fields():
    request = ModelRequest(
        model=SimpleNamespace(api_key="model-secret"),
        messages=[HumanMessage(content="question")],
        system_message=SystemMessage(content="system instructions"),
        tools=[{"type": "function", "function": {"name": "search", "description": "private-description"}}],
        runtime=SimpleNamespace(context={"password": "runtime-secret"}),
        model_settings={"temperature": 0.2, "api_key": "settings-secret"},
    )
    snapshot = Snapshotter(capture_content=True)(request)
    value = snapshot["value"]
    assert value["messages"][0]["content"] == "question"
    assert value["system_message"]["content"] == "system instructions"
    assert value["tools"] == ["search"]
    assert value["model_settings"]["temperature"] == 0.2
    assert all(
        secret not in json.dumps(snapshot) for secret in ("model-secret", "runtime-secret", "settings-secret", "private-description")
    )


def test_response_tool_request_and_command_preserve_inspectable_content():
    snapshotter = Snapshotter(capture_content=True)
    response = snapshotter(ModelResponse(result=[AIMessage(content="answer")], structured_response={"answer": "value"}))
    assert response["value"]["result"][0]["content"] == "answer"
    assert response["value"]["structured_response"] == {"answer": "value"}
    request = ToolCallRequest(
        tool_call={"name": "search", "id": "t1", "args": {"q": "query"}},
        tool=None,
        state={"messages": [HumanMessage(content="ask")], "hidden": "private-state"},
        runtime=SimpleNamespace(secret="private-runtime"),
    )
    tool_snapshot = snapshotter(request)
    assert tool_snapshot["value"]["tool_call"]["args"]["q"] == "query"
    assert tool_snapshot["value"]["state"]["messages"][0]["content"] == "ask"
    assert "private-state" not in json.dumps(tool_snapshot)
    assert "private-runtime" not in json.dumps(tool_snapshot)
    command = snapshotter(Command(update={"messages": [ToolMessage(content="result", tool_call_id="t1")]}, resume="private-resume"))
    assert command["value"]["update"]["messages"][0]["content"] == "result"
    assert "private-resume" not in json.dumps(command)


def test_unknown_objects_and_binary_content_are_never_represented():
    class Sensitive:
        def __repr__(self):
            raise AssertionError("must never call repr")

    value = {
        "unknown": Sensitive(),
        "binary": b"private-binary",
        "data": "data:image/png;base64,private-image",
        "content": [{"type": "image", "source_type": "base64", "data": "aGVsbG8="}],
        "anthropic_image": {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "sensitive-base64"}},
        "image_url": {"url": "https://private.example/image.png"},
    }
    encoded = json.dumps(Snapshotter(capture_content=True)(value))
    assert all(secret not in encoded for secret in ("private-binary", "private-image", "aGVsbG8=", "sensitive-base64", "private.example"))
    assert "Sensitive" in encoded


def test_total_serialized_snapshot_and_recursion_are_bounded():
    value = {f"field-{i}": ["😀" * 2000] * 30 for i in range(200)}
    snapshot = Snapshotter(capture_content=True, max_chars=256)(value)
    assert len(json.dumps(snapshot)) <= 256
    assert snapshot["truncated"] is True
    recursive = {"messages": []}
    recursive["recursive"] = recursive
    snapshot = Snapshotter(capture_content=True)(recursive)
    assert snapshot["truncated"] is True
    assert len(json.dumps(snapshot)) <= 20000


def test_invalid_snapshot_budget_is_rejected():
    with pytest.raises(ValueError):
        Snapshotter(max_chars=1)


def test_tool_inventory_is_bounded_and_ignores_malformed_entries():
    request = ModelRequest(
        model=None, messages=[], tools=[{"function": None}, {"function": "bad"}] + [{"name": f"tool-{index}"} for index in range(200)]
    )
    snapshot = Snapshotter(capture_content=True)(request)
    assert snapshot["truncated"] is True
    assert len(snapshot["value"]["tools"]) <= 100


def test_receipt_metadata_is_allowlisted_without_exposing_provider_payloads():
    from dearflow.receipts import LEDGER_KEY, RECEIPT_KEY, make_receipt

    message = ToolMessage(
        content="391", tool_call_id="calculation-1", name="calculator"
    )
    receipt = make_receipt(
        {"id": "calculation-1", "name": "calculator", "args": {}}, message
    )
    message.additional_kwargs = {
        RECEIPT_KEY: {**receipt, "unexpected": "private extra receipt data"},
        "provider_private": "private provider data",
        "authorization": "private bearer value",
    }
    reply = AIMessage(
        content="391",
        additional_kwargs={
            LEDGER_KEY: [{"id": "r1", **receipt, "extra": "private extra ledger data"}]
        },
    )
    original = copy.deepcopy([message, reply])
    snapshot = Snapshotter(capture_content=True)([message, reply])
    assert snapshot["value"][0]["additional_kwargs"] == {RECEIPT_KEY: receipt}
    assert snapshot["value"][1]["additional_kwargs"] == {
        LEDGER_KEY: [{"id": "r1", **receipt}]
    }
    assert "private" not in json.dumps(snapshot)
    assert [message, reply] == original
    message.additional_kwargs[RECEIPT_KEY]["status"] = "error"
    assert snapshot["value"][0]["additional_kwargs"][RECEIPT_KEY]["status"] == "success"


def test_redaction_marker_distinguishes_raw_from_already_redacted_input_without_leaking_pii():
    redactor = Redactor(DETECTORS, "test-secret-long-enough")
    snapshotter = Snapshotter(capture_content=True, redactor=redactor)
    before = snapshotter(HumanMessage(content="Contact demo@example.com"))
    after = snapshotter(
        HumanMessage(content=redactor.redact("Contact demo@example.com"))
    )
    assert before["value"] == after["value"]
    assert before["redaction_applied"] is True
    assert after["redaction_applied"] is False
    assert "demo@example.com" not in json.dumps([before, after])
