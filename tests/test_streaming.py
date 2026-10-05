import json
from contextlib import closing

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.cli import run_chat
from dearflow.config import AppConfig
from dearflow.persistence import Persistence
from dearflow.service import RunService, ThreadBusyError
from dearflow.streaming import StreamEvent, UnsupportedStreamModeError, normalize_modes, to_jsonable


def service_for(responses, **config):
    persistence = Persistence.in_memory()
    model = ToolCapableFakeModel(responses=responses)
    app = AppConfig(default_model="fake", **config)
    agent = LeadAgentFactory(app, model_factory=FakeModelFactory(model), checkpointer=persistence.checkpointer).build()
    return RunService(agent, persistence), persistence


def names(events):
    return [e.event for e in events]


def task_call(id="t1"):
    return {
        "name": "task",
        "args": {"prompt": "x", "subagent_type": "general-purpose", "description": "look up"},
        "id": id,
        "type": "tool_call",
    }


# --- modes & format -----------------------------------------------------------


def test_modes_are_validated_aliased_and_deduplicated():
    assert normalize_modes(None) == ["values"]
    assert normalize_modes("updates") == ["updates"]
    assert normalize_modes(["messages-tuple", "messages", "values"]) == ["messages", "values"]
    with pytest.raises(UnsupportedStreamModeError, match="debug") as info:
        normalize_modes(["values", "debug"])
    assert info.value.modes == ("debug",)


def test_sse_frame_matches_the_langgraph_wire_format():
    frame = StreamEvent(3, "updates", {"text": "héllo"}).sse()

    assert frame == 'event: updates\ndata: {"text": "héllo"}\nid: 3\n\n'


def test_to_jsonable_reduces_messages_and_survives_odd_values():
    hidden = HumanMessage(content="<memory>", additional_kwargs={"hide_from_ui": True})
    data = to_jsonable({"m": [hidden], "obj": object(), "t": (1, 2)})

    assert data["m"][0] == {"type": "human", "id": None, "content": "<memory>", "hidden": True}
    assert isinstance(data["obj"], str) and data["t"] == [1, 2]
    json.dumps(data)  # really serializable


def test_chunk_serialization_uses_message_roles_and_retains_tool_deltas():
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "lookup", "args": '{"query":', "id": "call-1", "index": 0}],
    )
    data = to_jsonable(chunk)
    assert data["type"] == "ai"
    assert data["tool_call_chunks"][0]["args"] == '{"query":'
    assert chunk.type == "AIMessageChunk"  # wire normalization must not mutate provider chunks

    result = to_jsonable(ToolMessageChunk(content="Which city?", tool_call_id="call-2", name="ask_clarification"))
    assert result["type"] == "tool"
    assert result["name"] == "ask_clarification"
    assert result["tool_call_id"] == "call-2"
    assert result["content"] == "Which city?"


# --- the stream ---------------------------------------------------------------


def test_stream_order_ids_and_requested_modes_only():
    service, _ = service_for([AIMessage(content="Hello there")])
    events = list(service.stream("hi", thread_id="t1", modes=["messages"]))

    assert names(events)[0] == "metadata" and names(events)[-1] == "end"
    assert set(names(events)) == {"metadata", "messages", "end"}  # `updates` was collected but not sent
    ids = [e.id for e in events]
    assert ids == sorted(set(ids)) and ids[0] == 1  # increasing; gaps are logged-but-unsent `updates`
    answer = [e.data for e in events if e.event == "messages" and e.data["node"] == "model"]
    assert answer[-1]["message"]["content"] == "Hello there"
    assert events[-1].data["status"] == "success" and "total_tokens" in events[-1].data["usage"]
    assert events[0].data["thread_id"] == "t1"


def test_hidden_messages_and_empty_updates_are_not_sent():
    service, _ = service_for([AIMessage(content="ok")])
    events = list(service.stream("hi", thread_id="t1", modes=["messages", "updates"]))

    token_events = [e.data["message"] for e in events if e.event == "messages"]
    assert token_events and not any("current_date" in str(m["content"]) for m in token_events)  # hidden stays hidden
    hidden_in_updates = [m for e in events if e.event == "updates" for m in str(e.data).split("'hidden': True")[:-1]]
    assert hidden_in_updates  # raw `updates` still carry it, flagged `hidden`, so a UI can skip it
    assert not any(e.event == "updates" and all(v is None for v in e.data.values()) for e in events)


def test_values_mode_ends_with_the_full_final_state():
    service, _ = service_for([AIMessage(content="Hello there")])
    events = list(service.stream("hi", thread_id="t1", modes=["values"]))

    final_state = [e for e in events if e.event == "values"][-1].data
    assert final_state["messages"][-1]["content"] == "Hello there"
    assert final_state["title"] == "hi"


def test_subagent_progress_arrives_as_custom_events():
    service, _ = service_for(
        [AIMessage(content="", tool_calls=[task_call()]), AIMessage(content="report"), AIMessage(content="done")],
        subagents={"enabled": True},
    )
    events = list(service.stream("go", thread_id="t1", modes=["custom"]))

    assert [e.data["type"] for e in events if e.event == "custom"] == ["subagent.start", "subagent.end"]
    assert events[1].data == {"type": "subagent.start", "subagent_type": "general-purpose", "description": "look up"}


def test_stream_run_is_recorded_like_invoke():
    service, persistence = service_for([AIMessage(content="Hello there")])
    events = list(service.stream("hi", thread_id="t1"))

    run_id = events[0].data["run_id"]
    run = persistence.runs.get(run_id)
    assert (run.status, run.last_ai_message, run.first_human_message) == ("success", "Hello there", "hi")
    assert persistence.threads.get("t1").status == "idle" and persistence.threads.get("t1").title == "hi"


# --- the run log --------------------------------------------------------------


def test_run_log_keeps_updates_but_not_tokens_or_snapshots_and_can_resume():
    service, _ = service_for([AIMessage(content="Hello there")])
    sent = list(service.stream("hi", thread_id="t1", modes=["messages", "values"]))
    run_id = sent[0].data["run_id"]

    logged = service.events(run_id)
    assert set(names(logged)) == {"metadata", "updates", "end"}  # no messages, no values
    assert logged[0].data["run_id"] == run_id and logged[-1].data["status"] == "success"
    assert [e.id for e in service.events(run_id, after_seq=logged[1].id)] == [e.id for e in logged[2:]]


def test_deleting_a_thread_deletes_its_log():
    service, persistence = service_for([AIMessage(content="ok")])
    run_id = list(service.stream("hi", thread_id="t1"))[0].data["run_id"]
    persistence.threads.delete("t1")

    assert service.events(run_id) == []


# --- failure, cancellation, rejection ------------------------------------------


def test_failure_becomes_error_then_end_and_is_recorded():
    service, persistence = service_for([])  # the fake model raises
    events = list(service.stream("hi", thread_id="t1"))

    assert names(events)[-2:] == ["error", "end"] and events[-1].data["status"] == "error"
    assert events[-2].data["message"]
    run = persistence.runs.list("t1")[0]
    assert run.status == "error" and persistence.threads.get("t1").status == "idle"
    assert [e.event for e in service.events(run.run_id)][-2:] == ["error", "end"]


def test_abandoning_the_stream_interrupts_the_run_and_frees_the_thread():
    service, persistence = service_for([AIMessage(content="Hello there")])
    stream = service.stream("hi", thread_id="t1")
    next(stream)  # metadata
    stream.close()  # the client went away

    assert persistence.runs.list("t1")[0].status == "interrupted"
    assert persistence.threads.get("t1").status == "idle"
    assert persistence.threads.claim("t1") is True  # a new run can start on the same thread


def test_unsupported_mode_fails_before_any_run_exists():
    service, persistence = service_for([AIMessage(content="ok")])

    with pytest.raises(UnsupportedStreamModeError):
        list(service.stream("hi", thread_id="t1", modes=["debug"]))
    assert persistence.runs.list("t1") == [] and persistence.threads.get("t1") is None


def test_busy_thread_is_rejected():
    service, persistence = service_for([AIMessage(content="ok")])
    persistence.threads.ensure("t1", "default")
    persistence.threads.claim("t1")

    with pytest.raises(ThreadBusyError):
        list(service.stream("hi", thread_id="t1"))


# --- CLI ----------------------------------------------------------------------


def chat_stream(responses, inputs, **config):
    lines: list[str] = []
    chunks: list[str] = []
    feed = iter(inputs)
    model = ToolCapableFakeModel(responses=responses)
    run_chat(
        app_config=AppConfig(default_model="fake", **config),
        thread_id="t1",
        model_factory=FakeModelFactory(model),
        stream=True,
        read=lambda _p: next(feed),
        write=lines.append,
        emit=chunks.append,
    )
    return lines, "".join(chunks)


def test_cli_prints_the_answer_live():
    lines, text = chat_stream([AIMessage(content="Hello from the CLI")], ["hi", "quit"])

    assert text == "DearFlow: Hello from the CLI\n"
    assert lines[0].startswith("DearFlow CLI")


class StreamingModel(ToolCapableFakeModel):
    """A provider-style stream: text deltas followed by a usage-only chunk."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="Hello "))
        yield ChatGenerationChunk(message=AIMessageChunk(content="there!"))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata={"input_tokens": 1054, "output_tokens": 124, "total_tokens": 1178},
            )
        )


def test_streaming_sse_uses_assistant_roles_and_keeps_each_text_delta():
    with closing(Persistence.in_memory()) as persistence:
        agent = LeadAgentFactory(
            AppConfig(default_model="fake"),
            model_factory=FakeModelFactory(StreamingModel(responses=[])),
            checkpointer=persistence.checkpointer,
        ).build()
        events = list(RunService(agent, persistence).stream("hi", thread_id="t1", modes=["messages"]))
    payloads = [
        json.loads(next(line.removeprefix("data: ") for line in event.sse().splitlines() if line.startswith("data: ")))
        for event in events
        if event.event == "messages" and event.data["node"] == "model"
    ]
    assert {payload["message"]["type"] for payload in payloads} == {"ai"}
    assert [payload["message"]["content"] for payload in payloads if payload["message"]["content"]] == ["Hello ", "there!"]


@pytest.mark.parametrize("snapshots", [False, True])
def test_cli_prints_provider_stream_chunks_and_token_usage(snapshots):
    lines, chunks = [], []
    feed = iter(["hi there", "hello", "quit"])
    run_chat(
        app_config=AppConfig(default_model="fake", observability={"middleware_snapshots": snapshots}),
        thread_id="t1",
        model_factory=FakeModelFactory(StreamingModel(responses=[])),
        stream=True,
        read=lambda _p: next(feed),
        write=lines.append,
        emit=chunks.append,
    )
    assert lines.count("  tokens: 1,054 in / 124 out") == 2
    assert "".join(chunks) == "DearFlow: Hello there!\nDearFlow: Hello there!\n"


def test_cli_shows_tool_and_subagent_activity():
    responses = [AIMessage(content="", tool_calls=[task_call()]), AIMessage(content="report"), AIMessage(content="All done")]
    lines, text = chat_stream(responses, ["go", "quit"], subagents={"enabled": True})

    assert "  → task" in lines
    assert "  ⇢ subagent general-purpose: look up" in lines
    assert text == "DearFlow: All done\n"


def test_cli_prints_a_clarification_question():
    ask = {
        "name": "ask_clarification",
        "args": {"question": "Which city?", "clarification_type": "missing_info"},
        "id": "c1",
        "type": "tool_call",
    }
    _, text = chat_stream([AIMessage(content="", tool_calls=[ask])], ["plan a trip", "quit"])

    assert "Which city?" in text


def test_cli_reports_a_failed_turn_without_crashing():
    lines, _ = chat_stream([], ["hi", "quit"])

    assert any(line.startswith("  error:") for line in lines)
