import asyncio
import json
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.config import AppConfig
from dearflow.gateway import create_app
from dearflow.gateway.routers.runs import _sse
from dearflow.persistence import Persistence


class GatedModel(ToolCapableFakeModel):
    """Blocks inside the model call until `gate` is set, so tests can hold a run open."""

    gate: Any = None

    def _generate(self, *args, **kwargs):
        if self.gate is not None:
            assert self.gate.wait(10), "test never released the model"
        return super()._generate(*args, **kwargs)


def body(text="hello", **extra):
    return {"input": {"messages": [{"role": "user", "content": text}]}, **extra}


def parse_sse(text: str) -> list[dict]:
    events = []
    for frame in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in fields:
            events.append({"event": fields["event"], "data": json.loads(fields["data"]), "id": int(fields["id"])})
    return events


def wait_for_status(client, thread, run_id, *, want_not="running", headers=None):
    for _ in range(200):
        run = client.get(f"/api/threads/{thread}/runs/{run_id}", headers=headers).json()
        if run["status"] != want_not:
            return run
        time.sleep(0.02)
    raise AssertionError("run did not finish")


@pytest.fixture
def make_client():
    clients = []

    def build(responses=None, *, gate=None, **config):
        model = GatedModel(responses=responses if responses is not None else [AIMessage(content="Hello there")], gate=gate)
        factory = FakeModelFactory(model)
        app = create_app(
            AppConfig(default_model="fake", gateway={"heartbeat_seconds": 0.2, **config.pop("gateway", {})}, **config),
            model_factory=factory,
            persistence=Persistence.in_memory(),
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        client.factory = factory
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


# --- health, auth, ownership ---------------------------------------------------


def test_health(make_client):
    client = make_client()

    assert client.get("/health").json() == {"status": "healthy", "service": "dearflow-gateway"}
    assert client.get("/health/ready").json() == {"status": "ready"}


def test_api_key_is_required_when_configured(make_client, monkeypatch):
    monkeypatch.setenv("GATEWAY_KEY", "s3cret")
    client = make_client(gateway={"api_key": "$GATEWAY_KEY"})

    assert client.post("/api/threads", json={}).status_code == 401
    assert client.post("/api/threads", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/threads", json={}, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # liveness stays open


def test_threads_are_private_to_their_owner(make_client):
    client = make_client()
    ann, bob = {"X-User-Id": "ann"}, {"X-User-Id": "bob"}
    client.post("/api/threads", json={"thread_id": "t1", "metadata": {"topic": "trip"}}, headers=ann)

    assert client.get("/api/threads/t1", headers=ann).status_code == 200
    assert client.get("/api/threads/t1", headers=bob).status_code == 404  # 404, not 403: ids can't be probed
    assert client.delete("/api/threads/t1", headers=bob).status_code == 404
    assert client.post("/api/threads/t1/runs", json=body(), headers=bob).status_code == 404
    assert client.post("/api/threads/search", json={}, headers=bob).json() == []
    assert client.get("/api/threads/t1", headers={"X-User-Id": "bad id!"}).status_code == 400


def test_thread_crud_and_search(make_client):
    client = make_client()
    created = client.post("/api/threads", json={"thread_id": "t1", "metadata": {"topic": "trip"}}).json()
    client.post("/api/threads", json={"thread_id": "t2"})

    assert created["thread_id"] == "t1" and created["status"] == "idle" and created["metadata"] == {"topic": "trip"}
    assert client.post("/api/threads", json={"thread_id": "t1"}).json()["metadata"] == {"topic": "trip"}  # idempotent
    assert {t["thread_id"] for t in client.post("/api/threads/search", json={}).json()} == {"t1", "t2"}
    assert [t["thread_id"] for t in client.post("/api/threads/search", json={"metadata": {"topic": "trip"}}).json()] == ["t1"]
    assert len(client.post("/api/threads/search", json={"limit": 1}).json()) == 1
    assert client.post("/api/threads", json={"thread_id": "has space"}).status_code == 422
    assert client.get("/api/threads/bad%20id").status_code == 422
    assert client.delete("/api/threads/t2").json()["success"] is True
    assert client.get("/api/threads/t2").status_code == 404


# --- request validation ----------------------------------------------------------


@pytest.mark.parametrize(
    "payload, needle",
    [
        (body(stream_mode=["debug"]), "Unsupported stream mode"),
        (body(multitask_strategy="interrupt"), "multitask_strategy"),
        ({**body(), "surprise": 1}, "surprise"),
        ({"input": {"messages": []}}, "user message"),
        ({"input": {"messages": [{"role": "assistant", "content": "hi"}]}}, "user message"),
        (body(on_disconnect="explode"), "on_disconnect"),
    ],
)
def test_bad_run_requests_are_rejected_by_name(make_client, payload, needle):
    response = make_client().post("/api/threads/t1/runs", json=payload)

    assert response.status_code == 422 and needle in response.text


# --- running ---------------------------------------------------------------------


def test_background_run_then_poll_then_read_messages(make_client):
    client = make_client()
    started = client.post("/api/threads/t1/runs", json=body("plan my trip")).json()

    assert started["thread_id"] == "t1" and started["run_id"]
    run = wait_for_status(client, "t1", started["run_id"])
    assert run["status"] == "success" and run["message_count"] >= 2
    messages = client.get("/api/threads/t1/messages").json()
    assert [(m["type"], m["content"]) for m in messages] == [("human", "plan my trip"), ("ai", "Hello there")]
    assert client.get("/api/threads/t1").json()["values"]["title"] == "plan my trip"
    assert [r["run_id"] for r in client.get("/api/threads/t1/runs").json()] == [started["run_id"]]


def test_stream_endpoint_sends_sse_with_content_location(make_client):
    client = make_client()
    with client.stream("POST", "/api/threads/t1/runs/stream", json=body(stream_mode=["messages"])) as response:
        text = "".join(response.iter_text())
        location = response.headers["content-location"]
        content_type = response.headers["content-type"]

    events = parse_sse(text)
    assert content_type.startswith("text/event-stream")
    assert [e["event"] for e in events][0] == "metadata" and events[-1] == {"event": "end", "data": {"status": "success"}, "id": events[-1]["id"]}
    assert location == f"/api/threads/t1/runs/{events[0]['data']['run_id']}"
    assert any(e["event"] == "messages" and e["data"]["message"]["content"] == "Hello there" for e in events)


def test_wait_endpoint_returns_final_state_or_failure(make_client):
    client = make_client()
    state = client.post("/api/threads/t1/runs/wait", json=body()).json()

    assert state["messages"][-1]["content"] == "Hello there" and state["title"] == "hello"

    failing = make_client(responses=[])  # the model raises
    result = failing.post("/api/threads/t1/runs/wait", json=body()).json()
    assert result["status"] == "error" and result["error"]


def test_context_overrides_pick_the_model(make_client):
    client = make_client()
    client.post("/api/threads/t1/runs/wait", json=body(context={"model_name": "fancy", "ignored": 1}))

    assert "fancy" in client.factory.requested


def test_run_log_endpoint_filters_and_pages(make_client):
    client = make_client()
    client.post("/api/threads/t1/runs/wait", json=body())
    run_id = client.get("/api/threads/t1/runs").json()[0]["run_id"]

    events = client.get(f"/api/threads/t1/runs/{run_id}/events").json()
    assert events[0]["event_type"] == "metadata" and events[-1]["event_type"] == "end"
    assert {e["event_type"] for e in client.get(f"/api/threads/t1/runs/{run_id}/events?event_types=updates").json()} == {"updates"}
    assert [e["seq"] for e in client.get(f"/api/threads/t1/runs/{run_id}/events?after_seq={events[0]['seq']}").json()][0] > events[0]["seq"]
    assert client.get("/api/threads/t2/runs/" + run_id + "/events").status_code == 404  # wrong thread


# --- concurrency: busy, cancel, join ------------------------------------------------


def test_second_run_on_a_busy_thread_gets_409(make_client):
    gate = threading.Event()
    client = make_client(gate=gate)
    first = client.post("/api/threads/t1/runs", json=body()).json()
    try:
        assert client.post("/api/threads/t1/runs", json=body()).status_code == 409
        assert client.delete("/api/threads/t1").status_code == 409
        assert client.get("/api/threads/t1").json()["status"] == "busy"
    finally:
        gate.set()
    assert wait_for_status(client, "t1", first["run_id"])["status"] == "success"
    assert client.post("/api/threads/t1/runs/wait", json=body()).status_code in (200,)  # free again (model wraps around)


def test_cancel_stops_the_run_and_frees_the_thread(make_client):
    gate = threading.Event()
    client = make_client(gate=gate)
    run = client.post("/api/threads/t1/runs", json=body()).json()

    assert client.post(f"/api/threads/t1/runs/{run['run_id']}/cancel").status_code == 202
    gate.set()  # the model call in flight finishes; the run stops at the next event
    finished = wait_for_status(client, "t1", run["run_id"])
    assert finished["status"] == "interrupted" and finished["stop_reason"] == "interrupted"
    assert client.get("/api/threads/t1").json()["status"] == "idle"
    assert client.post(f"/api/threads/t1/runs/{run['run_id']}/cancel").status_code == 409  # already over
    assert client.post(f"/api/threads/t1/runs/{run['run_id']}/cancel?action=rollback").status_code == 422


def test_join_replays_from_the_start_and_outlives_memory(make_client):
    client = make_client()
    run = client.post("/api/threads/t1/runs", json=body(stream_mode=["messages"])).json()
    wait_for_status(client, "t1", run["run_id"])

    with client.stream("GET", f"/api/threads/t1/runs/{run['run_id']}/join") as response:
        joined = parse_sse("".join(response.iter_text()))
    assert joined[0]["event"] == "metadata" and joined[-1]["event"] == "end"  # from the very beginning
    assert any(e["event"] == "messages" for e in joined)

    forgetful = make_client(gateway={"retain_seconds": 0})  # finished runs leave memory at once
    run = forgetful.post("/api/threads/t1/runs/wait", json=body()) and forgetful.get("/api/threads/t1/runs").json()[0]
    with forgetful.stream("GET", f"/api/threads/t1/runs/{run['run_id']}/join") as response:
        replay = parse_sse("".join(response.iter_text()))
    assert replay[0]["event"] == "metadata" and replay[-1]["data"] == {"status": "success"}  # from the run log
    assert forgetful.get("/api/threads/t1/runs/nope/join").status_code == 404


def test_disconnect_policy_cancels_only_when_asked():
    closed = []

    async def consume_then_leave():
        stream = _sse(None, iter([_Event(), None, _Event()]), on_close=lambda: closed.append("cancelled"))
        await anext(stream)
        await stream.aclose()  # the client went away mid-stream

    asyncio.run(consume_then_leave())
    assert closed == ["cancelled"]


class _Event:
    def sse(self):
        return "event: x\ndata: {}\nid: 1\n\n"


def test_idle_subscribers_get_heartbeats_and_late_ones_get_history():
    from dearflow.gateway.run_manager import ActiveRun
    from dearflow.streaming import StreamEvent

    run = ActiveRun("r1", "t1", "ann")
    subscription = run.subscribe(heartbeat=0.05)
    run.publish(StreamEvent(1, "metadata", {}))

    assert next(subscription).event == "metadata"
    assert next(subscription) is None  # nothing happened for 50 ms: heartbeat
    run.publish(StreamEvent(2, "end", {"status": "success"}))
    run.finish()
    assert [e.event for e in subscription] == ["end"]  # then it ends by itself

    assert [e.event for e in run.subscribe()] == ["metadata", "end"]  # a late joiner sees everything
    assert run.status == "success"


def test_sse_turns_heartbeats_into_comments():
    async def collect():
        return [frame async for frame in _sse(None, iter([_Event(), None]))]

    assert asyncio.run(collect()) == ["event: x\ndata: {}\nid: 1\n\n", ": heartbeat\n\n"]


def test_real_server_streams_incrementally_and_honours_on_disconnect():
    """TestClient can't do this: it runs the app to completion first. Use a real uvicorn."""
    import httpx
    import uvicorn

    gate = threading.Event()
    model = GatedModel(responses=[AIMessage(content="Hello there")], gate=gate)
    app = create_app(
        AppConfig(default_model="fake", gateway={"heartbeat_seconds": 0.3}),
        model_factory=FakeModelFactory(model),
        persistence=Persistence.in_memory(),
    )
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    base = "http://127.0.0.1:%d" % server.servers[0].sockets[0].getsockname()[1]
    payload = body(stream_mode=["messages"])

    def status_of(thread_id):
        for _ in range(100):
            run = httpx.get(f"{base}/api/threads/{thread_id}/runs").json()[0]
            if run["status"] != "running":
                return run["status"]
            time.sleep(0.03)

    try:
        with httpx.stream("POST", f"{base}/api/threads/t1/runs/stream", json=payload, timeout=10) as response:
            assert next(response.iter_lines()) == "event: metadata"  # arrives while the model is still blocked
        time.sleep(0.3)  # the client is gone; default policy is to cancel
        gate.set()
        assert status_of("t1") == "interrupted"

        gate.clear()
        with httpx.stream("POST", f"{base}/api/threads/t2/runs/stream", json={**payload, "on_disconnect": "continue"}, timeout=10) as response:
            next(response.iter_lines())
        gate.set()
        assert status_of("t2") == "success"  # `continue`: leaving does not stop the run
    finally:
        gate.set()
        server.should_exit = True
        thread.join(5)
