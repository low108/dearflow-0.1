import sqlite3

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.cli import run_chat
from dearflow.config import AppConfig, TokenBudgetConfig
from dearflow.gateway import create_app
from dearflow.persistence import Database, Persistence
from dearflow.service import RunService
from dearflow.tools import ToolRegistry
from dearflow.usage import RunUsage, UsageCallback


def usage(total_in, total_out):
    return {"input_tokens": total_in, "output_tokens": total_out, "total_tokens": total_in + total_out}


def reply(text="", tools=(), tokens=(0, 0)):
    calls = [{"name": n, "args": a, "id": f"c{i}", "type": "tool_call"} for i, (n, a) in enumerate(tools)]
    return AIMessage(content=text, tool_calls=calls, usage_metadata=usage(*tokens) if any(tokens) else None)


def task(id="t1"):
    return ("task", {"prompt": "do it", "subagent_type": "general-purpose"})


calls_made = []


@tool
def lookup(q: str = "x") -> str:
    """Look something up."""
    calls_made.append(q)
    return "found"


class Recording(ToolCapableFakeModel):
    """Keeps the messages of every model call (as (type, name, content) tuples)."""

    requests: list = []

    def _generate(self, messages, *args, **kwargs):
        self.requests.append([(m.type, getattr(m, "name", None), m.content) for m in messages])
        return super()._generate(messages, *args, **kwargs)


def service_for(responses, *, persistence=None, **config):
    calls_made.clear()
    persistence = persistence or Persistence.in_memory()
    model = Recording(responses=responses, requests=[])
    app = AppConfig(default_model="fake", **config)
    factory = LeadAgentFactory(
        app,
        model_factory=FakeModelFactory(model),
        tool_registry=ToolRegistry(app, extra_tools=[lookup]),
        checkpointer=persistence.checkpointer,
    )
    agent = factory.build()
    return RunService(agent, persistence, track_usage=app.token_usage.enabled), persistence, model


BUDGET = {"enabled": True, "max_tokens": 1000}


# --- counting -----------------------------------------------------------------------


def test_usage_callback_splits_tokens_by_caller_and_model():
    counter = RunUsage("r1")
    callback = UsageCallback(counter)
    from uuid import uuid4

    from langchain_core.outputs import ChatGeneration, LLMResult

    def call(metadata, tokens):
        rid = uuid4()
        callback.on_chat_model_start({}, [[]], run_id=rid, metadata=metadata)
        message = AIMessage(content="x", usage_metadata=usage(*tokens), response_metadata={"model_name": metadata.get("m", "gpt")})
        callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=rid)

    call({"langgraph_node": "model"}, (10, 5))
    call({"langgraph_node": "model", "dearflow_caller": "subagent"}, (20, 10))
    call({"langgraph_node": "TitleMiddleware.after_agent"}, (3, 2))

    snapshot = counter.snapshot()
    assert (snapshot["input_tokens"], snapshot["output_tokens"], snapshot["total_tokens"], snapshot["llm_calls"]) == (33, 17, 50, 3)
    assert (snapshot["lead_agent_tokens"], snapshot["subagent_tokens"], snapshot["middleware_tokens"]) == (15, 30, 5)
    assert snapshot["by_model"] == {"gpt": 50}


def test_run_record_end_event_and_result_carry_the_totals():
    service, persistence, _ = service_for([reply("Hi", tokens=(100, 20))])
    events = list(service.stream("hello", thread_id="t1"))

    end = events[-1].data
    assert end["usage"]["total_tokens"] == 120 and end["usage"]["lead_agent_tokens"] == 120
    run = persistence.runs.list("t1")[0]
    assert (run.total_input_tokens, run.total_output_tokens, run.total_tokens, run.llm_call_count) == (100, 20, 120, 1)
    assert run.lead_agent_tokens == 120 and run.stop_reason is None


def test_subagent_and_middleware_tokens_are_counted_separately():
    service, persistence, _ = service_for(
        [
            reply(tools=[task()], tokens=(10, 5)),  # lead decides to delegate
            reply("sub report", tokens=(20, 10)),  # subagent
            reply("Done", tokens=(10, 5)),  # lead answers
            reply("Trip plan", tokens=(3, 2)),  # the title model
        ],
        subagents={"enabled": True},
        context={"title": {"use_model": True}},
    )
    result = service.invoke("go", thread_id="t1")

    assert result.usage["lead_agent_tokens"] == 30 and result.usage["subagent_tokens"] == 30
    assert result.usage["middleware_tokens"] == 5 and result.usage["total_tokens"] == 65
    assert persistence.runs.get(result.run_id).subagent_tokens == 30


def test_counting_can_be_turned_off():
    service, persistence, _ = service_for([reply("Hi", tokens=(100, 20))], token_usage={"enabled": False})
    service.invoke("hello", thread_id="t1")

    assert persistence.runs.list("t1")[0].total_tokens == 0


# --- budget -------------------------------------------------------------------------


def test_warning_is_shown_to_the_next_model_call_but_not_stored():
    service, _, model = service_for(
        [reply(tools=[("lookup", {"q": "a"})], tokens=(500, 350)), reply("Wrapped up")],  # 850 of 1000 = 85%
        token_budget=BUDGET,
    )
    result = service.invoke("research", thread_id="t1")

    first, second = model.requests
    assert not any(name == "budget_warning" for _, name, _ in first)
    warnings = [content for _, name, content in second if name == "budget_warning"]
    assert len(warnings) == 1 and "85%" in warnings[0] and "850 of your 1,000" in warnings[0]
    assert result.usage["stop_reason"] is None and calls_made == ["a"]  # the tool still ran
    assert not any(getattr(m, "name", None) == "budget_warning" for m in result.state["messages"])  # not in the history


def test_warning_is_sent_only_once():
    service, _, model = service_for(
        [reply(tools=[("lookup", {})], tokens=(400, 420)), reply(tools=[("lookup", {})], tokens=(10, 10)), reply("Done")],
        token_budget=BUDGET,
    )
    service.invoke("research", thread_id="t1")

    sent = [any(name == "budget_warning" for _, name, _ in request) for request in model.requests]
    assert sent == [False, True, False]


def test_hard_stop_removes_tool_calls_and_the_run_ends_with_an_answer():
    service, persistence, model = service_for(
        [reply("Partial findings.", tools=[("lookup", {"q": "never"})], tokens=(700, 500)), reply("should not be reached")],
        token_budget=BUDGET,
    )
    result = service.invoke("research", thread_id="t1")

    assert calls_made == [] and len(model.requests) == 1  # tool never ran, no second model call
    assert "Partial findings." in result.text and "[TOKEN BUDGET EXCEEDED]" in result.text and "1,200" in result.text
    assert result.state["messages"][-1].tool_calls == []
    assert result.usage["stop_reason"] == "token_capped"
    run = persistence.runs.get(result.run_id)
    assert run.status == "success" and run.stop_reason == "token_capped"  # a capped run is a normal finish


def test_a_final_answer_over_budget_is_marked_capped():
    service, _, _ = service_for([reply("All done.", tokens=(900, 500))], token_budget=BUDGET)
    result = service.invoke("research", thread_id="t1")

    assert "All done." in result.text and "[TOKEN BUDGET EXCEEDED]" in result.text
    assert result.usage["stop_reason"] == "token_capped"


def test_output_only_limit():
    service, _, _ = service_for(
        [reply(tools=[("lookup", {})], tokens=(10, 600)), reply("nope")],
        token_budget={"enabled": True, "max_tokens": 200_000, "max_output_tokens": 500},
    )
    result = service.invoke("go", thread_id="t1")

    assert calls_made == [] and "output token usage (600)" in result.text


def test_each_run_gets_a_fresh_budget():
    service, _, _ = service_for(
        [reply("First.", tools=[("lookup", {})], tokens=(700, 500)), reply("Second.", tokens=(5, 5))],
        token_budget=BUDGET,
    )
    first = service.invoke("one", thread_id="t1")
    second = service.invoke("two", thread_id="t1")

    assert first.usage["stop_reason"] == "token_capped" and second.usage["stop_reason"] is None
    assert second.usage["total_tokens"] == 10  # not 1210


def test_subagents_share_the_runs_budget():
    service, persistence, _ = service_for(
        [
            reply(tools=[task()], tokens=(100, 50)),  # lead delegates
            reply("Sub notes.", tools=[("lookup", {"q": "deep"})], tokens=(800, 400)),  # subagent blows the budget
            reply("Final answer", tokens=(10, 10)),  # lead continues
        ],
        token_budget=BUDGET,
        subagents={"enabled": True},
    )
    result = service.invoke("go", thread_id="t1")

    assert calls_made == []  # the subagent's tool call was removed
    report = next(m.content for m in result.state["messages"] if m.type == "tool")
    assert "Sub notes." in report and "[TOKEN BUDGET EXCEEDED]" in report  # the lead sees why it stopped
    assert result.usage["stop_reason"] == "token_capped" and persistence.runs.get(result.run_id).stop_reason == "token_capped"


def test_budget_does_nothing_without_a_run_service():
    model = Recording(responses=[reply("Hi", tools=[("lookup", {})], tokens=(5000, 5000)), reply("ok")], requests=[])
    app = AppConfig(default_model="fake", token_budget=BUDGET)
    agent = LeadAgentFactory(app, model_factory=FakeModelFactory(model), tool_registry=ToolRegistry(app, extra_tools=[lookup])).build()
    calls_made.clear()
    agent.graph.invoke({"messages": [{"role": "user", "content": "hi"}]})  # nobody counts tokens here

    assert calls_made == ["x"]


# --- config -------------------------------------------------------------------------


def test_budget_config_is_validated():
    with pytest.raises(ValueError, match="hard_stop_threshold"):
        TokenBudgetConfig(warn_threshold=0.9, hard_stop_threshold=0.5)
    with pytest.raises(ValueError):
        TokenBudgetConfig(max_tokens=10)
    with pytest.raises(ValueError, match="token_usage"):
        AppConfig(default_model="x", token_budget=BUDGET, token_usage={"enabled": False})


# --- storage, API, CLI ------------------------------------------------------------------


def test_old_databases_get_the_new_columns(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE threads (thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, status TEXT NOT NULL DEFAULT 'idle', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, user_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'running', model_name TEXT, error TEXT, first_human_message TEXT, last_ai_message TEXT, message_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        "INSERT INTO runs VALUES ('r1','t1','u','success',NULL,NULL,NULL,NULL,0,'x','x');"
    )
    connection.commit()
    connection.close()

    database = Database(path)
    row = database.execute("SELECT total_tokens, stop_reason, token_usage_by_model FROM runs")[0]
    assert tuple(row) == (0, None, "{}")  # old rows survive with defaults


def test_gateway_reports_tokens_per_run_and_per_thread():
    model = ToolCapableFakeModel(responses=[reply("one", tokens=(100, 20)), reply("two", tokens=(50, 10))])
    app = create_app(AppConfig(default_model="fake"), model_factory=FakeModelFactory(model), persistence=Persistence.in_memory())
    body = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    with TestClient(app) as client:
        client.post("/api/threads/t1/runs/wait", json=body)
        client.post("/api/threads/t1/runs/wait", json=body)
        runs = client.get("/api/threads/t1/runs").json()
        usage = client.get("/api/threads/t1/token-usage").json()
        other = client.get("/api/threads/t1/token-usage", headers={"X-User-Id": "bob"})

    assert sorted(r["total_tokens"] for r in runs) == [60, 120]
    assert usage["total_tokens"] == 180 and usage["total_input_tokens"] == 150 and usage["total_runs"] == 2
    assert usage["by_caller"] == {"lead_agent": 180, "subagent": 0, "middleware": 0}
    assert other.status_code == 404


def test_cli_shows_tokens_and_budget_stop(tmp_path):
    lines: list[str] = []
    feed = iter(["go", "quit"])
    model = ToolCapableFakeModel(responses=[reply("Partial.", tools=[("lookup", {})], tokens=(900, 800))])
    app = AppConfig(default_model="fake", token_budget=BUDGET)
    run_chat(
        app_config=app,
        thread_id="t1",
        model_factory=FakeModelFactory(model),
        stream=True,
        read=lambda _p: next(feed),
        write=lines.append,
        emit=lambda _t: None,
    )

    assert "  tokens: 900 in / 800 out" in lines and "  stopped early: token budget reached" in lines
