import pytest
from langchain_core.messages import AIMessage, HumanMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.cli import list_threads, main, run_chat
from dearflow.config import AppConfig, PersistenceConfig
from dearflow.persistence import Database, Persistence, RunStore, ThreadStore
from dearflow.service import RunService, ThreadBusyError


class Recording(ToolCapableFakeModel):
    """Remembers the human messages it was shown on each call."""

    seen: list = []

    def _generate(self, messages, *args, **kwargs):
        self.seen.append([m.content for m in messages if isinstance(m, HumanMessage) and not m.additional_kwargs.get("hide_from_ui")])
        return super()._generate(messages, *args, **kwargs)


def session(path, responses, **config):
    """One 'process': persistence + agent + service, like run_chat builds them."""
    persistence = Persistence.from_config(PersistenceConfig(enabled=True, path=str(path)))
    model = Recording(responses=[AIMessage(content=r) for r in responses], seen=[])
    app = AppConfig(default_model="fake", **config)
    agent = LeadAgentFactory(app, model_factory=FakeModelFactory(model), checkpointer=persistence.checkpointer).build()
    return RunService(agent, persistence), persistence, model


# --- tables -------------------------------------------------------------------


def test_thread_store_ensure_list_title_and_claim():
    threads = ThreadStore(Database())
    threads.ensure("t1", "ann")
    threads.ensure("t1", "bob")  # second ensure does not steal ownership
    threads.ensure("t2", "ann")
    threads.set_title("t1", "Trip plan")

    assert threads.get("t1").user_id == "ann" and threads.get("t1").title == "Trip plan"
    assert {t.thread_id for t in threads.list("ann")} == {"t1", "t2"} and threads.list("bob") == []
    assert threads.claim("t1") is True and threads.claim("t1") is False  # only one run at a time
    threads.set_status("t1", "idle")
    assert threads.claim("t1") is True


def test_deleting_a_thread_removes_its_runs():
    db = Database()
    threads, runs = ThreadStore(db), RunStore(db)
    threads.ensure("t1", "ann")
    runs.start("t1", "ann")
    threads.delete("t1")

    assert threads.get("t1") is None and runs.list("t1") == []


def test_run_store_lifecycle_and_crash_recovery():
    db = Database()
    runs = RunStore(db)
    done, crashed = runs.start("t1", "ann", "m"), runs.start("t2", "ann", "m")
    runs.finish(done, "success", last_ai_message="hi", message_count=2)

    assert runs.get(done).status == "success" and runs.get(done).last_ai_message == "hi"
    assert runs.interrupt_unfinished() == ["t2"]  # only the one still running
    assert runs.get(crashed).status == "interrupted" and "stopped" in runs.get(crashed).error
    assert [r.run_id for r in runs.list("t1")] == [done]


def test_startup_frees_threads_left_busy_by_a_crash(tmp_path):
    path = tmp_path / "db.sqlite"
    first = Persistence.from_config(PersistenceConfig(enabled=True, path=str(path)))
    first.threads.ensure("t1", "ann")
    first.threads.claim("t1")
    first.runs.start("t1", "ann")
    first.close()  # "crash": the run never finished

    second = Persistence.from_config(PersistenceConfig(enabled=True, path=str(path)))
    assert second.threads.get("t1").status == "idle"
    assert second.runs.list("t1")[0].status == "interrupted"


# --- service ------------------------------------------------------------------


def test_run_is_recorded_with_title_and_preview(tmp_path):
    service, persistence, _ = session(tmp_path / "db.sqlite", ["Sure, here is a plan"])
    result = service.invoke("plan my trip", thread_id="t1", user_id="ann")

    run = persistence.runs.get(result.run_id)
    assert (run.status, run.first_human_message, run.last_ai_message) == ("success", "plan my trip", "Sure, here is a plan")
    assert run.message_count >= 2 and run.model_name == "fake"
    thread = persistence.threads.get("t1")
    assert thread.title == "plan my trip" and thread.status == "idle"  # title comes from the title middleware


def test_failed_run_is_recorded_and_frees_the_thread(tmp_path):
    service, persistence, _ = session(tmp_path / "db.sqlite", [])  # the model has nothing to say -> raises

    with pytest.raises(Exception):
        service.invoke("hi", thread_id="t1")

    run = persistence.runs.list("t1")[0]
    assert run.status == "error" and run.error
    assert persistence.threads.get("t1").status == "idle"  # can run again


def test_second_run_on_a_busy_thread_is_rejected(tmp_path):
    service, persistence, _ = session(tmp_path / "db.sqlite", ["ok"])
    persistence.threads.ensure("t1", "default")
    persistence.threads.claim("t1")  # someone else is running

    with pytest.raises(ThreadBusyError):
        service.invoke("hi", thread_id="t1")
    assert persistence.runs.list("t1") == []  # the rejected call left no run behind


# --- the point: surviving a restart --------------------------------------------


def test_conversation_survives_a_restart(tmp_path):
    path = tmp_path / "db.sqlite"
    service, persistence, _ = session(path, ["Nice to meet you, Ann"])
    service.invoke("I am Ann", thread_id="t1", user_id="ann")
    persistence.close()  # process ends

    service, persistence, model = session(path, ["Your name is Ann"])  # new process
    history = service.history("t1")
    assert [m.content for m in history] == ["I am Ann", "Nice to meet you, Ann"]  # hidden messages excluded

    service.invoke("what is my name?", thread_id="t1", user_id="ann")
    assert all(text.startswith("--- BEGIN USER INPUT ---") for text in model.seen[0])
    assert "I am Ann" in model.seen[0][0] and "what is my name?" in model.seen[0][1]
    assert [t.thread_id for t in persistence.threads.list("ann")] == ["t1"]
    assert len(persistence.runs.list("t1")) == 2


def test_in_memory_mode_needs_no_file_and_forgets_on_exit(tmp_path):
    persistence = Persistence.from_config(PersistenceConfig())  # disabled

    assert type(persistence.checkpointer).__name__ == "InMemorySaver"
    persistence.threads.ensure("t1", "ann")
    assert list(tmp_path.iterdir()) == []  # nothing written to disk


# --- CLI ----------------------------------------------------------------------


def chat(tmp_path, responses, inputs, **kw):
    out: list[str] = []
    feed = iter(inputs)
    config = AppConfig(default_model="fake", persistence={"enabled": True, "path": str(tmp_path / "cli.db")})
    model = ToolCapableFakeModel(responses=[AIMessage(content=r) for r in responses])
    run_chat(app_config=config, thread_id="t1", model_factory=FakeModelFactory(model), read=lambda _p: next(feed), write=out.append, **kw)
    return out


def test_cli_resumes_and_lists_threads(tmp_path):
    first = chat(tmp_path, ["Hello Ann"], ["I am Ann", "quit"])
    second = chat(tmp_path, ["Hi again"], ["hello", "quit"])

    assert not any("Resuming" in line for line in first)
    assert any("Resuming thread 't1' — I am Ann (1 earlier messages)" in line for line in second)

    listed: list[str] = []
    list_threads(AppConfig(default_model="fake", persistence={"enabled": True, "path": str(tmp_path / "cli.db")}), "default", listed.append)
    assert len(listed) == 1 and listed[0].startswith("t1\t") and listed[0].endswith("I am Ann")


def test_cli_flags_enable_persistence(tmp_path):
    seen = []
    cfg = tmp_path / "c.yaml"
    cfg.write_text('default_model: fast\nmodels:\n  fast: {model: "p:m"}\n', encoding="utf-8")

    main(["--config", str(cfg), "--persist"], chat_runner=lambda **kw: seen.append(kw["app_config"].persistence.enabled))
    main(["--config", str(cfg)], chat_runner=lambda **kw: seen.append(kw["app_config"].persistence.enabled))

    assert seen == [True, False]
