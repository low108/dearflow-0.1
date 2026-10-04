import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.memory import Exchange, Fact, FactExtractor, MemoryManager, MemoryStore, parse_facts
from dearflow.middlewares import MemoryMiddleware
from dearflow.middlewares.messages import injected_kind, injected_message


def facts_json(*items):
    return json.dumps({"facts": [{"content": c, "category": "preference", "confidence": conf} for c, conf in items]})


# --- store --------------------------------------------------------------------


def test_store_merges_duplicates_and_keeps_higher_confidence(tmp_path):
    store = MemoryStore(tmp_path)
    store.add("u", [Fact(content="Likes tea", confidence=0.7)])
    kept = store.add("u", [Fact(content="  likes   TEA ", confidence=0.95), Fact(content="Uses Python")])

    assert sorted(f.content for f in kept) == ["Likes tea", "Uses Python"]
    assert next(f for f in kept if f.content == "Likes tea").confidence == 0.95
    assert len(MemoryStore(tmp_path).load("u")) == 2  # really on disk


def test_store_cap_drops_the_weakest(tmp_path):
    store = MemoryStore(tmp_path, max_facts=2)
    store.add("u", [Fact(content="a", confidence=0.9), Fact(content="b", confidence=0.5), Fact(content="c", confidence=0.8)])

    assert sorted(f.content for f in store.load("u")) == ["a", "c"]


def test_store_users_are_separate_and_paths_are_safe(tmp_path):
    store = MemoryStore(tmp_path)
    store.add("ann", [Fact(content="a")])
    store.add("../evil", [Fact(content="b")])

    assert [f.content for f in store.load("ann")] == ["a"]
    assert store.path_for("../evil").resolve().is_relative_to(tmp_path.resolve())


def test_store_render_respects_token_budget_and_orders_by_confidence(tmp_path):
    store = MemoryStore(tmp_path)
    store.add("u", [Fact(content="low " * 20, confidence=0.71), Fact(content="Top fact", confidence=0.99)])

    text = store.render_context("u", max_tokens=10)
    assert "Top fact" in text and "low" not in text
    assert store.render_context("nobody", max_tokens=100) == ""


def test_store_delete(tmp_path):
    store = MemoryStore(tmp_path)
    fact = store.add("u", [Fact(content="x")])[0]

    assert store.delete("u", fact.id) is True and store.delete("u", fact.id) is False
    assert store.load("u") == []


# --- parsing ------------------------------------------------------------------


def test_parse_facts_handles_fences_think_blocks_and_threshold():
    raw = "<think>hmm</think>```json\n" + facts_json(("Likes tea", 0.9), ("Maybe sleeps", 0.3)) + "\n```"

    assert [f.content for f in parse_facts(raw, threshold=0.7)] == ["Likes tea"]


def test_parse_facts_survives_garbage():
    assert parse_facts("not json", threshold=0.5) == []
    assert parse_facts('{"facts": "oops"}', threshold=0.5) == []
    assert parse_facts('{"facts": [{"nope": 1}, {"content": "ok", "category": "weird"}]}', threshold=0.5)[0].category == "context"


# --- manager ------------------------------------------------------------------


def manager_for(tmp_path, responses, **kw):
    model = ToolCapableFakeModel(responses=[AIMessage(content=r) for r in responses])
    manager = MemoryManager(MemoryStore(tmp_path), FactExtractor(model), debounce_seconds=0, **kw)
    return manager, model


def test_quick_turns_are_batched_into_one_extraction(tmp_path):
    manager, _ = manager_for(tmp_path, [facts_json(("Likes tea", 0.9))])
    batches = []
    real_extract = manager._extractor.extract
    manager._extractor.extract = lambda exchanges: batches.append(exchanges) or real_extract(exchanges)
    manager.enqueue("u", [Exchange("I like tea")])
    manager.enqueue("u", [Exchange("and I code in Python")])
    manager.flush()

    assert [len(b) for b in batches] == [2]  # ONE extraction call covering both exchanges
    assert "Likes tea" in manager.context("u")


def test_extraction_failure_never_raises(tmp_path):
    manager, _ = manager_for(tmp_path, [])  # the fake model has nothing to say -> raises inside
    manager.enqueue("u", [Exchange("hi")])
    manager.flush()  # must not raise

    assert manager.context("u") == ""


# --- middleware ---------------------------------------------------------------


def user(text="hi", id="u1"):
    return HumanMessage(content=text, id=id)


def test_memory_is_injected_before_the_user_message_once(tmp_path):
    manager, _ = manager_for(tmp_path, [])
    manager._store.add("default", [Fact(content="Likes tea")])
    middleware = MemoryMiddleware(manager)

    remove, block, again = middleware.before_agent({"messages": [user()]}, None)["messages"]
    assert "Likes tea" in block.content and injected_kind(block) == "memory"
    assert again.content == "hi" and remove.id == "u1"

    history = [block, again, AIMessage(content="ok"), user("next", "u2")]
    assert middleware.before_agent({"messages": history}, None) is None  # unchanged -> no repeat

    manager._store.add("default", [Fact(content="Uses Python")])
    assert middleware.before_agent({"messages": history}, None) is not None  # changed -> refreshed


def test_no_injection_when_memory_is_empty_or_disabled(tmp_path):
    manager, _ = manager_for(tmp_path, [])
    assert MemoryMiddleware(manager).before_agent({"messages": [user()]}, None) is None
    manager._store.add("default", [Fact(content="x")])
    assert MemoryMiddleware(manager, inject=False).before_agent({"messages": [user()]}, None) is None


def test_after_agent_queues_only_the_real_exchange(tmp_path):
    manager, _ = manager_for(tmp_path, [])
    queued = []
    manager.enqueue = lambda uid, exchanges: queued.append((uid, exchanges))
    call = {"name": "x", "args": {}, "id": "1", "type": "tool_call"}
    messages = [
        user("old question", "u0"),
        AIMessage(content="old answer"),
        injected_message("<memory>hidden</memory>", "memory"),
        user("I live in Penang", "u1"),
        AIMessage(content="", tool_calls=[call]),
        AIMessage(content="Nice place!"),
    ]
    MemoryMiddleware(manager).after_agent({"messages": messages}, None)

    assert queued == [("default", [Exchange("I live in Penang", "Nice place!")])]


# --- end to end ---------------------------------------------------------------


def test_agent_learns_in_one_run_and_recalls_in_the_next(tmp_path):
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(content="Nice to meet you, Ann!"),  # run 1: lead
            AIMessage(content=facts_json(("Name is Ann", 0.95))),  # background extraction
            AIMessage(content="Hello again, Ann"),  # run 2: lead
        ]
    )
    factory = LeadAgentFactory(
        AppConfig(default_model="fake", memory={"enabled": True, "storage_path": str(tmp_path / "mem"), "debounce_seconds": 0}),
        model_factory=FakeModelFactory(model),
        checkpointer=InMemorySaver(),
    )
    agent = factory.build()
    run = {"configurable": {"thread_id": "t1", "user_id": "ann"}}

    agent.graph.invoke({"messages": [{"role": "user", "content": "I'm Ann"}]}, run)
    factory.flush_memory()
    second = factory.build().graph.invoke({"messages": [{"role": "user", "content": "hi again"}]}, {"configurable": {"thread_id": "t2", "user_id": "ann"}})

    assert (tmp_path / "mem" / "ann" / "memory.json").exists()
    hidden = [m.content for m in second["messages"] if injected_kind(m) == "memory"]
    assert len(hidden) == 1 and "Name is Ann" in hidden[0]
    assert second["messages"][-1].content == "Hello again, Ann"


def test_memory_is_off_by_default():
    factory = LeadAgentFactory(AppConfig(default_model="fake"), model_factory=FakeModelFactory(ToolCapableFakeModel(responses=[])))
    assert factory.memory is None


def test_debounce_timer_fires_on_its_own(tmp_path):
    import time

    model = ToolCapableFakeModel(responses=[AIMessage(content=facts_json(("Likes tea", 0.9)))])
    manager = MemoryManager(MemoryStore(tmp_path), FactExtractor(model), debounce_seconds=0.05)
    manager.enqueue("u", [Exchange("I like tea")])

    for _ in range(100):
        if manager.context("u"):
            break
        time.sleep(0.02)
    assert "Likes tea" in manager.context("u")
