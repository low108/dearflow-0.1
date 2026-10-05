from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig, ContextConfig
from dearflow.middlewares import (
    DynamicContextMiddleware,
    MiddlewareChain,
    TitleMiddleware,
    TodoMiddleware,
)
from dearflow.middlewares.messages import injected_kind


def clock_at(day: str):
    return lambda: datetime.fromisoformat(f"{day}T09:00:00")


def user(text="hi", id="u1"):
    return HumanMessage(content=text, id=id)


# --- date reminder -----------------------------------------------------------


def test_date_reminder_is_inserted_before_the_user_message():
    update = DynamicContextMiddleware(clock_at("2026-10-04")).before_agent({"messages": [user()]}, None)

    remove, reminder, again = update["messages"]
    assert remove.id == "u1"
    assert "2026-10-04, Sunday" in reminder.content
    assert injected_kind(reminder) == "date"
    assert again.content == "hi" and again.id != "u1" and injected_kind(again) is None


def test_date_reminder_not_repeated_on_the_same_day_but_renewed_next_day():
    middleware = DynamicContextMiddleware(clock_at("2026-10-04"))
    first = middleware.before_agent({"messages": [user()]}, None)["messages"]
    history = [first[1], first[2], AIMessage(content="hello"), user("again", "u2")]

    assert middleware.before_agent({"messages": history}, None) is None
    tomorrow = DynamicContextMiddleware(clock_at("2026-10-05"))
    assert tomorrow.before_agent({"messages": history}, None) is not None


def test_date_reminder_ignores_non_user_tail():
    middleware = DynamicContextMiddleware(clock_at("2026-10-04"))
    assert middleware.before_agent({"messages": []}, None) is None
    assert middleware.before_agent({"messages": [user(), AIMessage(content="x", id="a")]}, None) is None


# --- title -------------------------------------------------------------------


def test_title_defaults_to_shortened_first_message():
    state = {"messages": [user("x" * 100), AIMessage(content="ok")]}
    title = TitleMiddleware(max_chars=20).after_agent(state, None)["title"]

    assert title == "x" * 17 + "..."


def test_title_uses_model_and_strips_think_blocks():
    model = ToolCapableFakeModel(responses=[AIMessage(content='<think>hmm</think>"Trip Planning"')])
    state = {"messages": [user("plan my trip"), AIMessage(content="sure")]}

    assert TitleMiddleware(model=model).after_agent(state, None) == {"title": "Trip Planning"}


def test_title_falls_back_when_model_fails():
    class Boom(ToolCapableFakeModel):
        def invoke(self, *a, **k):
            raise RuntimeError("down")

    state = {"messages": [user("plan my trip"), AIMessage(content="sure")]}
    assert TitleMiddleware(model=Boom(responses=[])).after_agent(state, None) == {"title": "plan my trip"}


def test_title_only_once_and_only_after_first_exchange():
    middleware = TitleMiddleware()
    first = [user("a"), AIMessage(content="b")]

    assert middleware.after_agent({"messages": first, "title": "Done"}, None) is None
    assert middleware.after_agent({"messages": [user("a")]}, None) is None  # no reply yet
    assert middleware.after_agent({"messages": first + [user("c", "u2"), AIMessage(content="d")]}, None) is None


def test_injected_reminders_do_not_count_as_user_turns_for_title():
    reminder = TodoMiddleware._reminder([{"content": "x", "status": "pending"}])
    state = {"messages": [user("a"), AIMessage(content="b"), reminder, AIMessage(content="c")]}

    assert TitleMiddleware().after_agent(state, None) == {"title": "a"}


# --- wiring ------------------------------------------------------------------


def names(chain):
    return [type(m).__name__ for m in chain.build()]


def test_default_chain_order_with_context():
    chain = MiddlewareChain.default(plan_mode=True, context=ContextConfig(), model=ToolCapableFakeModel(responses=[]))

    assert names(chain) == [
        "InputSanitizationMiddleware",
        "DanglingToolCallMiddleware",
        "ToolErrorHandlingMiddleware",
        "DynamicContextMiddleware",
        "SummarizationMiddleware",
        "TitleMiddleware",
        "TodoMiddleware",
        "ClarificationMiddleware",
    ]


def test_context_middleware_can_be_disabled():
    context = ContextConfig.model_validate(
        {"date_reminder": False, "summarization": {"enabled": False}, "title": {"enabled": False}}
    )
    chain = MiddlewareChain.default(context=context, model=ToolCapableFakeModel(responses=[]))

    assert names(chain) == [
        "InputSanitizationMiddleware",
        "DanglingToolCallMiddleware",
        "ToolErrorHandlingMiddleware",
        "ClarificationMiddleware",
    ]


def test_real_graph_adds_date_and_title_and_keeps_history_ordered():
    model = ToolCapableFakeModel(responses=[AIMessage(content="hello"), AIMessage(content="again")])
    agent = LeadAgentFactory(
        AppConfig(default_model="fake", context={"timezone": "UTC"}),
        model_factory=FakeModelFactory(model),
        checkpointer=InMemorySaver(),
    ).build()
    config = {"configurable": {"thread_id": "t"}}

    first = agent.graph.invoke({"messages": [{"role": "user", "content": "plan my trip"}]}, config)
    second = agent.graph.invoke({"messages": [{"role": "user", "content": "thanks"}]}, config)

    kinds = [injected_kind(m) or m.type for m in second["messages"]]
    assert kinds == ["date", "human", "ai", "human", "ai"]  # one reminder, before the first message
    assert first["title"] == second["title"] == "plan my trip"
