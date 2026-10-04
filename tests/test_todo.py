from langchain_core.messages import AIMessage, HumanMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.middlewares import MiddlewareChain, TodoMiddleware, ToolErrorHandlingMiddleware

OPEN = [{"content": "step 1", "status": "pending"}]
DONE = [{"content": "step 1", "status": "completed"}]


def state(messages, todos):
    return {"messages": messages, "todos": todos}


def test_guard_sends_model_back_when_todos_are_open():
    update = TodoMiddleware().after_model(state([HumanMessage("go"), AIMessage(content="done!")], OPEN), None)

    assert update["jump_to"] == "model"
    assert "step 1" in update["messages"][0].content


def test_guard_allows_finish_when_all_done_or_still_calling_tools():
    middleware = TodoMiddleware()
    finish = [HumanMessage("go"), AIMessage(content="done!")]
    working = [HumanMessage("go"), AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])]

    assert middleware.after_model(state(finish, DONE), None) is None
    assert middleware.after_model(state(finish, []), None) is None
    assert middleware.after_model(state(working, OPEN), None) is None


def test_guard_stops_after_max_reminders():
    middleware = TodoMiddleware(max_reminders=2)
    messages = [HumanMessage("go"), AIMessage(content="a")]
    for _ in range(2):
        messages += [middleware.after_model(state(messages, OPEN), None)["messages"][0], AIMessage(content="again")]

    assert middleware.after_model(state(messages, OPEN), None) is None
    # a new real user message resets the budget
    messages.append(HumanMessage("next task"))
    messages.append(AIMessage(content="x"))
    assert middleware.after_model(state(messages, OPEN), None) is not None


def test_plan_mode_adds_write_todos_and_guard_in_the_real_graph():
    write = {
        "name": "write_todos",
        "args": {"todos": OPEN},
        "id": "w1",
        "type": "tool_call",
    }
    model = ToolCapableFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[write]),
            AIMessage(content="All done"),  # too early: guard objects
            AIMessage(content="Really done"),  # guard reminded once, model gives up again
        ]
    )
    agent = LeadAgentFactory(
        AppConfig(default_model="m", plan_mode=True), model_factory=FakeModelFactory(model)
    ).build()

    result = agent.graph.invoke({"messages": [HumanMessage("do stuff")]})

    assert "write_todos" in model.bound_tool_names
    assert result["todos"] == OPEN
    reminders = [m for m in result["messages"] if isinstance(m, HumanMessage) and "system_reminder" in m.content]
    assert len(reminders) == 2  # guard fired on both early finishes, then ran out of model replies


def test_plan_mode_is_off_by_default_and_overridable_per_run():
    def tools_for(config, run_config=None):
        model = ToolCapableFakeModel(responses=[AIMessage(content="hi")])
        agent = LeadAgentFactory(config, model_factory=FakeModelFactory(model)).build(run_config)
        agent.graph.invoke({"messages": [HumanMessage("x")]})
        return model.bound_tool_names

    assert "write_todos" not in tools_for(AppConfig(default_model="m"))
    assert "write_todos" in tools_for(AppConfig(default_model="m"), {"configurable": {"is_plan_mode": True}})
    assert "write_todos" not in tools_for(
        AppConfig(default_model="m", plan_mode=True), {"configurable": {"is_plan_mode": False}}
    )


def test_default_chain_order_puts_clarification_last():
    names = [type(m).__name__ for m in MiddlewareChain.default(plan_mode=True).build()]
    assert names == ["ToolErrorHandlingMiddleware", "TodoMiddleware", "ClarificationMiddleware"]
