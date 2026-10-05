from langgraph.graph import END, START, StateGraph

from dearflow.agents.thread_state import (
    ThreadState,
    merge_artifacts,
    merge_delegations,
    merge_goal,
    merge_skill_context,
    merge_todos,
)


def test_artifact_updates_merge_without_losing_order_or_adding_duplicates():
    existing = ["report.md", "chart.png"]
    concurrent_update = ["chart.png", "summary.txt"]

    merged = merge_artifacts(existing, concurrent_update)

    assert merged == ["report.md", "chart.png", "summary.txt"]


def test_goal_is_preserved_when_a_concurrent_node_does_not_update_it():
    active_goal = {"objective": "Build DearFlow", "status": "active"}

    assert merge_goal(active_goal, None) == active_goal


def test_explicit_goal_update_replaces_the_previous_goal():
    active_goal = {"objective": "Build DearFlow", "status": "active"}
    updated_goal = {"objective": "Ship DearFlow", "status": "active"}

    assert merge_goal(active_goal, updated_goal) == updated_goal


def test_todos_preserve_unrelated_writes_but_allow_explicit_replacement_and_clear():
    current = [{"content": "Run tests", "status": "pending"}]
    replacement = [{"content": "Ship", "status": "in_progress"}]

    assert merge_todos(current, None) == current
    assert merge_todos(current, replacement) == replacement
    assert merge_todos(current, []) == []


def test_delegation_updates_merge_by_run_and_do_not_downgrade_terminal_results():
    completed = {
        "id": "call-1",
        "run_id": "run-a",
        "description": "Research reducers",
        "subagent_type": "research",
        "status": "completed",
        "result_brief": "Reducers documented",
        "created_at": "2026-10-04T10:00:00Z",
    }
    stale_update = {
        **completed,
        "status": "in_progress",
        "result_brief": "Still working",
    }
    reused_provider_id = {
        **completed,
        "run_id": "run-b",
        "status": "failed",
        "result_brief": "Provider failed",
    }

    merged = merge_delegations([completed], [stale_update, reused_provider_id])

    assert merged == [completed, reused_provider_id]


def test_skill_context_refreshes_by_path_and_keeps_only_safe_references():
    existing = [
        {
            "name": "research",
            "path": "skills/research/SKILL.md",
            "description": "Old description",
            "loaded_at": 2,
            "body": "full instructions must not be checkpointed",
        },
        {
            "name": "writing",
            "path": "skills/writing/SKILL.md",
            "description": "Write clearly",
            "loaded_at": 3,
        },
    ]
    refreshed = {
        "name": "research",
        "path": "skills/research/SKILL.md",
        "description": "  Current   research guidance  ",
        "loaded_at": 9,
        "body": "new full instructions",
    }

    merged = merge_skill_context(existing, [refreshed])

    assert merged == [
        {
            "name": "writing",
            "path": "skills/writing/SKILL.md",
            "description": "Write clearly",
            "loaded_at": 3,
        },
        {
            "name": "research",
            "path": "skills/research/SKILL.md",
            "description": "Current research guidance",
            "loaded_at": 9,
        },
    ]


def test_thread_state_reducers_merge_parallel_graph_updates():
    graph = StateGraph(ThreadState)
    graph.add_node(
        "researcher",
        lambda _state: {
            "artifacts": ["research.md"],
            "delegations": [{"id": "call-1", "run_id": "run-1", "status": "completed"}],
        },
    )
    graph.add_node(
        "writer",
        lambda _state: {
            "artifacts": ["answer.md"],
            "delegations": [{"id": "call-2", "run_id": "run-1", "status": "completed"}],
        },
    )
    graph.add_edge(START, "researcher")
    graph.add_edge(START, "writer")
    graph.add_edge("researcher", END)
    graph.add_edge("writer", END)

    result = graph.compile().invoke({"messages": []})

    assert result["artifacts"] == ["research.md", "answer.md"]
    assert [entry["id"] for entry in result["delegations"]] == ["call-1", "call-2"]
