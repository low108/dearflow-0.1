"""Thread state and reducers for concurrent LangGraph updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, NotRequired, TypedDict, cast

from langchain.agents import AgentState


class GoalState(TypedDict):
    """A durable objective carried across graph steps and turns."""

    objective: str
    status: str
    created_at: NotRequired[str]
    updated_at: NotRequired[str]
    continuation_count: NotRequired[int]
    max_continuations: NotRequired[int]
    last_evaluation: NotRequired[dict[str, Any]]


class DelegationEntry(TypedDict):
    """One durable subagent execution record."""

    id: str
    status: str
    run_id: NotRequired[str]
    description: NotRequired[str]
    subagent_type: NotRequired[str]
    result_brief: NotRequired[str]
    created_at: NotRequired[str]


class SkillContextEntry(TypedDict):
    """A lightweight reference to a skill loaded during the conversation."""

    name: str
    path: str
    description: str
    loaded_at: int


TERMINAL_DELEGATION_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "timed_out", "polling_timed_out", "max_turns"}
)
_DELEGATION_LEDGER_MAX_ENTRIES = 50
_SKILL_CONTEXT_MAX_ENTRIES = 8
_SKILL_DESCRIPTION_MAX_CHARS = 500


def merge_artifacts(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Merge artifact references, preserving first-seen order."""
    if existing is None:
        return new or []
    if new is None:
        return existing
    return list(dict.fromkeys([*existing, *new]))


def merge_goal(existing: GoalState | None, new: GoalState | None) -> GoalState | None:
    """Preserve the active goal unless a node explicitly replaces it."""
    return existing if new is None else new


def merge_todos(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Preserve todos when untouched; an explicit update, including [], wins."""
    return existing if new is None else new


def merge_delegations(
    existing: list[DelegationEntry] | None,
    new: list[DelegationEntry] | None,
) -> list[DelegationEntry]:
    """Merge the bounded delegation ledger by run and provider call ID."""
    if not new:
        return existing or []

    by_key: dict[tuple[str | None, str], DelegationEntry] = {}
    order: list[tuple[str | None, str]] = []
    for incoming in [*(existing or []), *new]:
        entry = cast(DelegationEntry, dict(incoming))
        key = (entry.get("run_id"), entry["id"])
        if key[0] is None:
            key = next((prior for prior in reversed(order) if prior[1] == key[1]), key)

        previous = by_key.get(key)
        if (
            previous is not None
            and previous["status"] in TERMINAL_DELEGATION_STATUSES
            and entry["status"] not in TERMINAL_DELEGATION_STATUSES
        ):
            continue
        if key not in by_key:
            order.append(key)
        elif previous is not None:
            if previous.get("created_at"):
                entry["created_at"] = previous["created_at"]
            if previous.get("run_id") and not entry.get("run_id"):
                entry["run_id"] = previous["run_id"]
        by_key[key] = entry

    merged = [by_key[key] for key in order]
    return merged[-_DELEGATION_LEDGER_MAX_ENTRIES:]


def _normalize_skill_context_entry(entry: Mapping[str, object]) -> SkillContextEntry:
    description = entry.get("description")
    loaded_at = entry.get("loaded_at")
    return {
        "name": str(entry.get("name") or ""),
        "path": str(entry["path"]),
        "description": (
            " ".join(description.split())[:_SKILL_DESCRIPTION_MAX_CHARS]
            if isinstance(description, str)
            else ""
        ),
        "loaded_at": loaded_at if isinstance(loaded_at, int) else 0,
    }


def merge_skill_context(
    existing: list[SkillContextEntry] | None,
    new: list[SkillContextEntry] | None,
) -> list[SkillContextEntry]:
    """Merge skill references by path, keeping the most recently loaded eight."""
    normalized_existing = [_normalize_skill_context_entry(entry) for entry in existing or []]
    if not new:
        return normalized_existing

    by_path = {entry["path"]: entry for entry in normalized_existing}
    order = list(by_path)
    for entry in (_normalize_skill_context_entry(item) for item in new):
        path = entry["path"]
        if path in by_path:
            order.remove(path)
        order.append(path)
        by_path[path] = entry

    return [by_path[path] for path in order[-_SKILL_CONTEXT_MAX_ENTRIES:]]


class ThreadState(AgentState):
    """Checkpointed conversation state shared by graph nodes and middleware."""

    title: NotRequired[str]
    todos: Annotated[list[dict[str, Any]] | None, merge_todos]
    artifacts: Annotated[list[str], merge_artifacts]
    goal: Annotated[GoalState | None, merge_goal]
    summary_text: NotRequired[str]
    delegations: Annotated[list[DelegationEntry], merge_delegations]
    skill_context: Annotated[list[SkillContextEntry], merge_skill_context]
    uploaded_files: NotRequired[list[dict[str, Any]]]
