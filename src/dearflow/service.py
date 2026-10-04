"""Runs the agent and keeps the records straight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from dearflow.agents import LeadAgent
from dearflow.persistence import Persistence
from dearflow.persistence.runs import ERROR, INTERRUPTED, SUCCESS
from dearflow.persistence.threads import IDLE
from dearflow.text import text_of

_PREVIEW_CHARS = 2000


class ThreadBusyError(RuntimeError):
    """A run is already in progress on this thread."""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    thread_id: str
    state: dict[str, Any]

    @property
    def text(self) -> str:
        """The agent's final answer (or the clarification question)."""
        content = self.state["messages"][-1].content
        return content if isinstance(content, str) else str(content)


class RunService:
    """One run = claim the thread, record it, invoke the agent, record the outcome.

    The service never holds conversation data itself: messages are saved by the
    checkpointer, and `threads` / `runs` only describe them. That is the same
    split DeerFlow uses.
    """

    def __init__(self, agent: LeadAgent, persistence: Persistence) -> None:
        self._agent = agent
        self._db = persistence

    def invoke(self, message: str, *, thread_id: str, user_id: str = "default") -> RunResult:
        threads, runs = self._db.threads, self._db.runs
        threads.ensure(thread_id, user_id)
        if not threads.claim(thread_id):
            raise ThreadBusyError(f"thread {thread_id!r} already has a run in progress")

        run_id = runs.start(thread_id, user_id, self._agent.model_name)
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "run_id": run_id}}
        try:
            state = self._agent.graph.invoke({"messages": [{"role": "user", "content": message}]}, config)
        except Exception as exc:
            runs.finish(run_id, ERROR, error=f"{type(exc).__name__}: {exc}", first_human_message=message[:_PREVIEW_CHARS])
            raise
        except BaseException:  # Ctrl-C and friends: the run did not finish, and it isn't an error either
            runs.finish(run_id, INTERRUPTED, first_human_message=message[:_PREVIEW_CHARS])
            raise
        else:
            runs.finish(
                run_id,
                SUCCESS,
                first_human_message=message[:_PREVIEW_CHARS],
                last_ai_message=self._last_answer(state),
                message_count=len(state["messages"]),
            )
            self._save_title(thread_id, state)
            return RunResult(run_id, thread_id, state)
        finally:
            threads.set_status(thread_id, IDLE)

    def history(self, thread_id: str) -> list:
        """All messages of a thread, straight from the checkpointer (hidden ones excluded)."""
        snapshot = self._agent.graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = (snapshot.values or {}).get("messages", [])
        return [m for m in messages if not m.additional_kwargs.get("hide_from_ui")]

    def _save_title(self, thread_id: str, state: dict[str, Any]) -> None:
        title = state.get("title")
        if title:
            self._db.threads.set_title(thread_id, title)

    @staticmethod
    def _last_answer(state: dict[str, Any]) -> str | None:
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and not message.tool_calls and text_of(message).strip():
                return text_of(message)[:_PREVIEW_CHARS]
        return None


__all__ = ["RunResult", "RunService", "ThreadBusyError"]
