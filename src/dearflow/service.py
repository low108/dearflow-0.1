"""Runs the agent and keeps the records straight."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Overwrite

from dearflow.agents import LeadAgent
from dearflow.persistence import Persistence
from dearflow.persistence.runs import ERROR, INTERRUPTED, SUCCESS
from dearflow.persistence.threads import IDLE
from dearflow.streaming import END, ERROR as ERROR_EVENT, METADATA, StreamEvent, normalize_modes, to_jsonable
from dearflow.receipts import extract_receipts
from dearflow.text import text_of
from dearflow.usage import UsageCallback

_PREVIEW_CHARS = 2000
# Events worth keeping in the run log. Tokens (`messages`) and full snapshots
# (`values`) are too big and add nothing the checkpointer doesn't have.
_LOGGED_EVENTS = frozenset({METADATA, "updates", "custom", ERROR_EVENT, END})


class ThreadBusyError(RuntimeError):
    """A run is already in progress on this thread."""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    thread_id: str
    state: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)  # token counts, see `RunUsage.snapshot`

    @property
    def text(self) -> str:
        """The agent's final answer (or the clarification question)."""
        content = self.state["messages"][-1].content
        return content if isinstance(content, str) else str(content)


@dataclass
class _Run:
    """Handle for a run in progress (see `RunService._tracked`)."""

    run_id: str
    thread_id: str
    config: dict[str, Any]
    state: dict[str, Any] | None = None  # set by the caller when it already has the final state
    usage: dict[str, Any] = field(default_factory=dict)  # filled in when the run ends


class RunService:
    """One run = claim the thread, record it, run the agent, record the outcome.

    The service never holds conversation data itself: messages are saved by the
    checkpointer, and `threads` / `runs` / `run_events` only describe them. That
    is the same split DeerFlow uses.

    `invoke` returns when the run is done. `stream` yields `StreamEvent`s while
    it runs; both share the same bookkeeping.
    """

    def __init__(self, agent: LeadAgent, persistence: Persistence, *, track_usage: bool = True) -> None:
        self._agent = agent
        self._db = persistence
        self._track_usage = track_usage

    # --- public API ------------------------------------------------------------

    def invoke(
        self,
        message: str,
        *,
        thread_id: str,
        user_id: str = "default",
        non_interactive: bool = False,
    ) -> RunResult:
        with self._tracked(message, thread_id, user_id) as run:
            run.state = self._agent.graph.invoke(
                self._input(message),
                run.config,
                context={"non_interactive": True} if non_interactive else None,
            )
        return RunResult(run.run_id, thread_id, run.state, run.usage)

    def stream(
        self,
        message: str,
        *,
        thread_id: str,
        user_id: str = "default",
        modes: list[str] | str | None = None,
        non_interactive: bool = False,
        idempotency_key: str | None = None,
        request_hash: str | None = None,
        multitask_strategy: str = "reject",
    ) -> Iterator[StreamEvent]:
        """Run the agent and yield events as they happen.

        Order: `metadata`, then the chunks of the requested `modes`, then `end`
        (or `error` followed by `end`). Stop iterating early and the run is
        recorded as interrupted. Unsupported modes raise before anything starts.
        """
        requested = normalize_modes(modes)
        # `updates` is always collected for the run log, even if the caller didn't ask for it.
        graph_modes = list(dict.fromkeys([*requested, "updates"]))
        seq = 0
        run_id: str | None = None

        def make(name: str, data: Any) -> StreamEvent:
            nonlocal seq
            seq += 1
            item = StreamEvent(seq, name, to_jsonable(data))
            if name in _LOGGED_EVENTS:
                self._db.events.append(run_id, seq, name, item.data)
            return item

        failure: str | None = None
        handle: _Run | None = None
        try:
            with self._tracked(
                message,
                thread_id,
                user_id,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                multitask_strategy=multitask_strategy,
            ) as run:
                handle = run
                run_id = run.run_id
                yield make(METADATA, {"run_id": run_id, "thread_id": thread_id})
                for mode, chunk in self._agent.graph.stream(
                    self._input(message),
                    run.config,
                    context={"non_interactive": True} if non_interactive else None,
                    stream_mode=graph_modes,
                ):
                    data = self._shape(mode, chunk)
                    if data is None:
                        continue
                    if mode in requested:
                        yield make(mode, data)
                    elif mode == "updates":
                        make(mode, data)  # log only
        except ThreadBusyError:
            raise  # no run exists, so there is nothing to report in a stream
        except Exception as exc:  # `_tracked` already recorded the failure
            failure = f"{type(exc).__name__}: {exc}"
        # A consumer that stops early raises GeneratorExit inside the `with`; it is a
        # BaseException, so it skips everything below. The run is recorded "interrupted".

        # The run is fully recorded by now, so a client that sees `end` can trust the records.
        if failure is not None:
            yield make(ERROR_EVENT, {"message": failure})
        yield make(END, {"status": SUCCESS if failure is None else ERROR, "usage": handle.usage if handle else {}})

    def events(self, run_id: str, after_seq: int = 0) -> list[StreamEvent]:
        """Replay a finished (or running) run's log."""
        return [StreamEvent(seq, name, data) for seq, name, data in self._db.events.list(run_id, after_seq)]

    def run_by_idempotency_key(self, idempotency_key: str):
        """Return the durable run claimed by a scoped HTTP idempotency key."""
        return self._db.runs.get_by_idempotency_key(idempotency_key)

    def state(self, thread_id: str) -> dict[str, Any]:
        """The thread's current state values (empty dict for a thread with no checkpoint)."""
        return dict(self._agent.graph.get_state({"configurable": {"thread_id": thread_id}}).values or {})

    def restore_state(self, thread_id: str, snapshot: dict[str, Any]) -> None:
        """Replace a thread checkpoint with a complete pre-run snapshot.

        Reducer-backed channels need `Overwrite`; ordinary updates would merge
        the interrupted run's messages, todos, and ledgers back into the state.
        """
        if not snapshot:
            self._db.checkpointer.delete_thread(thread_id)
            return

        defaults: dict[str, Any] = {
            "messages": [],
            "todos": None,
            "artifacts": [],
            "goal": None,
            "delegations": [],
            "skill_context": [],
            "title": None,
            "summary_text": None,
            "uploaded_files": [],
        }
        reducer_channels = {"messages", "todos", "artifacts", "goal", "delegations", "skill_context"}
        restored = {}
        for name, default in defaults.items():
            value = deepcopy(snapshot.get(name, default))
            restored[name] = Overwrite(value) if name in reducer_channels else value
        for name, value in snapshot.items():
            if name not in restored:
                restored[name] = deepcopy(value)
        self._agent.graph.update_state({"configurable": {"thread_id": thread_id}}, restored)

    def receipts(self, thread_id: str) -> list[dict[str, Any]]:
        """Every tool call recorded in a thread, as receipts r1..rN.

        Receipts of `task` calls also carry `citation_verdict`: the check of that subagent's report.
        Numbering follows the messages as they are now (see `dearflow.receipts` for the caveat).
        """
        messages = self.state(thread_id).get("messages", [])
        verdicts = {
            m.tool_call_id: m.additional_kwargs["receipt_verdict"]
            for m in messages
            if isinstance(m, ToolMessage) and m.additional_kwargs.get("receipt_verdict")
        }
        return [
            {**receipt, **({"citation_verdict": verdicts[receipt["tool_call_id"]]} if receipt["tool_call_id"] in verdicts else {})}
            for receipt in extract_receipts(messages)
        ]

    def history(self, thread_id: str) -> list:
        """All messages of a thread, straight from the checkpointer (hidden ones excluded)."""
        snapshot = self._agent.graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = (snapshot.values or {}).get("messages", [])
        return [m for m in messages if not m.additional_kwargs.get("hide_from_ui")]

    # --- bookkeeping -----------------------------------------------------------

    @staticmethod
    def _shape(mode: str, chunk: Any) -> Any | None:
        """Tidy a raw chunk for clients; None means "not worth sending".

        - `messages`: LangGraph gives `(message, metadata)`; clients get
          `{"message", "node"}`. Hidden middleware messages (date, memory) are dropped.
        - `updates`: steps where nothing changed (`{"Middleware.hook": None}`) are dropped.
        """
        if mode == "messages":
            message, metadata = chunk
            if message.additional_kwargs.get("hide_from_ui"):
                return None
            return {"message": message, "node": (metadata or {}).get("langgraph_node")}
        if mode == "updates" and all(update is None for update in chunk.values()):
            return None
        return chunk

    @staticmethod
    def _input(message: str) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": message}]}

    @contextmanager
    def _tracked(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        *,
        idempotency_key: str | None = None,
        request_hash: str | None = None,
        multitask_strategy: str = "reject",
    ) -> Iterator[_Run]:
        """Claim the thread and record the run around the `with` body.

        The body sets `run.state` if it has the final state; otherwise it is read
        back from the checkpointer. An exception marks the run `error`; Ctrl-C or
        an abandoned stream marks it `interrupted`. The thread is always freed.
        """
        threads, runs = self._db.threads, self._db.runs
        threads.ensure(thread_id, user_id)
        if not threads.claim(thread_id):
            raise ThreadBusyError(f"thread {thread_id!r} already has a run in progress")

        run_id = runs.start(
            thread_id,
            user_id,
            self._agent.model_name,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            multitask_strategy=multitask_strategy,
        )
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id, "user_id": user_id, "run_id": run_id},
            "run_id": UUID(run_id),
            "run_name": "DearFlow",
            "metadata": {"thread_id": thread_id, "dearflow_run_id": run_id},
        }
        counter = self._agent.usage.open(run_id)
        if self._track_usage:
            config["callbacks"] = [UsageCallback(counter)]  # sees the lead agent, subagents and middleware
        run = _Run(run_id, thread_id, config)
        preview = message[:_PREVIEW_CHARS]
        try:
            yield run
        except Exception as exc:
            run.usage = counter.snapshot()
            runs.finish(run_id, ERROR, error=f"{type(exc).__name__}: {exc}", first_human_message=preview, usage=run.usage)
            raise
        except BaseException:  # Ctrl-C, abandoned stream: not finished, and not an error either
            run.usage = counter.snapshot()
            runs.finish(run_id, INTERRUPTED, first_human_message=preview, usage=run.usage)
            raise
        else:
            run.usage = counter.snapshot()
            state = run.state or self._agent.graph.get_state(config).values
            runs.finish(
                run_id,
                SUCCESS,
                first_human_message=preview,
                last_ai_message=self._last_answer(state),
                message_count=len(state["messages"]),
                usage=run.usage,
            )
            if state.get("title"):
                threads.set_title(thread_id, state["title"])
        finally:
            self._agent.usage.close(run_id)
            threads.set_status(thread_id, IDLE)

    @staticmethod
    def _last_answer(state: dict[str, Any]) -> str | None:
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and not message.tool_calls and text_of(message).strip():
                return text_of(message)[:_PREVIEW_CHARS]
        return None


__all__ = ["RunResult", "RunService", "ThreadBusyError"]
