"""Runs agents in the background and lets clients watch them.

Plays the roles of DeerFlow's `RunManager` (who is running what, cancel) and
`StreamBridge` (producer -> consumers). One worker thread drives
`RunService.stream`; every event goes into an in-memory buffer; any number of
clients can read that buffer from the start, live, or after the run ended.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from copy import deepcopy

from dearflow.service import RunService
from dearflow.streaming import END, StreamEvent

START_TIMEOUT_SECONDS = 30


class RunStartError(RuntimeError):
    """The worker could not start the run (and did not say why)."""


class IdempotencyConflictError(RuntimeError):
    """An idempotency key was already claimed by a different logical request."""


class ActiveRun:
    """A run this process is driving (or finished recently)."""

    def __init__(
        self,
        run_id: str,
        thread_id: str,
        user_id: str,
        *,
        service: RunService | None = None,
        rollback_state: dict | None = None,
    ) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.user_id = user_id
        self.events: list[StreamEvent] = []
        self.done = False
        self.status = "running"
        self.cancel_requested = threading.Event()
        self._cond = threading.Condition()
        self.service = service
        self.rollback_state = rollback_state or {}

    def publish(self, event: StreamEvent) -> None:
        with self._cond:
            self.events.append(event)
            if event.event == END:
                self.status = event.data["status"]
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.done = True
            self._cond.notify_all()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run is over. True if it finished within `timeout`."""
        with self._cond:
            return self._cond.wait_for(lambda: self.done, timeout)

    def subscribe(self, heartbeat: float = 15.0) -> Iterator[StreamEvent | None]:
        """Yield every event from the start, then new ones as they arrive.

        Yields None as a heartbeat when nothing happened for `heartbeat` seconds,
        and stops after the last event of a finished run. Late readers get the
        full history, so joining a run never misses its beginning.
        """
        index = 0
        while True:
            with self._cond:
                self._cond.wait_for(lambda: index < len(self.events) or self.done, heartbeat)
                batch = self.events[index:]
                finished = self.done and index + len(batch) >= len(self.events)
            if not batch and not finished:
                yield None
                continue
            yield from batch
            index += len(batch)
            if finished:
                return


class RunManager:
    def __init__(self, *, retain_seconds: float = 120) -> None:
        self._runs: dict[str, ActiveRun] = {}
        self._lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._retain = retain_seconds

    def get(self, run_id: str) -> ActiveRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def start(
        self,
        service: RunService,
        message: str,
        *,
        thread_id: str,
        user_id: str,
        modes: list[str] | None,
        idempotency_key: str | None = None,
        request_hash: str | None = None,
        multitask_strategy: str = "reject",
    ) -> ActiveRun:
        """Serialize admission so concurrent retries cannot create duplicate runs."""
        with self._admission_lock:
            if idempotency_key is not None:
                existing = service.run_by_idempotency_key(idempotency_key)
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise IdempotencyConflictError("Idempotency-Key already used with a different request")
                    return self._remember(service, existing.run_id, existing.thread_id, existing.user_id, existing.status)
            return self._start_admitted(
                service,
                message,
                thread_id=thread_id,
                user_id=user_id,
                modes=modes,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                multitask_strategy=multitask_strategy,
            )

    def _start_admitted(
        self,
        service: RunService,
        message: str,
        *,
        thread_id: str,
        user_id: str,
        modes: list[str] | None,
        idempotency_key: str | None,
        request_hash: str | None,
        multitask_strategy: str,
    ) -> ActiveRun:
        """Start a run in the background and return once it exists.

        Claiming the thread happens inside the worker, so "thread busy" and bad
        modes surface here as exceptions, before the caller answers its client.
        """
        active = self._active_for_thread(thread_id)
        if active is not None and not active.done and multitask_strategy == "rollback":
            active.cancel_requested.set()
            if not active.wait(START_TIMEOUT_SECONDS):
                raise RunStartError("active run did not stop in time for rollback")
            if active.service is None:
                raise RunStartError("active run has no rollback service")
            active.service.restore_state(thread_id, active.rollback_state)

        rollback_state = deepcopy(service.state(thread_id))
        ready = threading.Event()
        outcome: dict = {}

        def work() -> None:
            stream = service.stream(
                message,
                thread_id=thread_id,
                user_id=user_id,
                modes=modes,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                multitask_strategy=multitask_strategy,
            )
            run: ActiveRun | None = None
            try:
                first = next(stream)  # `metadata`: the run now exists in the database
                run = ActiveRun(
                    first.data["run_id"],
                    thread_id,
                    user_id,
                    service=service,
                    rollback_state=rollback_state,
                )
                run.publish(first)
                with self._lock:
                    self._runs[run.run_id] = run
                outcome["run"] = run
                ready.set()
                for event in stream:
                    run.publish(event)
                    if run.cancel_requested.is_set():
                        stream.close()  # records the run as interrupted
                        run.publish(StreamEvent(event.id + 1, END, {"status": "interrupted"}))
                        break
            except BaseException as exc:  # before `ready`: report to the caller
                outcome["error"] = exc
            finally:
                ready.set()
                if run is not None:
                    run.finish()
                    self._forget_later(run.run_id)

        threading.Thread(target=work, name="dearflow-run", daemon=True).start()
        if not ready.wait(START_TIMEOUT_SECONDS):
            raise RunStartError("run did not start in time")
        if "error" in outcome and "run" not in outcome:
            raise outcome["error"]
        return outcome["run"]

    def _remember(self, service: RunService, run_id: str, thread_id: str, user_id: str, status: str) -> ActiveRun:
        active = self.get(run_id)
        if active is not None:
            return active
        run = ActiveRun(run_id, thread_id, user_id)
        for event in service.events(run_id):
            run.publish(event)
        run.status = status
        run.finish()
        return run

    def _active_for_thread(self, thread_id: str) -> ActiveRun | None:
        with self._lock:
            return next(
                (run for run in self._runs.values() if run.thread_id == thread_id and not run.done),
                None,
            )

    def cancel(self, run_id: str) -> bool:
        """Ask a run to stop. It stops at its next event (a model call in flight finishes first)."""
        run = self.get(run_id)
        if run is None or run.done:
            return False
        run.cancel_requested.set()
        return True

    def _forget_later(self, run_id: str) -> None:
        def forget() -> None:
            with self._lock:
                self._runs.pop(run_id, None)

        if self._retain <= 0:
            forget()
        else:
            timer = threading.Timer(self._retain, forget)
            timer.daemon = True
            timer.start()


__all__ = ["ActiveRun", "IdempotencyConflictError", "RunManager", "RunStartError"]
