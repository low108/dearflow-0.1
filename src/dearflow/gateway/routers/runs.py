"""Run endpoints. Paths and behaviour follow DeerFlow's `thread_runs` router.

- `POST /runs`         start in the background, return the run at once
- `POST /runs/stream`  start and stream events (SSE)
- `POST /runs/wait`    start and block until done, return the final state
- `GET  /runs/{id}/join`  watch an existing run (any number of watchers)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Iterable
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from dearflow.gateway.deps import Gateway, current_user, get_gateway, owned_thread
from dearflow.gateway.models import RunCreateRequest, RunResponse, ThreadId
from dearflow.gateway.run_manager import ActiveRun, IdempotencyConflictError
from dearflow.persistence.runs import RUNNING
from dearflow.service import ThreadBusyError
from dearflow.streaming import StreamEvent, to_jsonable

router = APIRouter(prefix="/api/threads/{thread_id}/runs", tags=["runs"])

_DONE = object()
_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


# --- starting runs ----------------------------------------------------------


def _idempotency_scope(user: str, thread_id: str, key: str) -> str:
    return "http-run:" + hashlib.sha256(f"{user}\0{thread_id}\0{key}".encode()).hexdigest()


def _request_hash(body: RunCreateRequest) -> str:
    logical_request = {
        "assistant_id": body.assistant_id,
        "input": body.input,
        "metadata": body.metadata or {},
        "context": body.overrides,
    }
    encoded = json.dumps(logical_request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


async def _start(
    gw: Gateway,
    body: RunCreateRequest,
    thread_id: str,
    user: str,
    idempotency_key: str | None,
) -> ActiveRun:
    owned_thread(gw, thread_id, user)  # 404 for someone else's thread; a new thread is created by the run
    scoped_key: str | None = None
    fingerprint: str | None = None
    if idempotency_key is not None:
        idempotency_key = idempotency_key.strip()
        if not idempotency_key:
            raise HTTPException(status_code=422, detail="Idempotency-Key must not be blank")
        scoped_key = _idempotency_scope(user, thread_id, idempotency_key)
        fingerprint = _request_hash(body)
        existing = gw.persistence.runs.get_by_idempotency_key(scoped_key)
        if existing is not None:
            if existing.request_hash != fingerprint:
                raise HTTPException(status_code=409, detail="Idempotency-Key already used with a different request")
            active = gw.runs.get(existing.run_id)
            if active is not None:
                return active
    try:
        service = await asyncio.to_thread(gw.service, body.overrides)
    except Exception as exc:  # unknown model name, missing API key...
        raise HTTPException(status_code=422, detail=f"Could not build the agent: {exc}") from exc
    try:
        return await asyncio.to_thread(
            gw.runs.start,
            service,
            body.message,
            thread_id=thread_id,
            user_id=user,
            modes=body.stream_mode,
            idempotency_key=scoped_key,
            request_hash=fingerprint,
            multitask_strategy=body.multitask_strategy,
        )
    except ThreadBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _response(gw: Gateway, run: ActiveRun) -> RunResponse:
    record = gw.persistence.runs.get(run.run_id)
    return RunResponse.from_record(record)  # type: ignore[arg-type]


@router.post("", response_model=RunResponse)
async def create_run(
    thread_id: ThreadId,
    body: RunCreateRequest,
    gw: Gateway = Depends(get_gateway),
    user: str = Depends(current_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=255),
):
    """Start a run in the background (returns immediately)."""
    return _response(gw, await _start(gw, body, thread_id, user, idempotency_key))


@router.post("/stream")
async def stream_run(
    thread_id: ThreadId,
    body: RunCreateRequest,
    gw: Gateway = Depends(get_gateway),
    user: str = Depends(current_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=255),
):
    """Start a run and stream its events as Server-Sent Events."""
    run = await _start(gw, body, thread_id, user, idempotency_key)
    return StreamingResponse(
        _sse(gw, run.subscribe(gw.config.gateway.heartbeat_seconds), on_close=_cancel_if(gw, run, body.on_disconnect)),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, "Content-Location": f"/api/threads/{thread_id}/runs/{run.run_id}"},
    )


@router.post("/wait")
async def wait_run(
    thread_id: ThreadId,
    body: RunCreateRequest,
    gw: Gateway = Depends(get_gateway),
    user: str = Depends(current_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=255),
):
    """Start a run, wait for it, return the final thread state (or the run's status if it failed)."""
    run = await _start(gw, body, thread_id, user, idempotency_key)
    await asyncio.to_thread(run.wait)
    record = gw.persistence.runs.get(run.run_id)
    if record.status != "success":
        return {"run_id": run.run_id, "status": record.status, "error": record.error}
    return to_jsonable(await asyncio.to_thread(gw.service().state, thread_id))


# --- reading runs -------------------------------------------------------------


def _visible_run(gw: Gateway, thread_id: str, run_id: str, user: str):
    thread = owned_thread(gw, thread_id, user)
    record = gw.persistence.runs.get(run_id)
    if thread is None or record is None or record.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return record


@router.get("", response_model=list[RunResponse])
async def list_runs(thread_id: ThreadId, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    if owned_thread(gw, thread_id, user) is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    return [RunResponse.from_record(r) for r in gw.persistence.runs.list(thread_id, limit=100)]


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(thread_id: ThreadId, run_id: str, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    return RunResponse.from_record(_visible_run(gw, thread_id, run_id, user))


@router.get("/{run_id}/events")
async def list_run_events(
    thread_id: ThreadId,
    run_id: str,
    event_types: str | None = Query(default=None, description="comma-separated filter"),
    limit: int = Query(default=500, ge=1, le=2000),
    after_seq: int = Query(default=0, ge=0),
    gw: Gateway = Depends(get_gateway),
    user: str = Depends(current_user),
):
    """The run log (debug / audit): updates, custom events, errors, start and end."""
    _visible_run(gw, thread_id, run_id, user)
    wanted = set(event_types.split(",")) if event_types else None
    events = gw.persistence.events.list(run_id, after_seq)
    return [
        {"run_id": run_id, "seq": seq, "event_type": name, "content": data}
        for seq, name, data in events
        if wanted is None or name in wanted
    ][:limit]


@router.post("/{run_id}/cancel")
async def cancel_run(
    thread_id: ThreadId,
    run_id: str,
    wait: bool = Query(default=False, description="block until the run has stopped"),
    action: Literal["interrupt"] = Query(default="interrupt", description="rollback is not supported"),
    gw: Gateway = Depends(get_gateway),
    user: str = Depends(current_user),
):
    """Stop a running run at its next event. 202 = asked, 204 = stopped (with wait=true)."""
    _visible_run(gw, thread_id, run_id, user)
    run = gw.runs.get(run_id)
    if run is None or not gw.runs.cancel(run_id):
        raise HTTPException(status_code=409, detail=f"Run {run_id} is not running on this server")
    if wait:
        await asyncio.to_thread(run.wait)
        return Response(status_code=204)
    return Response(status_code=202)


@router.get("/{run_id}/join")
async def join_run(thread_id: ThreadId, run_id: str, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    """Watch a run that is already going. Observers never cancel it by leaving."""
    record = _visible_run(gw, thread_id, run_id, user)
    active = gw.runs.get(run_id)
    if active is not None:
        source: Iterable[StreamEvent | None] = active.subscribe(gw.config.gateway.heartbeat_seconds)
    elif record.status != RUNNING:
        source = gw.service().events(run_id)  # finished long ago: replay the log (no tokens, see RunEventStore)
    else:
        raise HTTPException(status_code=409, detail=f"Run {run_id} is not active on this server and cannot be streamed")
    return StreamingResponse(
        _sse(gw, source),
        media_type="text/event-stream",
        headers={**_SSE_HEADERS, "Content-Location": f"/api/threads/{thread_id}/runs/{run_id}"},
    )


# --- SSE ----------------------------------------------------------------------


def _cancel_if(gw: Gateway, run: ActiveRun, policy: str):
    """`on_disconnect: cancel` (default): a client that leaves stops the run."""

    def close() -> None:
        if policy == "cancel" and not run.done:
            gw.runs.cancel(run.run_id)

    return close


async def _sse(gw: Gateway, events: Iterable[StreamEvent | None], on_close=None) -> AsyncIterator[str]:
    """Turn the blocking event iterator into SSE text frames without blocking the server.

    A None from the iterator is a heartbeat, sent as an SSE comment so proxies keep
    the connection open. When the client disconnects, this coroutine is cancelled and
    `finally` runs, which is where `on_disconnect` is applied.
    """
    iterator = iter(events)
    try:
        while True:
            item = await asyncio.to_thread(next, iterator, _DONE)
            if item is _DONE:
                return
            yield ": heartbeat\n\n" if item is None else item.sse()
    finally:
        if on_close is not None:
            on_close()
