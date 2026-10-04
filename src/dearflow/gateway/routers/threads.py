"""Thread endpoints: create, search, read, delete, state, messages."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from dearflow.gateway.deps import Gateway, current_user, get_gateway, owned_thread
from dearflow.gateway.models import ThreadCreateRequest, ThreadId, ThreadResponse, ThreadSearchRequest
from dearflow.streaming import to_jsonable

router = APIRouter(prefix="/api/threads", tags=["threads"])


@router.post("", response_model=ThreadResponse)
async def create_thread(body: ThreadCreateRequest, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    thread_id = body.thread_id or str(uuid.uuid4())
    existing = owned_thread(gw, thread_id, user)
    record = existing or gw.persistence.threads.ensure(thread_id, user, body.metadata)
    return ThreadResponse.from_record(record)


@router.post("/search", response_model=list[ThreadResponse])
async def search_threads(body: ThreadSearchRequest, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    records = gw.persistence.threads.list(user, limit=1000)
    matching = [
        r
        for r in records
        if (body.status is None or r.status == body.status)
        and all(r.metadata.get(k) == v for k, v in body.metadata.items())
    ]
    return [ThreadResponse.from_record(r) for r in matching[body.offset : body.offset + body.limit]]


@router.get("/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: ThreadId, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    record = _require_thread(gw, thread_id, user)
    values = await asyncio.to_thread(gw.service().state, thread_id)
    return ThreadResponse.from_record(record, to_jsonable(values))


@router.delete("/{thread_id}")
async def delete_thread(thread_id: ThreadId, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    record = _require_thread(gw, thread_id, user)
    if record.status == "busy":
        raise HTTPException(status_code=409, detail=f"Thread {thread_id} has a run in progress")
    gw.persistence.threads.delete(thread_id)
    await asyncio.to_thread(gw.persistence.checkpointer.delete_thread, thread_id)
    return {"success": True, "message": f"Deleted thread {thread_id}"}


@router.get("/{thread_id}/state")
async def get_thread_state(thread_id: ThreadId, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    _require_thread(gw, thread_id, user)
    values = await asyncio.to_thread(gw.service().state, thread_id)
    return {"thread_id": thread_id, "values": to_jsonable(values)}


@router.get("/{thread_id}/messages")
async def get_thread_messages(thread_id: ThreadId, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    """The conversation as the user saw it (hidden middleware messages left out)."""
    _require_thread(gw, thread_id, user)
    messages = await asyncio.to_thread(gw.service().history, thread_id)
    return to_jsonable(messages)


def _require_thread(gw: Gateway, thread_id: str, user: str):
    record = owned_thread(gw, thread_id, user)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    return record


@router.get("/{thread_id}/token-usage")
async def thread_token_usage(thread_id: ThreadId, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    """Token totals for the whole thread, split by caller (lead / subagent / middleware) and model."""
    _require_thread(gw, thread_id, user)
    return {"thread_id": thread_id, **gw.persistence.runs.aggregate_tokens(thread_id)}


@router.get("/{thread_id}/receipts")
async def thread_receipts(thread_id: ThreadId, gw: Gateway = Depends(get_gateway), user: str = Depends(current_user)):
    """The audit trail: one receipt per tool call (name, status, argument and output hashes), oldest first."""
    _require_thread(gw, thread_id, user)
    receipts = await asyncio.to_thread(gw.service().receipts, thread_id)
    return {"thread_id": thread_id, "receipts": receipts}
