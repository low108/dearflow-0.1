"""Shared pieces the routes depend on."""

from __future__ import annotations

import hmac
import re
import threading
from dataclasses import dataclass, field

from fastapi import HTTPException, Request

from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig, expand_env
from dearflow.gateway.models import THREAD_ID_PATTERN
from dearflow.gateway.run_manager import RunManager
from dearflow.persistence import Persistence, ThreadRecord
from dearflow.service import RunService

DEFAULT_USER = "default"
_USER_ID = re.compile(THREAD_ID_PATTERN)


@dataclass
class Gateway:
    """Everything the API needs, built once at startup (`app.state.gateway`)."""

    config: AppConfig
    persistence: Persistence
    factory: LeadAgentFactory
    runs: RunManager
    api_key: str | None
    _default: RunService | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def service(self, overrides: dict | None = None) -> RunService:
        """A service for one run. Per-run overrides (model, plan mode...) mean a fresh agent graph;
        without overrides the default one is built once and reused (reads, replays)."""
        if not overrides:
            with self._lock:
                if self._default is None:
                    self._default = self._build({})
                return self._default
        return self._build(overrides)

    def _build(self, overrides: dict) -> RunService:
        agent = self.factory.build({"configurable": dict(overrides)})
        return RunService(agent, self.persistence, track_usage=self.config.token_usage.enabled)


def get_gateway(request: Request) -> Gateway:
    return request.app.state.gateway


def current_user(request: Request) -> str:
    """Who is calling.

    With `gateway.api_key` set, callers must send `Authorization: Bearer <key>`.
    The user is then named by the `X-User-Id` header (default "default"), so the key
    holder is trusted to say who they act for, as a backend behind a login would.
    DeerFlow has full user accounts (JWT cookies, CSRF); this is the compact version.
    """
    gateway = get_gateway(request)
    if gateway.api_key is not None:
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), gateway.api_key.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
    user = request.headers.get("X-User-Id", DEFAULT_USER)
    if not _USER_ID.fullmatch(user):
        raise HTTPException(status_code=400, detail="Invalid X-User-Id: expected 1-64 letters, digits, '-' or '_'")
    return user


def owned_thread(gateway: Gateway, thread_id: str, user: str) -> ThreadRecord | None:
    """The thread if it exists and is yours; None if it doesn't exist.

    Someone else's thread is a 404, not a 403, so ids can't be probed.
    """
    record = gateway.persistence.threads.get(thread_id)
    if record is not None and record.user_id != user:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    return record


def resolve_api_key(config: AppConfig) -> str | None:
    key = config.gateway.api_key
    return expand_env(key) if key else None
