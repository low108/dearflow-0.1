"""Observe middleware without changing its capabilities or execution order.

Each hook gets a LangChain child span, so normal callbacks and LangSmith share
the same parent/thread metadata. Wrapping hooks capture the request at each
handler invocation and its response on return, before outer middleware edits it.
Nothing is written to conversation state or the durable conversation event log.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import wraps
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableLambda

from dearflow.config import ObservabilityConfig
from dearflow.observability.snapshots import Snapshotter
from dearflow.pii import Redactor

logger = logging.getLogger(__name__)
_NODE_HOOKS = ("before_agent", "before_model", "after_model", "after_agent")
_WRAP_HOOKS = ("wrap_model_call", "wrap_tool_call")
_MAX_CAPTURED_CALLS = 8


class _HookError(Exception):
    """Sanitized error for the snapshot span; callers receive the original error."""


class _Observation:
    def __init__(self, owner, hook, value):
        self.snapshot = owner.snapshot
        self.inputs = {"before": self.snapshot(value)}
        self.config = {
            "run_name": f"middleware.{owner.name}.{hook}",
            "metadata": {"dearflow_middleware": owner.name, "dearflow_hook": hook},
            "tags": ["dearflow:middleware-snapshot"],
        }
        self.calls: list[dict] = []
        self.call_count = 0
        self.started = False
        self.result = None
        self.error: BaseException | None = None

    def forward(self, request):
        self.call_count += 1
        call = {"request": self.snapshot(request)} if len(self.calls) < _MAX_CAPTURED_CALLS else None
        if call is not None:
            self.calls.append(call)
        return call

    def response(self, call, value):
        if call is not None:
            call["response"] = self.snapshot(value)

    def failed(self, call, exc):
        if call is not None:
            call["error_type"] = type(exc).__name__

    def outputs(self, result_key):
        return {
            result_key: self.snapshot(self.result),
            "calls": self.calls,
            "handler_calls": self.call_count,
            "calls_truncated": self.call_count > len(self.calls),
        }

    def unwrap(self):
        if self.error is not None:
            raise self.error.with_traceback(self.error.__traceback__)
        return self.result


def _sync_hook(original, hook, wrapping):
    @wraps(original)
    def observed(self, *args, **kwargs):
        value = args[0] if args else kwargs["request" if wrapping else "state"]
        runtime_or_handler = args[1] if len(args) > 1 else kwargs["handler" if wrapping else "runtime"]
        obs = _Observation(self, hook, value)

        def handler(request):
            call = obs.forward(request)
            try:
                result = runtime_or_handler(request)
            except BaseException as exc:
                obs.failed(call, exc)
                raise
            obs.response(call, result)
            return result

        def execute(_):
            obs.started = True
            try:
                obs.result = original(self.inner, value, handler if wrapping else runtime_or_handler)
            except BaseException as exc:
                obs.error = exc
                raise _HookError(type(exc).__name__) from None
            return obs.outputs("result" if wrapping else "state_update")

        try:
            RunnableLambda(execute).invoke(obs.inputs, obs.config)
        except Exception:
            # Observability must not retry an already-executed model/tool call.
            # Only a failure before execution may fall back to an untraced call.
            if not obs.started:
                logger.warning("Middleware tracing unavailable for %s.%s", self.name, hook)
                return original(self.inner, value, runtime_or_handler)
        return obs.unwrap()

    return observed


def _async_hook(original, hook, wrapping):
    @wraps(original)
    async def observed(self, *args, **kwargs):
        value = args[0] if args else kwargs["request" if wrapping else "state"]
        runtime_or_handler = args[1] if len(args) > 1 else kwargs["handler" if wrapping else "runtime"]
        obs = _Observation(self, hook, value)

        async def handler(request):
            call = obs.forward(request)
            try:
                result = await runtime_or_handler(request)
            except BaseException as exc:
                obs.failed(call, exc)
                raise
            obs.response(call, result)
            return result

        async def execute(_):
            obs.started = True
            try:
                obs.result = await original(self.inner, value, handler if wrapping else runtime_or_handler)
            except BaseException as exc:
                obs.error = exc
                raise _HookError(type(exc).__name__) from None
            return obs.outputs("result" if wrapping else "state_update")

        try:
            await RunnableLambda(execute).ainvoke(obs.inputs, obs.config)
        except Exception:
            if not obs.started:
                logger.warning("Middleware tracing unavailable for %s.%s", self.name, hook)
                return await original(self.inner, value, runtime_or_handler)
        return obs.unwrap()

    return observed


class _ObservedMiddleware(AgentMiddleware):
    def __init__(self, inner, snapshot):
        self.inner = inner
        self._snapshot = snapshot
        self.state_schema = inner.state_schema
        self.tools = getattr(inner, "tools", ())
        self.transformers = getattr(inner, "transformers", ())
        self.trace_policy = getattr(inner, "trace_policy", None)
        # Newer LangChain versions also create native hook spans. Keep their
        # payload empty: the child snapshot span owns the bounded, redacted copy.
        # Older supported versions don't have this native tracing policy.
        if hasattr(AgentMiddleware, "trace_policy"):
            from langgraph.types import TracePolicy

            self.trace_policy = TracePolicy(
                process_inputs=lambda _: {"details": "See middleware snapshot child span"},
                process_outputs=lambda _: {"details": "See middleware snapshot child span"},
            )

    @property
    def name(self):
        return self.inner.name

    def snapshot(self, value):
        try:
            return self._snapshot(value)
        except Exception as exc:
            # Never export the original value or the exception's message when
            # serialization/redaction fails; observation must not stop the run.
            return {"unavailable": True, "error_type": type(exc).__name__}


def instrument_middlewares(
    middlewares: Sequence[AgentMiddleware],
    config: ObservabilityConfig,
    *,
    redactor: Redactor | None = None,
) -> list[AgentMiddleware]:
    """Return middleware with opt-in snapshots, preserving only implemented hooks.

    A distinct proxy class is needed because create_agent discovers overrides on
    the class, including hook_config routing attributes (retained by wraps).
    No existing middleware instance/class is modified.
    """
    if not config.middleware_snapshots:
        return list(middlewares)
    snapshot = Snapshotter(capture_content=config.capture_content, max_chars=config.max_snapshot_chars, redactor=redactor)
    observed = []
    for middleware in middlewares:
        methods: dict[str, Any] = {}
        for hook in (*_NODE_HOOKS, *_WRAP_HOOKS):
            for prefix, make_hook in (("", _sync_hook), ("a", _async_hook)):
                name = prefix + hook
                original = getattr(type(middleware), name)
                if original is not getattr(AgentMiddleware, name):
                    methods[name] = make_hook(original, name, hook in _WRAP_HOOKS)
        proxy = type(f"Observed{type(middleware).__name__}", (_ObservedMiddleware,), methods)
        observed.append(proxy(middleware, snapshot))
    return observed
