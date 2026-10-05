"""Repair incomplete tool-call history before it reaches the model."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

_UNKNOWN_TOOL = "unknown_tool"
_MAX_ERROR_CHARS = 500


class DanglingToolCallMiddleware(AgentMiddleware):
    """Give strict providers a causally valid, model-bound transcript.

    A cancelled run can leave an ``AIMessage`` with a tool call but no matching
    ``ToolMessage``. Summarization can do the reverse and leave an orphan tool
    result. Both shapes are rejected by OpenAI-compatible providers.

    This middleware repairs only ``ModelRequest.messages``. It never writes its
    placeholders back to LangGraph state, so the checkpoint still records what
    actually happened.
    """

    @staticmethod
    def _calls(message: AIMessage) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for call in message.tool_calls or []:
            if isinstance(call, dict):
                calls.append(call)
        for call in message.invalid_tool_calls or []:
            if isinstance(call, dict):
                calls.append({**call, "invalid": True})
        return calls

    @staticmethod
    def _safe_name(call: dict[str, Any]) -> str:
        name = call.get("name")
        return name.strip() if isinstance(name, str) and name.strip() else _UNKNOWN_TOOL

    @classmethod
    def _placeholder(cls, call: dict[str, Any]) -> ToolMessage:
        if call.get("invalid"):
            detail = call.get("error")
            suffix = f": {detail[:_MAX_ERROR_CHARS]}" if isinstance(detail, str) and detail else ""
            content = f"[Tool call could not be executed because its arguments were invalid{suffix}.]"
        elif cls._safe_name(call) == _UNKNOWN_TOOL:
            content = "[Tool call could not be executed because its name was missing.]"
        else:
            content = "[Tool call was interrupted and did not return a result.]"
        return ToolMessage(
            content=content,
            tool_call_id=str(call.get("id") or "missing-tool-call-id"),
            name=cls._safe_name(call),
            status="error",
        )

    @classmethod
    def _with_safe_ids(cls, message: AIMessage, message_index: int) -> AIMessage:
        update: dict[str, Any] = {}
        for field, calls in (
            ("tool_calls", message.tool_calls or []),
            ("invalid_tool_calls", message.invalid_tool_calls or []),
        ):
            rewritten = []
            changed = False
            for position, call in enumerate(calls):
                if not isinstance(call, dict) or (
                    isinstance(call.get("id"), str) and call["id"].strip()
                ):
                    rewritten.append(call)
                    continue
                rewritten.append(
                    {
                        **call,
                        "id": f"dearflow-synthetic-{message_index}-{field}-{position}",
                    }
                )
                changed = True
            if changed:
                update[field] = rewritten
        return message.model_copy(update=update) if update else message

    @classmethod
    def _sanitize_ai_message(cls, message: AIMessage) -> AIMessage:
        """Make malformed calls serializable by strict OpenAI-compatible APIs."""
        changed = False
        tool_calls = []
        for call in message.tool_calls or []:
            if not isinstance(call, dict):
                tool_calls.append(call)
                continue
            safe = dict(call)
            name = cls._safe_name(safe)
            if safe.get("name") != name:
                safe["name"] = name
                changed = True
            tool_calls.append(safe)

        invalid_calls = []
        for call in message.invalid_tool_calls or []:
            if not isinstance(call, dict):
                invalid_calls.append(call)
                continue
            safe = dict(call)
            name = cls._safe_name(safe)
            if safe.get("name") != name:
                safe["name"] = name
                changed = True
            args = safe.get("args")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                parsed = None
            normalized = args if isinstance(parsed, dict) else "{}"
            if safe.get("args") != normalized:
                safe["args"] = normalized
                changed = True
            invalid_calls.append(safe)

        if not changed:
            return message
        return message.model_copy(update={"tool_calls": tool_calls, "invalid_tool_calls": invalid_calls})

    @classmethod
    def repair(cls, messages: list) -> list | None:
        """Return request-safe messages, or ``None`` when no repair is needed."""
        normalized = [
            cls._with_safe_ids(message, index) if isinstance(message, AIMessage) else message
            for index, message in enumerate(messages)
        ]
        results: dict[str, deque[ToolMessage]] = defaultdict(deque)
        declared_ids: set[str] = set()
        for message in normalized:
            if isinstance(message, ToolMessage):
                results[str(message.tool_call_id)].append(message)
            elif isinstance(message, AIMessage):
                declared_ids.update(
                    str(call["id"])
                    for call in cls._calls(message)
                    if call.get("id") not in (None, "")
                )

        repaired: list = []
        changed = normalized != messages
        for message in normalized:
            if isinstance(message, ToolMessage):
                # Valid results are re-emitted immediately after their call.
                # Orphans are omitted from this model request.
                changed = True
                continue

            safe_message = cls._sanitize_ai_message(message) if isinstance(message, AIMessage) else message
            repaired.append(safe_message)
            changed = changed or safe_message is not message
            if not isinstance(message, AIMessage):
                continue

            for call in cls._calls(message):
                call_id = str(call.get("id") or "missing-tool-call-id")
                existing = results[call_id].popleft() if results[call_id] else None
                repaired.append(existing or cls._placeholder(call))
                changed = changed or existing is None

        # A ToolMessage whose call disappeared is deliberately absent. The set
        # is retained here to make that contract explicit and easy to inspect.
        if any(tool_id not in declared_ids for tool_id in results):
            changed = True
        return repaired if changed and repaired != messages else None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        repaired = self.repair(list(request.messages))
        return handler(request.override(messages=repaired) if repaired is not None else request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        repaired = self.repair(list(request.messages))
        return await handler(request.override(messages=repaired) if repaired is not None else request)
