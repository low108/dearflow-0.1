"""Stops the run when the model asks the user a question."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from dearflow.tools.clarification import ASK_CLARIFICATION

_ICONS = {
    "missing_info": "❓",
    "ambiguous_requirement": "🤔",
    "approach_choice": "🔀",
    "risk_confirmation": "⚠️",
    "suggestion": "💡",
}
_FORM_FIELD_TYPES = frozenset({"text", "textarea", "number", "select", "multi_select", "checkbox", "date"})
_OPTION_FIELD_TYPES = frozenset({"select", "multi_select"})
_RESERVED_FIELD_NAMES = frozenset({"__proto__", "constructor", "prototype", "toString", "valueOf"})
_MAX_FORM_FIELDS = 16
_MAX_FIELD_OPTIONS = 24
_MAX_FIELD_TEXT_CHARS = 200
_MAX_FORM_BYTES = 16_384


class ClarificationMiddleware(AgentMiddleware):
    """Turns an `ask_clarification` tool call into "show the question, end the run".

    Two hooks cooperate:

    1. `after_model` runs right after the model answers. If the model called
       `ask_clarification` together with other tools, the other calls are
       removed, so nothing with side effects runs before the user replies.
    2. `wrap_tool_call` intercepts the clarification call itself and returns
       the formatted question as the tool result. Because the tool is
       `return_direct`, the graph then ends instead of calling the model
       again. The user's next message continues the same thread as a new run.
    """

    # --- hook 1: drop sibling tool calls -----------------------------------

    @hook_config(can_jump_to=[])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._clarification_disabled(runtime):
            return None
        last = state["messages"][-1] if state["messages"] else None
        if not isinstance(last, AIMessage):
            return None
        clarifications = [c for c in last.tool_calls if c["name"] == ASK_CLARIFICATION]
        invalid_clarifications = [
            call
            for call in (last.invalid_tool_calls or [])
            if call.get("name") == ASK_CLARIFICATION
        ]
        if not clarifications and not invalid_clarifications:
            return None
        siblings = [call for call in last.tool_calls if call["name"] != ASK_CLARIFICATION]
        if not siblings:
            return None
        # Same message id => LangGraph replaces the old message with this one.
        return {
            "messages": [
                last.model_copy(
                    update={
                        "tool_calls": clarifications[:1],
                        "invalid_tool_calls": invalid_clarifications,
                    }
                )
            ]
        }

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    # --- hook 2: end the run with the question -----------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != ASK_CLARIFICATION:
            return handler(request)
        if self._clarification_disabled(request.runtime):
            return self._continue_without_user(request)
        return self._end_with_question(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != ASK_CLARIFICATION:
            return await handler(request)
        if self._clarification_disabled(request.runtime):
            return self._continue_without_user(request)
        return self._end_with_question(request)

    @staticmethod
    def _clarification_disabled(runtime: Any) -> bool:
        context = getattr(runtime, "context", None) or {}
        return bool(context.get("disable_clarification") or context.get("non_interactive"))

    @staticmethod
    def _continue_without_user(request: ToolCallRequest) -> Command:
        tool_call_id = str(request.tool_call.get("id") or "")
        message = ToolMessage(
            id=f"clarification-disabled:{tool_call_id}",
            content=(
                "Clarification is disabled because the human is not present to answer synchronously. "
                "For low-risk, reversible work, use the smallest reasonable assumption and state it. "
                "For risky or irreversible work without sufficient authorization, do not guess or act; "
                "return a concise BLOCKED result naming the missing decision."
            ),
            tool_call_id=tool_call_id,
            name=ASK_CLARIFICATION,
        )
        return Command(update={"messages": [message]}, goto="model")

    def _end_with_question(self, request: ToolCallRequest) -> ToolMessage:
        call = request.tool_call
        args = call.get("args") or {}
        fields = self._normalize_fields(args.get("fields"))
        content = self.format_question(args, fields=fields)
        request_id = f"clarification:{call['id']}"
        return ToolMessage(
            id=request_id,
            content=content,
            tool_call_id=call["id"],
            name=ASK_CLARIFICATION,
            artifact={
                "human_input": self._human_input_payload(
                    args,
                    tool_call_id=call["id"],
                    request_id=request_id,
                    fields=fields,
                )
            },
        )

    @classmethod
    def format_question(
        cls,
        args: dict[str, Any],
        *,
        fields: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render the tool arguments as text for the user."""
        # Invalid provider output can contain a non-string (even an unhashable
        # list) here. Rendering a fallback question must never crash the run.
        clarification_type = args.get("clarification_type", "")
        icon = _ICONS.get(clarification_type, "❓") if isinstance(clarification_type, str) else "❓"
        lines = []
        if args.get("context"):
            lines.append(f"{icon} {args['context']}")
            lines.append(str(args.get("question", "")))
        else:
            lines.append(f"{icon} {args.get('question', '')}")
        normalized_fields = cls._normalize_fields(args.get("fields")) if fields is None else fields
        if normalized_fields:
            lines.append("")
            for number, field in enumerate(normalized_fields, start=1):
                line = f"  {number}. {field['label']}"
                if field["required"]:
                    line += " (required)"
                if field.get("options"):
                    line += " — options: " + " / ".join(option["label"] for option in field["options"])
                    if field["type"] == "multi_select":
                        line += " (multiple allowed)"
                lines.append(line)
            lines.extend(["", "Please reply with a value for each field."])
        else:
            for number, option in enumerate(cls._normalize_options(args.get("options")), start=1):
                lines.append(f"  {number}. {option}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_options(raw: Any) -> list[str]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = [raw]
        if raw is None:
            return []
        if not isinstance(raw, list):
            raw = [raw]
        result: list[str] = []
        for option in raw:
            text = str(option).strip()
            if text and text not in result:
                result.append(text)
        return result

    @classmethod
    def _normalize_fields(cls, raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, list) or len(raw) > _MAX_FORM_FIELDS:
            return []

        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                return []
            name = item.get("name")
            if not isinstance(name, str):
                return []
            name = name.strip()
            if not name or name in seen or name in _RESERVED_FIELD_NAMES or len(name) > _MAX_FIELD_TEXT_CHARS:
                return []
            seen.add(name)

            raw_label = item.get("label")
            label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else name
            if len(label) > _MAX_FIELD_TEXT_CHARS:
                return []
            field_type = item.get("type")
            if not isinstance(field_type, str) or field_type not in _FORM_FIELD_TYPES:
                field_type = "text"
            options = cls._normalize_options(item.get("options")) if field_type in _OPTION_FIELD_TYPES else []
            if len(options) > _MAX_FIELD_OPTIONS or any(len(option) > _MAX_FIELD_TEXT_CHARS for option in options):
                return []
            if field_type in _OPTION_FIELD_TYPES and not options:
                field_type = "text"

            field: dict[str, Any] = {
                "name": name,
                "label": label,
                "type": field_type,
                "required": item.get("required") is True,
            }
            if field_type in _OPTION_FIELD_TYPES:
                field["options"] = [
                    {"id": f"{name}-option-{index}", "label": option, "value": option}
                    for index, option in enumerate(options, start=1)
                ]
            placeholder = item.get("placeholder")
            if isinstance(placeholder, str) and placeholder.strip():
                placeholder = placeholder.strip()
                if len(placeholder) > _MAX_FIELD_TEXT_CHARS:
                    return []
                field["placeholder"] = placeholder
            result.append(field)

        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > _MAX_FORM_BYTES:
            return []
        return result

    @classmethod
    def _human_input_payload(
        cls,
        args: dict[str, Any],
        *,
        tool_call_id: str,
        request_id: str,
        fields: list[dict[str, Any]],
    ) -> dict[str, Any]:
        options = cls._normalize_options(args.get("options"))
        mode = "form" if fields else "choice_with_other" if options else "free_text"
        payload: dict[str, Any] = {
            "version": 2 if fields else 1,
            "kind": "human_input_request",
            "source": ASK_CLARIFICATION,
            "request_id": request_id,
            "tool_call_id": tool_call_id,
            "clarification_type": str(args.get("clarification_type") or "missing_info"),
            "question": str(args.get("question") or ""),
            "input_mode": mode,
        }
        if args.get("context") is not None:
            payload["context"] = str(args["context"])
        if fields:
            payload["fields"] = fields
        elif options:
            payload["options"] = [
                {"id": f"option-{index}", "label": option, "value": option}
                for index, option in enumerate(options, start=1)
            ]
        return payload
