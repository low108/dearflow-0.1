"""Small, immutable, redacted snapshots for middleware trace boundaries."""

from __future__ import annotations

import json
import math
import secrets
from collections.abc import Mapping
from typing import Any

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from dearflow.pii import DETECTORS, Redactor
from dearflow.receipts import LEDGER_KEY, RECEIPT_KEY

_SECRET_PARTS = (
    "apikey",
    "password",
    "passwd",
    "secret",
    "credential",
    "authorization",
    "cookie",
    "privatekey",
)
_BINARY_KEYS = {
    "image",
    "imageurl",
    "inputimage",
    "audio",
    "inputaudio",
    "video",
    "filedata",
    "base64",
    "b64json",
}
_RECEIPT_TEXT_FIELDS = (
    "id",
    "tool_call_id",
    "tool_name",
    "status",
    "args_sha256",
    "output_sha256",
    "created_at",
)
_BINARY_TYPES = {
    "image",
    "image_url",
    "input_image",
    "audio",
    "input_audio",
    "video",
    "file",
    "base64",
}


class Snapshotter:
    """Capture supported values without inspecting model/runtime objects or reprs.

    Content capture is opt-in. Every snapshot uses the same redactor so placeholders
    remain comparable across boundaries. The default secret lives only in memory.
    ``max_chars`` bounds the entire JSON envelope, including escaped Unicode.
    """

    def __init__(
        self,
        *,
        capture_content: bool = False,
        max_chars: int = 20000,
        redactor: Redactor | None = None,
    ) -> None:
        if max_chars < 128:
            raise ValueError("max_chars must be at least 128")
        self.capture_content = capture_content
        self.max_chars = max_chars
        self.redactor = redactor or Redactor(DETECTORS, secrets.token_hex(32))

    def __call__(self, value: Any) -> dict:
        truncated = False
        redaction_applied = False
        nodes = 0

        def redact(text: str) -> str:
            nonlocal redaction_applied
            safe = self.redactor.redact(text)
            redaction_applied = redaction_applied or safe != text
            return safe

        def receipt_fields(receipt: Any) -> dict:
            if not isinstance(receipt, Mapping):
                return {"omitted": "invalid receipt"}
            fields = {key: receipt[key] for key in _RECEIPT_TEXT_FIELDS if isinstance(receipt.get(key), str)}
            size = receipt.get("output_bytes")
            if isinstance(size, int) and not isinstance(size, bool):
                fields["output_bytes"] = size
            return fields

        def walk(item: Any, depth: int = 0) -> Any:
            nonlocal truncated, redaction_applied, nodes
            nodes += 1
            if depth > 8 or nodes > 1000:
                truncated = True
                return {"omitted": "depth or item limit"}
            if isinstance(item, BaseMessage):
                result = {
                    "type": redact(item.type),
                    "content": walk(item.content, depth + 1),
                }
                for field in (
                    "id",
                    "name",
                    "tool_call_id",
                    "status",
                    "usage_metadata",
                    "tool_calls",
                ):
                    entry = getattr(item, field, None)
                    if entry is not None:
                        result[field] = walk(entry, depth + 1)
                # Only DearFlow-owned provenance is inspectable; provider metadata stays out.
                metadata = {}
                if RECEIPT_KEY in item.additional_kwargs:
                    metadata[RECEIPT_KEY] = receipt_fields(item.additional_kwargs[RECEIPT_KEY])
                ledger = item.additional_kwargs.get(LEDGER_KEY)
                if isinstance(ledger, list):
                    if len(ledger) > 100:
                        truncated = True
                    metadata[LEDGER_KEY] = [receipt_fields(entry) for entry in ledger[:100]]
                if metadata:
                    result["additional_kwargs"] = walk(metadata, depth + 1)
                return result
            if isinstance(item, ModelRequest):
                names = []
                for index, tool in enumerate(item.tools or []):
                    if index >= 100:
                        truncated = True
                        break
                    definition = tool.get("function", tool) if isinstance(tool, dict) else None
                    name = definition.get("name") if isinstance(definition, dict) else getattr(tool, "name", None)
                    if isinstance(name, str):
                        names.append(redact(name))
                return {
                    "messages": walk(item.messages, depth + 1),
                    "system_message": walk(item.system_message, depth + 1),
                    "tools": names,
                    "model_settings": walk(item.model_settings, depth + 1),
                }
            if isinstance(item, ModelResponse):
                return {
                    "result": walk(item.result, depth + 1),
                    "structured_response": walk(item.structured_response, depth + 1),
                }
            if isinstance(item, ToolCallRequest):
                messages = item.state.get("messages", []) if isinstance(item.state, Mapping) else []
                return {
                    "tool_call": walk(item.tool_call, depth + 1),
                    "state": {"messages": walk(messages, depth + 1)},
                }
            if isinstance(item, Command):
                return {"update": walk(item.update, depth + 1)}
            if isinstance(item, Mapping):
                media_type = item.get("type")
                if item.get("source_type") == "base64" or (isinstance(media_type, str) and media_type in _BINARY_TYPES):
                    return {"omitted": "binary content"}
                result = {}
                for index, (key, entry) in enumerate(item.items()):
                    if index >= 100 or nodes >= 1000:
                        truncated = True
                        break
                    if not isinstance(key, str):
                        truncated = True
                        continue
                    safe_key = redact(key)
                    normalized = "".join(char for char in key.lower() if char.isalnum())
                    if any(part in normalized for part in _SECRET_PARTS) or normalized.endswith("token"):
                        result[safe_key] = "[REDACTED]"
                        redaction_applied = True
                    elif normalized in _BINARY_KEYS:
                        result[safe_key] = "[BINARY OMITTED]"
                    else:
                        result[safe_key] = walk(entry, depth + 1)
                return result
            if isinstance(item, (list, tuple)):
                result = []
                for index, entry in enumerate(item):
                    if index >= 100 or nodes >= 1000:
                        truncated = True
                        break
                    result.append(walk(entry, depth + 1))
                return result
            if isinstance(item, str):
                if item.lstrip().lower().startswith("data:"):
                    return "[BINARY OMITTED]"
                return redact(item) if self.capture_content else {"type": "str", "chars": len(item)}
            if item is None:
                return None
            if isinstance(item, (bool, int, float)):
                if isinstance(item, float) and not math.isfinite(item):
                    return {"type": "float"}
                return item if self.capture_content else {"type": type(item).__name__}
            return {"type": type(item).__name__}

        def fit(item: Any, budget: int) -> Any:
            """Trim JSON-safe data to an exact serialized-character budget."""
            nonlocal truncated
            if len(json.dumps(item)) <= budget:
                return item
            truncated = True
            if isinstance(item, str):
                low, high = 0, len(item)
                while low < high:
                    middle = (low + high + 1) // 2
                    if len(json.dumps(item[:middle])) <= budget:
                        low = middle
                    else:
                        high = middle - 1
                return item[:low]
            if isinstance(item, (dict, list)):
                result = {} if isinstance(item, dict) else []
                remaining = budget - 2
                entries = item.items() if isinstance(item, dict) else enumerate(item)
                for key, entry in entries:
                    overhead = (len(json.dumps(key)) + 2 if isinstance(item, dict) else 0) + (2 if result else 0)
                    if remaining - overhead < 4:
                        break
                    fitted = fit(entry, remaining - overhead)
                    if isinstance(result, dict):
                        result[key] = fitted
                    else:
                        result.append(fitted)
                    remaining -= overhead + len(json.dumps(fitted))
                return result
            return None

        result = {
            "value": walk(value),
            "truncated": truncated,
            "content_captured": self.capture_content,
            "redaction_applied": redaction_applied,
        }
        overhead = len(json.dumps({**result, "value": None})) - 4
        result["value"] = fit(result["value"], self.max_chars - overhead)
        result["truncated"] = truncated
        return result
