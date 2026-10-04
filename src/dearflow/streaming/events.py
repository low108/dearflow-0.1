"""The event format shared by the CLI, the run log and (next slice) the HTTP API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage

from dearflow.receipts import RECEIPT_KEY

# Names besides the stream modes:
METADATA = "metadata"  # first event: which run this is
ERROR = "error"  # the run failed; `data.message` says why
END = "end"  # last event: `data.status` is success | error | interrupted


def to_jsonable(value: Any) -> Any:
    """Turn graph output into plain JSON data.

    Messages are reduced to the fields a client needs. Everything unknown
    becomes a string rather than failing the stream.
    """
    if isinstance(value, BaseMessage):
        data = {"type": value.type, "id": value.id, "content": value.content}
        if getattr(value, "tool_calls", None):
            data["tool_calls"] = to_jsonable(value.tool_calls)
        if getattr(value, "tool_call_chunks", None):
            data["tool_call_chunks"] = to_jsonable(value.tool_call_chunks)
        if value.type == "tool":
            data["tool_call_id"] = value.tool_call_id
            data["name"] = value.name
            data["status"] = getattr(value, "status", None)
            # The audit evidence travels with the message: the receipt, and for `task` results the citation check.
            if value.additional_kwargs.get(RECEIPT_KEY):
                data["receipt"] = value.additional_kwargs[RECEIPT_KEY]
            if value.additional_kwargs.get("receipt_verdict"):
                data["receipt_verdict"] = value.additional_kwargs["receipt_verdict"]
        if value.additional_kwargs.get("hide_from_ui"):
            data["hidden"] = True
        return data
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class StreamEvent:
    """One thing that happened in a run.

    `id` is the event's position in the run log, so it always increases but can skip
    numbers: a gap means an event that was logged but not requested (or not sent).
    Pass the last id you saw to `RunService.events(run_id, after_seq=...)` to catch up.
    """

    id: int
    event: str
    data: Any

    def sse(self) -> str:
        """Server-Sent Events frame (`event:` / `data:` / `id:`), same field order as DeerFlow."""
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"event: {self.event}\ndata: {payload}\nid: {self.id}\n\n"
