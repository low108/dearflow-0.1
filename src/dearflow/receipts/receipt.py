"""Tool-call receipts: a record, made by code, that a tool call happened.

Every tool result gets a receipt stamped into its message by `ToolReceiptMiddleware`.
A receipt holds the tool name, the status, hashes of the arguments and the output, and the
output size. It does NOT hold the output itself. Receipts are never stored anywhere else:
they are read back out of the messages, so what the model is shown and what the parent
checks can't disagree.

Display ids (`r1`, `r2`, ...) are positions in the message list.
Caveat (same as DeerFlow): if summarization drops old tool messages, the survivors renumber.
That is why the receipt list a reply was shown is also stamped on that reply (`LEDGER_KEY`),
and citations are checked against that snapshot, not against the messages as they are later.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, ToolMessage

RECEIPT_KEY = "dearflow_tool_receipt"  # on a ToolMessage's additional_kwargs
LEDGER_KEY = "dearflow_tool_receipt_ledger"  # on an AIMessage: the receipts it was shown

RECEIPT_ID_PREFIX = "r"
_HASH_CHARS = 16
_MAX_ID_DIGITS = 10
_LEDGER_CHAR_BUDGET = 2000

# `[r2]`, or anchored with the tool name: `[r2 write_file]`. The anchor lets the checker notice
# a citation that points at the wrong kind of call.
CITATION = re.compile(rf"\[{RECEIPT_ID_PREFIX}(\d+)(?:\s+([A-Za-z_][\w.-]*))?\]")


class Receipt(TypedDict):
    id: str  # display id, assigned when receipts are listed ("r1"...)
    tool_call_id: str
    tool_name: str
    status: str  # success | error
    args_sha256: str
    output_sha256: str
    output_bytes: int
    created_at: str


def receipt_id(position: int) -> str:
    """Display id of the `position`-th receipt (1-based)."""
    return f"{RECEIPT_ID_PREFIX}{position}"


def format_citation(rid: str, tool_name: str | None = None) -> str:
    return f"[{rid} {tool_name}]" if tool_name else f"[{rid}]"


def parse_citations(text: str) -> list[tuple[str, str | None]]:
    """`(id, anchor)` pairs cited in a report, each once, in order of first appearance."""
    seen: set[tuple[str, str | None]] = set()
    found: list[tuple[str, str | None]] = []
    for match in CITATION.finditer(text):
        if len(match.group(1)) > _MAX_ID_DIGITS:
            continue  # model output is untrusted: an absurdly long number is not a citation
        citation = (receipt_id(int(match.group(1))), match.group(2))
        if citation not in seen:
            seen.add(citation)
            found.append(citation)
    return found


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:_HASH_CHARS]


def make_receipt(tool_call: dict[str, Any], message: ToolMessage) -> dict[str, Any]:
    """The receipt for one tool call and its result (no display id yet)."""
    args = tool_call.get("args")
    args_bytes = json.dumps(args if isinstance(args, dict) else {}, sort_keys=True, default=str).encode()
    content = message.content if isinstance(message.content, str) else json.dumps(message.content, sort_keys=True, default=str)
    return {
        "tool_call_id": str(tool_call.get("id") or ""),
        "tool_name": str(tool_call.get("name") or ""),
        "status": str(getattr(message, "status", None) or "success"),
        "args_sha256": _short_hash(args_bytes),
        "output_sha256": _short_hash(content.encode()),
        "output_bytes": len(content.encode()),
        "created_at": datetime.now(UTC).isoformat(),
    }


_STRING_FIELDS = ("tool_call_id", "tool_name", "status", "args_sha256", "output_sha256", "created_at")


def is_valid_receipt(value: object) -> bool:
    """Shape check. Receipts come back out of saved checkpoints, so nothing is trusted."""
    if not isinstance(value, dict) or any(not isinstance(value.get(f), str) for f in _STRING_FIELDS):
        return False
    size = value.get("output_bytes")
    return isinstance(size, int) and not isinstance(size, bool)


def _numbered(receipt: dict[str, Any], rid: str) -> Receipt:
    return Receipt(id=rid, **{k: receipt[k] for k in (*_STRING_FIELDS, "output_bytes")})  # type: ignore[typeddict-item]


def extract_receipts(messages: list) -> list[Receipt]:
    """Every valid stamped receipt in message order, numbered r1..rN. Bad entries are skipped."""
    receipts: list[Receipt] = []
    for message in messages:
        if isinstance(message, ToolMessage) and is_valid_receipt(message.additional_kwargs.get(RECEIPT_KEY)):
            receipts.append(_numbered(message.additional_kwargs[RECEIPT_KEY], receipt_id(len(receipts) + 1)))
    return receipts


def extract_citing_turn_receipts(messages: list) -> list[Receipt] | None:
    """The receipts the LAST reply that was shown a ledger could cite (its stamped snapshot).

    None means "unknown": nothing was stamped, or the snapshot is malformed. Callers must
    treat that as "can't verify", never fall back to renumbering the current messages.
    """
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or LEDGER_KEY not in message.additional_kwargs:
            continue
        raw = message.additional_kwargs[LEDGER_KEY]
        if not isinstance(raw, list):
            return None
        receipts: list[Receipt] = []
        first: int | None = None
        for index, entry in enumerate(raw):
            if not is_valid_receipt(entry):
                return None
            match = re.fullmatch(rf"{RECEIPT_ID_PREFIX}([1-9]\d*)", str(entry.get("id")))
            if match is None:
                return None
            first = int(match.group(1)) if first is None else first
            if entry["id"] != receipt_id(first + index):  # ids must be consecutive
                return None
            receipts.append(_numbered(entry, entry["id"]))
        return receipts
    return None


def render_ledger(receipts: list[Receipt], *, max_chars: int = _LEDGER_CHAR_BUDGET) -> tuple[str, list[Receipt]]:
    """Ledger text for the model, plus exactly the receipts that made it into that text.

    When there are too many to fit, the oldest are dropped and the text says so. The returned
    list is what gets stamped on the reply, so a citation can never point at an entry the
    model was not shown.
    """
    if not receipts:
        return "", []
    header = [
        "## Tool receipts (execution record)",
        f"Cite receipt ids (e.g. {format_citation(receipt_id(1), 'write_file')}) in your final report for every claim about an action you took.",
        "Execution evidence only: receipts record that a call happened and its status; they do not validate claim correctness or task acceptance.",
    ]
    lines = [
        f"- [{r['id']}] {r['tool_name']} status={r['status']} args_sha256={r['args_sha256']} output_sha256={r['output_sha256']}"
        for r in receipts
    ]
    if len("\n".join([*header, *lines])) <= max_chars:
        return "\n".join([*header, *lines]), receipts
    omission = "- ... older receipts omitted (context budget)"
    kept: list[str] = []
    for line in reversed(lines):
        if len("\n".join([*header, omission, line, *kept])) > max_chars:
            break
        kept.insert(0, line)
    text = "\n".join([*header, omission, *kept])
    return text, receipts[len(receipts) - len(kept) :] if kept else []
