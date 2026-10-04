"""Checks a subagent's report against its receipts.

Advisory, not a gate. The result says `citation_resolved`, never "verified" or "passed":
a resolved citation shows the cited call happened and succeeded, nothing about whether
the sentence next to it is true. Pure functions: no I/O, no model call.
"""

from __future__ import annotations

import re
from typing import TypedDict

from dearflow.receipts.receipt import Receipt, parse_citations

VERDICT_SOURCE = "receipt_citations"
VERDICT_REQUIREMENT = "cited_ids_in_execution_record"

# A report that claims actions but cites nothing is a weak-negative sign. Biased toward
# catching: a false alarm costs one "UNVERIFIED" line, a miss lets a false success through.
_ACTION_VERB = re.compile(
    r"\b(wrote|written|created|saved|generated|ran|executed|uploaded|downloaded|deleted|modified|updated"
    r"|installed|deployed|fetched|built|compiled|produced|exported|fixed|added|changed|removed|implemented"
    r"|patched|refactored|renamed|moved|merged|committed|edited|replaced|tested|verified|cleaned|configured)\b",
    re.IGNORECASE,
)
# Chinese has no word boundaries, so the English list never fires on it; match the verbs directly.
_CJK_ACTION_VERB = re.compile(r"创建|生成|写入|保存|修改|更新|删除|运行|执行|安装|部署|上传|下载|修复|添加|新增|编写|编译|构建|导出|测试|提交|移动|重命名|配置|替换|清理")
_FILE_PATH = re.compile(r"(?:/[\w.\-]+){2,}|\b[\w.\-]+\.(?:py|md|txt|json|ya?ml|csv|html|js|ts|sh|log|pdf|png|jpe?g)\b")
# Safety net beyond any word list: the subagent DID run tools, so a paragraph-sized report that
# cites none of them is unverified whatever the language. Short confirmations pass.
_LONG_REPORT_CHARS = 240

LIMITATION = "execution evidence only, does not validate claim correctness"


class CitationFailure(TypedDict):
    id: str
    reason: str


class ReceiptVerdict(TypedDict):
    source: str
    requirement: str
    citation_resolved: bool
    cited: list[str]
    resolved: list[str]
    failed: list[CitationFailure]
    unknown: list[str]
    no_citation_claims: bool


def has_action_claims(text: str) -> bool:
    return bool(_ACTION_VERB.search(text) or _CJK_ACTION_VERB.search(text) or _FILE_PATH.search(text))


def verify_receipt_citations(report: str, receipts: list[Receipt]) -> ReceiptVerdict:
    """Check every `[rN]` in `report` against the receipts the subagent could see."""
    by_id = {r["id"]: r for r in receipts}
    cited: list[str] = []
    resolved: list[str] = []
    failed: list[CitationFailure] = []
    unknown: list[str] = []
    for rid, anchor in parse_citations(report):
        cited.append(rid)
        receipt = by_id.get(rid)
        if receipt is None:
            unknown.append(rid)
        elif receipt["status"] != "success":
            failed.append({"id": rid, "reason": f"receipt status={receipt['status']}"})
        elif anchor is not None and anchor != receipt["tool_name"]:
            failed.append({"id": rid, "reason": f"anchor mismatch: cited as {anchor}, receipt {rid} is {receipt['tool_name']}"})
        else:
            resolved.append(rid)

    long_report = bool(receipts) and len(report.strip()) >= _LONG_REPORT_CHARS
    uncited_claims = not cited and (has_action_claims(report) or long_report)
    ok = (not failed and not unknown) if cited else not uncited_claims
    return ReceiptVerdict(
        source=VERDICT_SOURCE,
        requirement=VERDICT_REQUIREMENT,
        citation_resolved=ok,
        cited=cited,
        resolved=resolved,
        failed=failed,
        unknown=unknown,
        no_citation_claims=uncited_claims,
    )


def render_verdict(verdict: ReceiptVerdict) -> str:
    """One line for the lead agent to read ("" when there is nothing to say)."""
    if verdict["no_citation_claims"]:
        return "citations: UNVERIFIED - action claims without receipt citations"
    if not verdict["cited"]:
        return ""
    parts = [f"{len(verdict['resolved'])} resolved"]
    if verdict["failed"]:
        parts.append(f"{len(verdict['failed'])} failed")
    if verdict["unknown"]:
        parts.append(f"{len(verdict['unknown'])} unknown")
    return f"citations: {', '.join(parts)} - {LIMITATION}"
