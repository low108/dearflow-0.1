"""Run real-provider middleware examples and verify their LangSmith exports.

Usage from the configured DearFlow shell: uv run python scripts/middleware_demo.py
Synthetic sample data and seeded recovery histories are explicitly labelled.
No provider response or reported token usage is fabricated.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, tracing_context

from dearflow.config import AppConfig
from dearflow.pii import make_redactor
from dearflow.config import expand_env
from dearflow.receipts import RECEIPT_KEY
from dearflow.observability.snapshots import Snapshotter
from middleware_demo_helpers import (
    build_service,
    with_overrides,
    run_seeded,
    loop_history,
    interrupted_history,
    summarization_history,
    task_limit_history,
    todo_history,
)


class Evidence(BaseCallbackHandler):
    """Retain bounded, already-redacted custom spans; count actual model usage."""

    def __init__(self):
        self.spans = {}

    def on_chain_start(
        self, serialized, inputs, *, run_id, name=None, parent_run_id=None, **kwargs
    ):
        if (name or "").startswith("middleware."):
            self.spans[str(run_id)] = {
                "id": str(run_id),
                "name": name,
                "parent": str(parent_run_id),
                "inputs": deepcopy(inputs),
            }

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        if str(run_id) in self.spans:
            self.spans[str(run_id)]["outputs"] = deepcopy(outputs)

    def on_chain_error(self, error, *, run_id, **kwargs):
        if str(run_id) in self.spans:
            self.spans[str(run_id)]["error"] = type(error).__name__


class MemoryEvidence(BaseCallbackHandler):
    """Record only model calls made during the demo's synchronous memory flush.

    A completed model call does not imply successful parsing or persistence;
    the scenario separately checks what was actually saved and injected.
    """

    def __init__(self, redactor):
        self.enabled = False
        self.attempts = {}
        self.snapshot = Snapshotter(
            capture_content=True, max_chars=12000, redactor=redactor
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        if self.enabled:
            self.attempts[str(run_id)] = {
                "run_id": str(run_id),
                "attempt": len(self.attempts) + 1,
                "status": "started",
            }

    def on_llm_end(self, response, *, run_id, **kwargs):
        attempt = self.attempts.get(str(run_id))
        if attempt is not None:
            replies = []
            for batch in response.generations:
                for generation in batch:
                    message = getattr(generation, "message", None)
                    replies.append(
                        {
                            "content": getattr(message, "content", generation.text),
                            "tool_calls": getattr(message, "tool_calls", []),
                            "invalid_tool_calls": getattr(
                                message, "invalid_tool_calls", []
                            ),
                            "usage": getattr(message, "usage_metadata", None),
                        }
                    )
            attempt.update(status="model_completed", reply=self.snapshot(replies))

    def on_llm_error(self, error, *, run_id, **kwargs):
        attempt = self.attempts.get(str(run_id))
        if attempt is not None:
            attempt.update(status="model_error", error=type(error).__name__)


def memory_preferences(text):
    """Recognize the two synthetic preferences without requiring exact prose."""
    text = text.casefold()
    return {
        "green_notebook": bool(
            re.search(r"\bnote\s*books?\b", text) and re.search(r"\bgreen\b", text)
        ),
        "concise_bullets": bool(
            "bullet" in text
            and re.search(r"\b(concise|brief|short|succinct|compact)\b", text)
        ),
    }


def examine(spans):
    """Checks inspect actual changes, not just presence of a middleware name."""
    found = {}

    def mark(key, condition, span):
        if condition:
            found.setdefault(key, span["id"])

    for s in spans:
        name = s["name"]
        before = s.get("inputs", {}).get("before", {})
        output = s.get("outputs", {})
        update = output.get("state_update", {}).get("value")
        result = output.get("result", {}).get("value")
        calls = output.get("calls", [])
        if "DynamicContextMiddleware.before_agent" in name:
            mark("date_injected", "<current_date>" in json.dumps(update), s)
        if "MemoryMiddleware.before_agent" in name:
            text = json.dumps(update)
            mark("memory_injected", "<memory>" in text, s)
            mark(
                "memory_preferences_injected",
                "<memory>" in text and all(memory_preferences(text).values()),
                s,
            )
        if "TitleMiddleware.after_agent" in name:
            mark(
                "title_created",
                isinstance(update, dict) and bool(update.get("title")),
                s,
            )
        if "SummarizationMiddleware.before_model" in name:
            mark(
                "history_summarized",
                isinstance(update, dict) and bool(update.get("messages")),
                s,
            )
        if "TodoMiddleware" in name:
            mark(
                "todo_context_restored",
                "earlier todo list is no longer visible" in json.dumps(update),
                s,
            )
            mark(
                "todo_reminder",
                isinstance(update, dict) and update.get("jump_to") == "model",
                s,
            )
        if "SubagentLimitMiddleware.after_model" in name:
            mark(
                "subagent_limit_enforced",
                "SUBAGENT LIMIT REACHED" in json.dumps(update),
                s,
            )
        if "LoopDetectionMiddleware.after_model" in name:
            mark("loop_stopped", "FORCED STOP" in json.dumps(update), s)
        if "TokenBudgetMiddleware.after_model" in name:
            mark("budget_stopped", "TOKEN BUDGET EXCEEDED" in json.dumps(update), s)
        if "ClarificationMiddleware.wrap_tool_call" in name:
            call = before.get("value", {}).get("tool_call", {})
            mark(
                "clarification_intercepted",
                call.get("name") == "ask_clarification"
                and output.get("handler_calls") == 0
                and bool(result),
                s,
            )
        if "ToolErrorHandlingMiddleware.wrap_tool_call" in name:
            mark(
                "tool_error_recovered",
                isinstance(result, dict)
                and result.get("status") == "error"
                and "Cannot divide by zero" in result.get("content", ""),
                s,
            )
        if "ToolReceiptMiddleware.wrap_tool_call" in name:
            mark(
                "receipt_stamped",
                isinstance(result, dict)
                and RECEIPT_KEY in result.get("additional_kwargs", {})
                and any(
                    RECEIPT_KEY
                    not in c.get("response", {})
                    .get("value", {})
                    .get("additional_kwargs", {})
                    for c in calls
                ),
                s,
            )
        for c in calls:
            req = c.get("request", {})
            req_text = json.dumps(req.get("value"))
            before_text = json.dumps(before.get("value"))
            if "InputSanitizationMiddleware.wrap_model_call" in name:
                mark(
                    "input_sanitized",
                    "&lt;system&gt;" in req_text and "<system>" in before_text,
                    s,
                )
            if "PiiRedactionMiddleware.wrap_model_call" in name:
                mark(
                    "pii_masked",
                    before.get("redaction_applied") is True
                    and req.get("redaction_applied") is False
                    and "[EMAIL_" in req_text,
                    s,
                )
            if "DanglingToolCallMiddleware.wrap_model_call" in name:
                mark(
                    "interrupted_history_repaired",
                    "Tool call was interrupted" in req_text
                    and "Tool call was interrupted" not in before_text
                    and "Fixture orphan result" not in req_text,
                    s,
                )
            if "LoopDetectionMiddleware.wrap_model_call" in name:
                mark(
                    "loop_warning",
                    "[LOOP DETECTED]" in req_text
                    and "[LOOP DETECTED]" not in before_text,
                    s,
                )
            if "TokenBudgetMiddleware.wrap_model_call" in name:
                mark("budget_warning", "[TOKEN BUDGET WARNING]" in req_text, s)
            if "ToolReceiptMiddleware.wrap_model_call" in name:
                mark(
                    "receipt_ledger_injected",
                    req_text != before_text and "receipt" in req_text.lower(),
                    s,
                )
    return found


def write_results(folder, data):
    (folder / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [
        "# Middleware live verification",
        "",
        f"Model: `{data['model']}`. Project: `{data['project']}`.",
        "",
        "Seeded histories and forced tool choices below are test fixtures; continuation calls use the real provider.",
        "",
        "| Scenario | Local behavior | LangSmith | Tokens |",
        "| --- | --- | --- | --- |",
    ]
    for r in data["scenarios"]:
        local = "PASS" if r.get("passed") else "FAIL / incomplete"
        remote = r.get("langsmith_status", "pending")
        if r.get("url"):
            remote = f"[{remote}]({r['url']})"
        lines.append(
            f"| {r['scenario']} | {local} | {remote} | {r.get('usage', {}).get('total_tokens', 0)} |"
        )
    for r in data["scenarios"]:
        lines += [
            "",
            f"## {r['scenario']}",
            "",
            r.get("fixture", "Ordinary live user request."),
            "",
            f"Observed checks: {', '.join(r.get('checks', {})) or 'none'}.",
            f"Missing checks: {', '.join(r.get('missing', [])) or 'none'}.",
            f"LangSmith token totals match local usage: {r.get('tokens_match', 'not verified')}.",
        ]
        if r.get("error"):
            lines.append(f"Error type: `{r['error']}`.")
        if r.get("export_error"):
            lines.append(
                f"LangSmith query error: `{r['export_error']}`; HTTP status: {r.get('export_http_status', 'unavailable')}."
            )
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Demo:
    def __init__(self, config, folder, client, project):
        self.folder, self.client, self.project = folder, client, project
        self.prefix = folder.name
        self.user = self.prefix + "-synthetic-user"
        self.config = with_overrides(
            config,
            memory={"storage_path": str(folder / "memory"), "debounce_seconds": 0},
        )
        self.redactor = make_redactor(
            config.pii_redaction, expand_env(config.pii_redaction.token_secret)
        )
        self.results = []
        self.observed = set()

    def write(self):
        data = {
            "model": self.config.resolve_model(None),
            "project": self.project,
            "synthetic_user": self.user,
            "observed_middleware": sorted(self.observed),
            "scenarios": self.results,
        }
        write_results(self.folder, data)

    def verify_remote(self, record, local_spans):
        for key in (
            "export_error",
            "export_http_status",
            "remote_span_count",
            "remote_snapshots_match",
            "remote_parents_match",
            "remote_tokens",
            "tokens_match",
            "url",
        ):
            record.pop(key, None)
        wait_for_all_tracers()
        self.client.flush()
        expected = {s["id"] for s in local_spans}
        for attempt in range(5):
            try:
                runs = list(
                    self.client.list_runs(
                        project_name=self.project, trace_id=record["run_id"]
                    )
                )
                remote = {str(r.id): r for r in runs}
                missing = expected - remote.keys()
                if not missing and runs:
                    identical = all(
                        remote[s["id"]].inputs == s.get("inputs")
                        and remote[s["id"]].outputs == s.get("outputs")
                        for s in local_spans
                    )
                    record["remote_span_count"] = len(runs)
                    record["remote_snapshots_match"] = identical
                    parents_match = all(
                        str(remote[s["id"]].parent_run_id) == s["parent"]
                        for s in local_spans
                    )
                    record["remote_parents_match"] = parents_match
                    root = remote.get(record["run_id"])
                    if root:
                        record["url"] = self.client.get_run_url(run=root)
                    record["remote_tokens"] = sum(
                        (r.total_tokens or 0) for r in runs if r.run_type == "llm"
                    )
                    record["tokens_match"] = (
                        record["remote_tokens"] == record["usage"]["total_tokens"]
                    )
                    if (
                        root
                        and root.end_time
                        and identical
                        and parents_match
                        and record["tokens_match"]
                    ):
                        record["langsmith_status"] = "verified"
                        record.pop("export_error", None)
                        record.pop("export_http_status", None)
                        return
            except Exception as exc:
                record["export_error"] = type(exc).__name__
                # Exception text can contain request payloads or credentials. Keep
                # only the HTTP status from the requests exception chain.
                current, seen = exc, set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    status = getattr(
                        getattr(current, "response", None), "status_code", None
                    )
                    if isinstance(status, int):
                        record["export_http_status"] = status
                        break
                    current = current.__cause__ or current.__context__
            time.sleep(3)
        record["langsmith_status"] = "not verified"

    def run(
        self,
        name,
        *,
        prompt=None,
        seed=None,
        overrides=None,
        forced_tool=None,
        expected=(),
        fixture=None,
    ):
        print(f"START {name}", flush=True)
        cfg = with_overrides(self.config, **(overrides or {}))
        extraction = MemoryEvidence(self.redactor)
        factory, service, db = build_service(
            cfg, forced_tool=forced_tool, model_callback=extraction
        )
        recorder = Evidence()
        service._agent.graph = service._agent.graph.with_config(
            callbacks=[recorder],
            recursion_limit=80,
            tags=["middleware-live-demo"],
            metadata={
                "demo_suite": self.prefix,
                "scenario": name,
                "synthetic_data": True,
                "forced_tool_choice": forced_tool or "auto",
            },
        )
        record = {
            "scenario": name,
            "fixture": fixture
            or "Ordinary live request; no seeded history or forced tool choice.",
        }
        self.results.append(record)
        try:
            memory_before = factory.memory.context(self.user) if factory.memory else ""
            record["memory_before"] = self.redactor.redact(memory_before)
            if seed is None:
                result = service.invoke(
                    prompt, thread_id=f"{self.prefix}-{name}", user_id=self.user
                )
            else:
                result = run_seeded(
                    service,
                    label=name,
                    state=seed,
                    thread_id=f"{self.prefix}-{name}",
                    user_id=self.user,
                    fixture_details=fixture,
                )
            record.update(
                run_id=str(__import__("uuid").UUID(result.run_id)),
                usage=result.usage,
                final_text=self.redactor.redact(result.text),
                title=result.state.get("title"),
                todos=result.state.get("todos"),
                receipts=len(service.receipts(result.thread_id)),
            )
            spans = list(recorder.spans.values())
            checks = examine(spans)
            if name == "memory-recall" and all(
                memory_preferences(result.text).values()
            ):
                checks["memory_preferences_recalled"] = result.run_id
            if result.state.get("todos") and all(
                t.get("status") == "completed" for t in result.state["todos"]
            ):
                checks["todos_completed"] = result.run_id
            delegated = [
                r
                for r in service.receipts(result.thread_id)
                if r.get("tool_name") == "task" and r.get("status") == "success"
            ]
            if len(delegated) >= 2 and result.usage.get("subagent_tokens", 0) > 0:
                checks["live_subagents_completed"] = result.run_id
            # Flush uses the real extraction model; verify persisted synthetic-user facts.
            extraction.enabled = True
            try:
                factory.flush_memory()
            finally:
                extraction.enabled = False
            record["memory_extraction"] = {
                "attempt_count": len(extraction.attempts),
                "attempts": list(extraction.attempts.values()),
            }
            memory_after = factory.memory.context(self.user) if factory.memory else ""
            record["memory_after"] = self.redactor.redact(memory_after)
            if memory_after and memory_after != memory_before:
                checks["memory_persisted"] = result.run_id
                before_preferences = memory_preferences(memory_before)
                after_preferences = memory_preferences(memory_after)
                if all(after_preferences.values()) and not any(
                    before_preferences.values()
                ):
                    checks["memory_preferences_persisted"] = result.run_id
            record["checks"] = checks
            record["missing"] = sorted(set(expected) - checks.keys())
            record["passed"] = not record["missing"]
            self.observed.update(s["name"].split(".")[1] for s in spans)
            (self.folder / f"{name}.snapshots.json").write_text(
                json.dumps(spans, indent=2), encoding="utf-8"
            )
            self.verify_remote(record, spans)
        except Exception as exc:
            record.update(
                passed=False, error=type(exc).__name__, langsmith_status="not verified"
            )
        finally:
            factory.flush_memory()
            db.close()
            self.write()
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in (
                        "scenario",
                        "run_id",
                        "passed",
                        "missing",
                        "langsmith_status",
                        "tokens_match",
                        "error",
                    )
                }
            ),
            flush=True,
        )


def verify_saved(folder, client):
    """Recheck existing exports; never create an agent or call a model."""
    data = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    verifier = object.__new__(Demo)
    verifier.client, verifier.project = client, data["project"]
    for record in data["scenarios"]:
        snapshots = folder / f"{record['scenario']}.snapshots.json"
        if not record.get("run_id") or not snapshots.is_file():
            record["langsmith_status"] = "not verified"
            record["export_error"] = "MissingSavedEvidence"
        else:
            spans = json.loads(snapshots.read_text(encoding="utf-8"))
            verifier.verify_remote(record, spans)
        write_results(folder, data)
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in (
                        "scenario",
                        "langsmith_status",
                        "tokens_match",
                        "export_error",
                        "export_http_status",
                    )
                }
            ),
            flush=True,
        )
    return (
        0
        if all(
            r.get("passed") and r.get("langsmith_status") == "verified"
            for r in data["scenarios"]
        )
        else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--memory-smoke",
        action="store_true",
        help="Run only simple memory learning and recall, without tool calls or delegation",
    )
    mode.add_argument(
        "--memory-only",
        action="store_true",
        help="Run memory learning and full-flow injection in a fresh demo folder",
    )
    mode.add_argument(
        "--verify-only",
        metavar="RESULT_FOLDER",
        help="Recheck saved LangSmith exports without any model calls",
    )
    args = parser.parse_args()
    required = [] if args.verify_only else ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    missing = [name for name in required if not os.environ.get(name)]
    if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        missing.append("LANGSMITH_API_KEY")
    if missing:
        print(
            "Live verification requires these variables in this shell: "
            + ", ".join(missing)
        )
        print(
            "Run this command in the terminal already configured for DearFlow; do not paste keys into chat."
        )
        return 2
    if args.verify_only:
        client = Client()
        try:
            return verify_saved(Path(args.verify_only).resolve(), client)
        finally:
            client.close()
    cfg = AppConfig.from_yaml(args.config)
    if not all(
        (
            cfg.plan_mode,
            cfg.memory.enabled,
            cfg.pii_redaction.enabled,
            cfg.token_budget.enabled,
            cfg.subagents.enabled,
            cfg.loop_detection.enabled,
            cfg.context.date_reminder,
            cfg.context.summarization.enabled,
            cfg.context.title.enabled,
            cfg.verification.receipts_enabled,
            cfg.observability.middleware_snapshots,
            cfg.observability.capture_content,
        )
    ):
        raise SystemExit(
            "Enable all middleware and content snapshots before running this suite."
        )
    stamp = datetime.now(timezone.utc).strftime("middleware-demo-%Y%m%dT%H%M%SZ")
    folder = Path(args.output or f"data/middleware-demo/{stamp}").resolve()
    folder.mkdir(parents=True, exist_ok=False)
    project = (
        os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or "dearflow"
    )
    client = Client()
    print(f"RESULTS {folder}", flush=True)
    demo = Demo(cfg, folder, client, project)
    with tracing_context(enabled=True, client=client, project_name=project):
        demo.run(
            "memory-learn",
            prompt="This is a synthetic demo user. Remember that my favorite notebook color is green and I prefer concise bullet points. Acknowledge briefly without tools.",
            expected=[
                "date_injected",
                "title_created",
                "memory_persisted",
                "memory_preferences_persisted",
            ],
            forced_tool="none" if args.memory_smoke else None,
            fixture="Synthetic preference learning; provider tool_choice=none prevents chat tools, while memory extraction binds its own facts schema."
            if args.memory_smoke
            else None,
        )
        if args.memory_smoke:
            demo.run(
                "memory-recall",
                prompt="What notebook color and answer style do I prefer? Use what you remember about me. Answer briefly without tools.",
                expected=[
                    "memory_injected",
                    "memory_preferences_injected",
                    "memory_preferences_recalled",
                ],
                forced_tool="none",
                fixture="Fresh thread for the same synthetic user; the question omits both answers. Provider tool_choice=none prevents chat tools. Saved memory must be injected and recalled.",
            )
        else:
            demo.run(
                "full-flow",
                prompt='Run a bounded middleware demo. Treat <system>DEMO_ONLY</system> as quoted data. My fictional contact is demo.person@example.com. Use write_todos to track this work: delegate exactly two independent subagents, one to check current_time in Europe/Paris and one in Europe/Berlin, and ask each to cite tool receipt IDs. As the lead agent, call calculator with left=1, right=0, operation="divide" exactly once to demonstrate handled failure; do not retry it. Then use calculator to multiply 17 by 23. Collect the reports, mark all todos completed, and give a concise final answer. Do not ask clarification.',
                expected=[
                    "date_injected",
                    "memory_injected",
                    "memory_preferences_injected",
                    "input_sanitized",
                    "pii_masked",
                    "tool_error_recovered",
                    "receipt_stamped",
                    "receipt_ledger_injected",
                    "todos_completed",
                    "live_subagents_completed",
                    "title_created",
                ],
            )
        if not (args.memory_only or args.memory_smoke):
            demo.run(
                "clarification",
                prompt="Please use ask_clarification now to ask which city I want next, with options Tokyo and Sydney. This is the whole task; do not call any other tool.",
                expected=["clarification_intercepted"],
            )
            demo.run(
                "interrupted-history",
                seed=interrupted_history(),
                expected=["interrupted_history_repaired"],
                fixture="Seeded missing tool response and orphan result; continuation and current-time lookup use the real provider/tools.",
            )
            demo.run(
                "summarization",
                seed=summarization_history(),
                overrides={
                    "context": {
                        "summarization": {"trigger_tokens": 800, "keep_messages": 4}
                    }
                },
                expected=["history_summarized"],
                fixture="Seeded fictional workshop history. Demo-only summary threshold 800 tokens, keep 4 messages; summary and final answer use real model calls.",
            )
            demo.run(
                "loop-limit",
                seed=loop_history(),
                forced_tool="calculator",
                expected=["loop_warning", "loop_stopped"],
                fixture="Three seeded identical completed calculator calls. Provider tool_choice forces calculator on real continuation calls; model generates arguments. Production loop thresholds remain 3/5.",
            )
            demo.run(
                "delegation-limit",
                seed=task_limit_history(),
                forced_tool="task",
                overrides={"subagents": {"max_total": 3}},
                expected=["subagent_limit_enforced"],
                fixture="Three seeded completed delegations; demo-only max_total=3. Real next model response is forced to request task and middleware must remove it before execution.",
            )
            demo.run(
                "todo-reminder",
                seed=todo_history(),
                forced_tool="none",
                expected=["todo_context_restored", "todo_reminder"],
                fixture="Seeded pending todo with missing original tool history; provider tool_choice=none forces real final-text responses. The guard reminds twice, then stops with todo still pending by design.",
            )
            demo.run(
                "budget-warning",
                prompt="Call current_time for Europe/Paris, then report the result in a sentence. No delegation or todo list is needed.",
                overrides={
                    "token_budget": {"max_tokens": 200000, "warn_threshold": 0.001}
                },
                expected=["budget_warning"],
                fixture="Ordinary live tool request. Demo-only soft warning at 0.1% of the 200000-token cap; actual provider usage triggers it.",
            )
            demo.run(
                "budget-stop",
                prompt="Use calculator to multiply 17 by 23 and report the result.",
                overrides={"token_budget": {"max_tokens": 1000}},
                expected=["budget_stopped"],
                fixture="Ordinary live request with a demo-only 1000-token budget; real provider token usage must trigger the stop.",
            )
    client.flush()
    client.close()
    print(f"REPORT {folder / 'report.md'}", flush=True)
    return (
        0
        if all(
            r.get("passed")
            and r.get("langsmith_status") == "verified"
            and r.get("tokens_match")
            for r in demo.results
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
