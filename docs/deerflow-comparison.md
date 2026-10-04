# DearFlow vs. real DeerFlow

Date: 2026-10-04 · Source compared: `bytedance/deer-flow` (main, cloned 2026-10-04) · Living doc: update it before each slice.

**TL;DR** DearFlow keeps DeerFlow's *shape* (one lead agent built by `create_agent`, a
middleware pipeline, skills loaded on demand, delegation through a `task` tool) and drops its
*hardening*. DeerFlow's `executor.py` alone is 2,278 lines, its middleware folder has 56 files
files, and most of that code deals with sandboxes, streaming, cost, safety and audit, not with
what an agent is. A slice is in scope only if it teaches a core idea.

## Scale

| | DeerFlow | DearFlow |
|---|---|---|
| Middleware files | 56 (`agents/middlewares/`) | 8 |
| `lead_agent/agent.py` + `prompt.py` | 1,426 + 1,165 lines | ~110 + ~55 lines |
| Subagent package | 20+ modules, ~6k lines | 6 modules, ~386 lines (incl. `task` tool + limit middleware) |
| Clarification middleware | 579 lines | 94 lines |
| Summarization middleware | 1,080 lines (custom) | LangChain's built-in, 20 lines of wiring |

## Feature by feature

Legend: ✅ same idea, simplified · ⏭ planned · ✂️ deliberately dropped

| Area | DeerFlow | DearFlow |
|---|---|---|
| Lead agent | `make_lead_agent`, runtime flags via `configurable` | ✅ `LeadAgentFactory.build(run_config)` |
| Config | big `AppConfig`, env substitution, hot reload | ✅ small pydantic `AppConfig`, `extra=forbid` |
| Models | many providers, thinking/vision flags, retries | ✅ one provider via `init_chat_model` |
| Tools | config groups + MCP + deferred `tool_search` | ✅ config tools + builtins; ⏭ MCP; ✂️ deferred tool loading |
| Skills | activation middleware, allowed-tools policy, skill package review | ✅ `load_skill` + name/description index; ✂️ policy/authorization |
| Clarification | middleware intercepts `ask_clarification`, ends run | ✅ same |
| Todos | `TodoMiddleware` with context-loss recovery | ✅ guard with reminder budget; ✂️ recovery |
| Date / dynamic context | 806-line middleware (memory, date, more) | ✅ date reminder only |
| Summarization | custom, skill/tool-aware | ✅ LangChain built-in |
| Title | model-written, with retries | ✅ fallback-first, optional model |
| Subagents | background executor, polling, batches, ledger, receipts, acceptance checks | ✅ `task` tool + registry + executor + limit middleware |
| Memory | `MemoryMiddleware` + queued LLM updater | ⏭ slice 6 |
| Persistence / streaming | SQL repos, migrations, run manager, SSE | ⏭ slice 8 (checkpointer + `stream`) |
| Gateway / UI | FastAPI + Next.js frontend | ⏭ slice 9 (FastAPI only) |
| **Sandbox** | local/Docker/AIO sandbox, virtual paths, bash, file tools | ✂️ **by design** |
| Safety & governance | PII redaction, input sanitization, loop detection, finish-reason guards, token budgets, read-before-write | ✂️ |
| Observability | token usage, tool receipts, progress events | ✂️ |
| Uploads / images / artifacts | uploads middleware, `view_image`, `present_files` | ✂️ (state keys reserved) |

## Subagents: what DeerFlow actually does

The parts worth copying:

1. **`task` tool.** `task(prompt, subagent_type, ...)`. The lead model decides when to delegate.
2. **`SubagentConfig`.** `name`, `description`, `system_prompt`, `tools`/`disallowed_tools`,
   `model` (`"inherit"` by default), `max_turns`, `timeout_seconds`.
3. **Registry.** Built-in types plus `custom_agents` from config. An unknown type returns an
   error that lists the valid ones.
4. **Fresh agent per task, isolated context.** The subagent gets its own system prompt, a
   filtered copy of the parent's tools, and only the task prompt as its input.
5. **No nesting.** `task` and `ask_clarification` are on every subagent's denylist.
6. **Result = last AI message text.** If there is none, the sentinel `"No response generated"`.
7. **`SubagentLimitMiddleware`.** On `after_model`, excess `task` calls are dropped from the
   model's response: a per-response concurrency cap (3) and a per-run total cap (6). Code
   enforces the limit, not just the prompt.
8. **Prompt section.** It teaches *when not to delegate*, because delegation costs tokens.
9. **Bounded runs.** `max_turns` becomes the graph `recursion_limit`; `timeout_seconds` caps wall time.

What we drop: background threads and polling, the isolated event loop, batches, the
delegation ledger, receipt verification, acceptance checks, token collectors, the bash
subagent, and context snapshots. Parallelism comes for free: when the model emits several
`task` calls in one response, LangGraph's tool node runs them concurrently.

## Where we deliberately differ

- **Run counting.** DeerFlow keeps a durable `delegations` ledger keyed by `run_id`. We count
  `task` calls since the last real user message, using the hidden-message markers from slice 4.
  The result is the same without extra state.
- **Enablement.** DeerFlow switches subagents on per run (`subagent_enabled`). We mirror that
  with `subagents.enabled` in config plus `configurable.subagent_enabled` and `--subagents`.
