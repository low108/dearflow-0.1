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
| Tools | config groups + MCP + deferred `tool_search` | ✅ config tools + builtins + MCP; ✂️ deferred tool loading |
| Skills | activation middleware, allowed-tools policy, skill package review | ✅ `load_skill` + name/description index; ✂️ policy/authorization |
| Clarification | middleware intercepts `ask_clarification`, ends run | ✅ same |
| Todos | `TodoMiddleware` with context-loss recovery | ✅ guard with reminder budget; ✂️ recovery |
| Date / dynamic context | 806-line middleware (memory, date, more) | ✅ date reminder only |
| Summarization | custom, skill/tool-aware | ✅ LangChain built-in |
| Title | model-written, with retries | ✅ fallback-first, optional model |
| Subagents | background executor, polling, batches, ledger, receipts, acceptance checks | ✅ `task` tool + registry + executor + limit middleware |
| Memory | `MemoryMiddleware` + debounce queue + LLM updater; facts as Markdown files; pluggable backends | ✅ slice 6: same flow, one JSON file per user |
| Persistence | LangGraph checkpointer (memory/SQLite/Postgres) + SQL tables `threads_meta`, `runs`, `run_events` | ✅ slice 8: SQLite, plain `sqlite3` |
| Streaming | `StreamBridge` + `RunManager`, SSE, modes `values`/`messages-tuple`/`updates`/... | ✅ slice 9: `RunService.stream` + run log |
| Gateway / UI | FastAPI + Next.js frontend | ✅ slice 10: same routes and body fields, no UI |
| **Sandbox** | local/Docker/AIO sandbox, virtual paths, bash, file tools | ✂️ **by design** |
| Token usage + budget | usage logging, per-run budget with soft warning and hard stop | ✅ slice 11 |
| PII redaction | regex detectors + HMAC placeholders at model/tool boundaries | ✅ slice 12 |
| Tool receipts | hashed audit record per tool call, `[rN]` citations checked against the ledger | ⏭ slice 13 |
| Other safety | input sanitization, loop detection, finish-reason guards, read-before-write | ✂️ |
| Other observability | progress events, tool output synopsis | ✂️ |
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

## Memory (slice 6): what we kept

DeerFlow (`agents/memory/`, 16k lines across backends) does: `MemoryMiddleware.after_agent` queues
user messages plus final AI replies (no tool calls) -> a debounce queue batches a burst of turns
-> an LLM extracts facts (`content`, `category`, `confidence`) -> facts are stored per user ->
`<memory>` is injected as a user-role block. We keep all of that.

We drop: swappable backends (DeerMem, mem0, OpenViking, Honcho), summaries (`user`/`history`),
staleness review and consolidation, pre-screen and signal classifiers, agent-scoped memory,
model-facing tools (`memory_search`/`add`/`update`/`delete`), relevance ranking by query, and
the PII-redacted queue (comes with slice 12). Facts are ranked by confidence then recency and cut to a token budget.

## MCP (slice 7): what we kept

DeerFlow (`mcp/`, ~6k lines) uses `langchain-mcp-adapters`' `MultiServerMCPClient`, reads
`extensions_config.json` (`mcpServers`: `stdio` / `http` / `sse`, `$ENV` values, `enabled`,
`tool_name_prefix` default on), loads tools lazily, caches them, and skips a server that fails.
We keep exactly that, with the config inside `config.yaml`, plus one convenience: sync
`graph.invoke` works, because MCP tools are async-only and we wrap them.

We drop: the persistent stdio session pool (loop-aware LRU, disconnect recovery), hot reload
by file signature, OAuth and per-user/per-request credentials, interceptors, routing hints,
long-running MCP tasks (SQL-backed), stdio file-path translation, and the HTTP-boundary launch
policy. If you expose config editing over an API later, add that launch screening: a stdio
`command` is arbitrary code execution.

## Persistence (slice 8): what we kept

DeerFlow splits storage in two: the LangGraph checkpointer holds messages (memory / SQLite / Postgres),
and SQL tables hold metadata (`threads_meta`, `runs`, `run_events`). It allows one pending or running
run per thread (`multitask_strategy: reject`) and stores a preview on each run (`first_human_message`,
`last_ai_message`, `message_count`). We keep all of that, as `threads` and `runs` tables in one SQLite
file, written with plain `sqlite3` instead of SQLAlchemy, plus crash recovery at startup.

We drop: Postgres, migrations, ORM, multi-worker leases and cancel flags, idempotency keys, projects,
`run_events` (comes back with streaming in slice 9), token columns (slice 11), per-user access checks
(a thread records its owner; the gateway in slice 10 will enforce it), and async savers (slice 10).

## Streaming (slice 9): what we kept

DeerFlow's worker runs `agent.astream(input, stream_mode=[...])`, publishes every chunk to a
`StreamBridge` keyed by run id, and an SSE endpoint subscribes. Modes are an allowlist
(`messages-tuple` means LangGraph `messages`); unknown modes are rejected, never silently changed.
Events are `metadata` first, `end` last, and the same frames go into a `run_events` table.
We keep: the allowlist and alias, event names, SSE frame format (`event:` / `data:` / `id:`),
`updates` always logged, and the persisted run log. Our producer and consumer are one generator
instead of two sides of a bridge: `RunService.stream`.

We drop: the bridge (memory and Redis), heartbeats, `Last-Event-ID` reconnect and gap detection
(we offer `events(run_id, after_seq)` for catch-up), joining a running stream from another
client, `debug`/`tasks`/`checkpoints` modes, subgraph streaming, cancel with rollback.

Added for teaching: `custom` events from the `task` tool (`subagent.start` / `subagent.end`).
That makes delegation visible, where DeerFlow emits task events from its background executor.

## Gateway (slice 10): what we kept

Routes, request body and status codes follow `app/gateway/routers/thread_runs.py` and `threads.py`:
`POST /runs` starts a background run; `/runs/stream` adds SSE with `Content-Location`; `/runs/wait`
blocks and returns the state; `/join` attaches without owning the run; `/cancel` answers 202 or 204;
`/events` lists the run log. `RunCreateRequest` is `extra="forbid"` and rejects unsupported options
by name (`unsupported_run_option`, `unsupported_stream_mode`). `on_disconnect` defaults to `cancel`
and joins never apply it. `multitask_strategy: reject` means 409 for a busy thread. Other users'
threads are 404, not 403. Thread ids use DeerFlow's pattern `^[A-Za-z0-9_-]{1,64}$`.

We drop: user accounts (JWT, CSRF, roles, `require_permission`), the LangGraph-Platform assistants
compat layer, idempotency keys, `rollback`/`interrupt` multitask strategies and cancel-with-rollback,
checkpoint ids and history endpoints, branches, projects, goals, feedback, uploads, artifacts,
multi-worker leases and cross-process bridges, pagination endpoints. Auth is one optional API key plus
an `X-User-Id` header.

Different on purpose: the API is sync Python under async routes (`asyncio.to_thread`), so the
checkpointer stays the simple `SqliteSaver`; DeerFlow is async end to end. The bridge is one
in-memory event list per run (`ActiveRun`). Cancel acts between events: a model call already in flight
finishes first.

## Token usage + budget (slice 11): what we kept

From `token_budget_middleware.py` and `token_usage_middleware.py`: per-run scope keyed by `run_id`; a limit on total,
input or output tokens (same config names and validation: `max_tokens >= 1000`, `warn_threshold <= hard_stop_threshold`);
the soft warning is queued in `after_model` and delivered in `wrap_model_call` as a `HumanMessage(name="budget_warning")` on the
NEXT request only, so tool-call/result pairs stay intact; the hard stop never raises, it strips tool calls from every provider
surface (structured, raw `additional_kwargs`, content blocks) and appends `[TOKEN BUDGET EXCEEDED]`; the run records
`stop_reason=token_capped`; run rows store input/output/total tokens, `llm_call_count`, lead/subagent/middleware split and
tokens by model; `GET /threads/{id}/token-usage` aggregates them.

**Different on purpose:** DeerFlow measures by summing `usage_metadata` over AI messages and folds subagent usage into the
message that dispatched it (`SubagentTokenCollector` + a merge step in `TokenUsageMiddleware`). We count with one callback
(`UsageCallback`) on the run config. LangChain callbacks flow into nested agents, so the lead agent, subagents and
middleware model calls are all seen by one counter, tagged by caller, with no message surgery. Consequences: tokens from
model calls outside a run (memory extraction in its timer thread) are not counted, and the budget only works under
`RunService`. A hard stop on a message that already has no tool calls leaves the answer untouched (DeerFlow appends the
notice anyway). We drop per-step attribution metadata for the UI, the `context_usage` block, and cache-token accounting.

## PII redaction (slice 12): what we kept

A port of `pii_redaction_middleware.py` and its config: the same five detectors in the same pinned order
(email, api_key, national_id, credit_card, phone), the same patterns (digit lookarounds instead of `\b` so Chinese labels
don't hide a number), the same checksums (Luhn, GB 11643 for CN resident IDs, CPF mod-11; CUIT and RFC are format-only),
the same placeholder (HMAC-SHA256 of `category\0value` under a required 16+ character secret, 128 bits as 27 base-26 letters, so
a token has no digit runs), per-detector switches, default off. Scope matches: user messages are rewritten in the request only
(the state keeps the original and every call redacts the history again), outside tool results are rewritten at the tool
boundary (MCP tools by tag, others by name), subagents are covered, and the memory queue, titles (before truncation) and
summarization input (`redact_queued_messages`, title, compaction seams) use the same redactor. Errors fail open with a warning.
It sits innermost among wrappers, just before the clarification middleware.

We drop: redaction of the durable-context summary (we have none), the tool-transform audit marker, the hardcoded list of
first-party web tool names (ours is `pii_redaction.tool_names`, since DearFlow has no web tools yet), and the independent
deep-copy fallbacks (our rewrite builds new strings and never mutates messages). Not covered by either project: the
assistant's own replies and raw storage (checkpoints, run previews, run log), which keep the original text.

## Planned hardening slices (how DeerFlow does it, and what we keep)

Read from DeerFlow at `agents/middlewares/`, `persistence/`, `backend/app/gateway/`.

### Slice 8: persistence
- **DeerFlow:** two layers. The LangGraph checkpointer holds thread state and messages
  (memory, SQLite or Postgres). Separate SQL tables hold metadata: `threads_meta`, `runs`
  (status, stop reason, token totals), `run_events`. There is no messages table.
- **Keep:** the same two layers. `SqliteSaver` as checkpointer, plus tiny `threads` and `runs` tables.
- **Drop:** Postgres, migrations, ownership and users, projects, scheduled tasks.

### Slice 9: streaming
- **DeerFlow:** `RunManager` tracks runs, `StreamBridge` carries live events, `sse_consumer` turns
  them into Server-Sent Events. Public modes: `values`, `messages-tuple` (= LangGraph
  `messages`), `updates`, `debug`, `tasks`, `checkpoints`, `custom`. Unsupported modes are rejected.
- **Keep:** `agent.stream(..., stream_mode=[...])`, a mode allowlist, and a `Run` record with status.
- **Drop:** join or reconnect to a running stream, event replay, subgraph streaming.

### Slice 10: gateway
- `POST /threads/{id}/runs/stream` returns `text/event-stream`; `GET /threads/{id}` reads state.

### Slice 11: token usage + budget
- **DeerFlow:** `TokenUsageMiddleware` reads `usage_metadata` from each model reply.
  `TokenBudgetMiddleware` keeps a per-run total. At the soft threshold it queues a warning
  (inserted as a message at the next model call, so tool-call/result pairs stay intact). At the
  hard threshold it strips the tool calls and adds a stop message, so the agent gives a final
  answer instead of raising. Subagent usage is added to the message that launched it.
- **Keep:** usage totals in state, soft warning, hard stop by stripping tool calls, and
  subagent usage rolled up into the lead's total. Config: `max_tokens`, `warn_threshold`, `hard_stop_threshold`.
- **Drop:** separate input/output caps, per-step action attribution.
- **Depends on:** slice 5 (subagents) and the hidden-message helpers from slice 4.

### Slice 12: PII redaction
- **DeerFlow:** no model call. Fixed regexes plus checksum validators (email, API key, credit
  card with Luhn, phone, national ID) replace matches with an HMAC-derived placeholder. The
  placeholder is stable for the same value but cannot be reversed without `token_secret`
  (16+ characters). It runs on real user messages before the model call and on remote-content
  tool results. Off by default; each detector has its own switch.
- **Keep:** the detectors, the HMAC placeholder, and the two boundaries (user input and tool output).
- **Drop:** redaction of durable context, MCP-specific scoping, national-ID variants.

### Slice 13: tool receipts + audit
- **DeerFlow:** `ToolReceiptMiddleware` stamps every tool result with metadata: tool name and
  call id, status, hashes of the arguments and output, byte count, timestamp. The output
  itself is not stored. Receipts get display ids `r1..rN`. Subagents are told to cite `[rN]`
  for every action claim, and the lead checks that each cited id exists and succeeded. The
  check is advisory: it proves the call happened, not that the claim is right.
- **Keep:** the stamp, the ledger built from message metadata, `[rN]` citations for subagent
  reports, and the advisory check.
- **Drop:** acceptance criteria, bash execution records, render modes other than `delegation_only`.
- **Depends on:** slice 5 (subagent reports) and slice 8 (so receipts survive restarts).
