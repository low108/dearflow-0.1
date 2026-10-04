# DearFlow

A compact, readable, object-oriented reconstruction of DeerFlow's lead-agent
architecture on LangChain/LangGraph. It keeps the model, tool, middleware,
skills, memory, MCP, subagent, persistence and streaming concepts and
intentionally excludes the sandbox (no shell, no virtual filesystem).

## How it fits together

```
config.yaml ──► AppConfig (pydantic)
                   │
   ModelFactory ───┤
   ToolRegistry ───┼──► LeadAgentFactory.build(run_config) ──► LeadAgent(graph, model_name)
   MiddlewareChain ┘            │
                                └─ create_agent(model, tools, middleware, ThreadState)
```

| Class | Responsibility |
|---|---|
| `AppConfig` | typed, validated settings; resolves model aliases |
| `ModelFactory` | model id -> chat model (single provider) |
| `ToolRegistry` | loads config tools + extra tools, filters by group, de-duplicates |
| `MiddlewareChain` | the ordered middleware pipeline (`default()` shows it all) |
| `Skill` / `SkillCatalog` | parse `skills/**/SKILL.md` (frontmatter + body) and index them by name |
| `create_load_skill_tool` | `load_skill(name)` tool: the model pulls a skill's full text on demand |
| `PromptBuilder` | builds the system prompt; lists skill names + descriptions only |
| `ask_clarification` + `ClarificationMiddleware` | model asks you a question; the run ends and your reply starts the next turn |
| `TodoMiddleware` | plan mode: `write_todos` task list + guard against finishing with open todos |
| `DynamicContextMiddleware` | injects the current date as a hidden message (keeps the system prompt cacheable) |
| `SummarizationMiddleware` | LangChain's; compresses old messages past a token threshold |
| `TitleMiddleware` | sets `state["title"]` after the first exchange |
| `SubagentSpec` / `SubagentRegistry` | a subagent type (prompt, tools, model, limits); built-ins + `subagents.custom_agents` |
| `SubagentExecutor` | builds a fresh agent per task, runs it, returns its last message (turn + time limits) |
| `task` tool | the lead agent delegates: `task(prompt, subagent_type)`; several calls per response run in parallel |
| `SubagentLimitMiddleware` | enforces max calls per response / per user message in code |
| `MemoryStore` | facts per user in `memory/<user_id>/memory.json`: dedupe, cap, token-budgeted rendering |
| `FactExtractor` | asks a model for facts (JSON), tolerates fences / `<think>` / bad items |
| `MemoryManager` | debounced background updates: a burst of turns costs one extraction |
| `MemoryMiddleware` | before a run: hidden `<memory>` message; after it: queue the exchange |
| `McpServerConfig` / `McpToolLoader` | connects to MCP servers (stdio/http/sse), one at a time; a broken server is skipped. Tools are prefixed `server_tool` |
| `Persistence` | the storage bundle: SQLite checkpointer (messages) + `ThreadStore` + `RunStore` (metadata); in-memory when off |
| `RunService` | one run: claim the thread, record it, invoke the agent, record the outcome; rejects a second run on a busy thread |
| `RunService.stream` | yields `StreamEvent`s while the agent runs: `metadata`, your chosen modes, `end`; abandoning it marks the run interrupted |
| `normalize_modes` | allowlist of stream modes (`values`, `updates`, `messages`, `custom`); unknown modes are an error |
| `RunEventStore` | the run log (`updates`, `custom`, errors, start/end) for replay |
| `create_app` (gateway) | FastAPI app with DeerFlow's routes: `/api/threads`, `/runs`, `/runs/stream`, `/runs/wait`, `/join`, `/cancel`, `/events` |
| `RunManager` / `ActiveRun` | runs agents in background threads; buffers events so many clients can watch, join late, or cancel |
| `UsageCallback` / `RunUsage` | counts tokens of one run across the lead agent, subagents and middleware; saved on the run record |
| `TokenBudgetMiddleware` | soft warning (shown to the next model call only), then hard stop (tool calls removed); shared with subagents |
| `Redactor` / `make_redactor` | regex detectors with checksums (Luhn, CN ID, CPF) and HMAC placeholders `[EMAIL_abc...]`; no model call |
| `PiiRedactionMiddleware` | redacts what the model is shown: user messages and results of outside tools (MCP). Memory queue, titles and summaries use the same redactor |
| `ToolReceiptMiddleware` | outermost layer: stamps a receipt (name, status, argument/output hashes) on every tool result; shows the receipt list to the model and stamps that list on its reply |
| `verify_receipt_citations` | checks a subagent report's `[rN]` citations against its receipts; the lead gets a `citations:` line (advisory, never "verified") |
| `LeadAgentFactory` | the only composition root; builds the graph |
| `ToolErrorHandlingMiddleware` | tool exceptions become messages the model can react to |

Every later feature plugs in through one of those seams: a new tool goes into
`ToolRegistry`, new behaviour is a middleware in `MiddlewareChain`, new
settings are a new section on `AppConfig`.

## Skills

A skill is a folder with a `SKILL.md`:

```markdown
---
name: code-review
description: Use when the user asks you to review code.
---
Full step-by-step instructions...
```

Only `name: description` lines go into the system prompt. When a task matches,
the model calls `load_skill("code-review")` and receives the full body
(progressive disclosure keeps the context small). See `skills/public/`.

## Subagents

Turn on with `subagents.enabled: true` or `dearflow --subagents`. The lead agent then
sees a `task` tool. A subagent gets only the task text, its own system prompt and a
filtered copy of the lead agent's tools (never `task`, never `ask_clarification`), and
returns only its final message, so the lead's context stays small. How this differs
from real DeerFlow is in `docs/deerflow-comparison.md`.

## Persistence

`dearflow --persist --thread-id trip` saves the conversation to `data/dearflow.db`;
run the same command tomorrow and it continues, and `dearflow --list-threads` shows
what you have. Two layers, like DeerFlow: LangGraph's checkpointer stores the messages;
the `threads` table (owner, title, idle/busy) and `runs` table (status, error, preview)
describe them. A crash leaves no stuck state: at startup, runs still marked `running`
become `interrupted` and their threads are freed. Only one run per thread at a time.

## Streaming

The CLI prints answers as they are produced, with status lines for tool calls (`→ task`)
and subagents (`⇢ subagent general-purpose: look up`); `--no-stream` waits for the full answer.
In code:

```python
for event in service.stream("hi", thread_id="t1", modes=["messages", "custom"]):
    print(event.event, event.data)      # metadata ... messages ... custom ... end
    # event.sse() is the same event as a Server-Sent Events frame
```

Modes: `values` (full state each step), `updates` (what each node changed), `messages`
(model output, as `{"message", "node"}`), `custom` (tool progress such as subagent start/end).
Hidden middleware messages (date, memory) are not sent in `messages`. Every run also writes a
log (`service.events(run_id)`) with `updates`, `custom`, errors and start/end, but not tokens or
full snapshots.

## HTTP API

`uv run dearflow-gateway --config config.yaml` (docs at `/docs`). Routes follow DeerFlow's gateway:

| Route | What it does |
|---|---|
| `POST /api/threads`, `POST /api/threads/search` | create / list your threads |
| `GET` / `DELETE /api/threads/{id}` | read (with state values) / delete |
| `GET /api/threads/{id}/state`, `/messages` | current state / visible conversation |
| `POST /api/threads/{id}/runs` | start a run in the background, return it at once |
| `POST /api/threads/{id}/runs/stream` | start and stream SSE (`Content-Location: .../runs/{run_id}`) |
| `POST /api/threads/{id}/runs/wait` | start, block, return the final state |
| `GET /api/threads/{id}/runs`, `/runs/{run_id}` | run records |
| `GET /api/threads/{id}/runs/{run_id}/join` | watch a run (from its start; after it ends, replays the run log) |
| `POST /api/threads/{id}/runs/{run_id}/cancel` | stop at the next event (`202`, or `204` with `?wait=true`) |
| `GET /api/threads/{id}/runs/{run_id}/events` | run log (`event_types`, `after_seq`, `limit`) |

```bash
curl -N -X POST localhost:8001/api/threads/t1/runs/stream -H 'content-type: application/json' \
  -d '{"input": {"messages": [{"role": "user", "content": "hi"}]}, "stream_mode": ["messages"]}'
```

Body fields are DeerFlow's (`input`, `context`, `stream_mode`, `on_disconnect`, `multitask_strategy`);
unknown fields and unsupported options are `422` errors. `context` may set `model_name`,
`is_plan_mode`, `subagent_enabled`. A second run on a busy thread is `409`.
`on_disconnect: "cancel"` (default) stops the run when the streaming client leaves; `"continue"`
keeps it going; `join` observers never cancel it. With `gateway.api_key` set, send
`Authorization: Bearer <key>`; the `X-User-Id` header then names the user, and other users'
threads answer `404`.

## Tool receipts

Every tool result is stamped with a receipt by code, not by the model: tool name, status, and hashes of the
arguments and the output (the output itself is never stored). The layer is outermost, so calls that failed or
were rewritten by inner layers are recorded too, and a tool cannot write its own receipt. Subagents are shown
their receipts (`[r1] lookup status=success ...`) and asked to cite them in their final report
(`I looked it up [r1 lookup]`). When the subagent finishes, the `task` tool checks every citation against the
receipts that subagent was shown, and appends one line for the lead agent:

```
citations: 2 resolved - execution evidence only, does not validate claim correctness
citations: 1 resolved, 1 unknown - ...          (it cited a receipt that does not exist)
citations: UNVERIFIED - action claims without receipt citations      ("I created the file ..." with no citation)
```

"Resolved" proves the call happened and succeeded, not that the sentence next to it is true. The receipts and
the check are saved with the conversation: `GET /api/threads/{id}/receipts` is the audit trail, and tool
messages in `/messages` and `updates` events carry their `receipt`.

## PII redaction

With `pii_redaction.enabled`, the model never sees emails, API keys, card numbers, phone numbers or
national IDs. They become placeholders such as `[EMAIL_kdjwhqmnepsjzhxuwqgtbvacirl]`. A placeholder is an HMAC of
the value under your `token_secret`: the same address always gets the same token (so "write to ann@..." then
"and cc ann@..." still refers to one person) but nobody can compute a token from a guessed value, or go back from a
token to the value. Numbers are only redacted if they pass their checksum, so order numbers are left alone.
Redaction happens where outside text enters: user messages (at each model call), results of MCP tools and tools
listed in `pii_redaction.tool_names`, the memory queue, titles, and the input of summarization. Subagents are
covered too. **The stored conversation keeps the original** (so the user sees what they wrote; run previews are also
raw), and the model's answers can contain placeholders, which are not turned back. If redaction itself fails, the
text goes through and a warning is logged.

## Tokens

Every run records its tokens (`runs` table, `RunResponse`, the `end` event, and the CLI's `tokens:` line),
split into `lead_agent`, `subagent` and `middleware` (title/summary) and by model;
`GET /api/threads/{id}/token-usage` sums them per thread. Turn on a limit with `token_budget.enabled`:
at 80% the model gets a one-time "wrap up" notice (only in its next request, never stored in the
conversation); at 100% its tool calls are removed, so it has to answer with what it has, and the run is
recorded with `stop_reason: "token_capped"`. Nothing is raised. Subagents count toward the same budget.
The budget needs `RunService` (CLI, gateway) because that is what counts tokens; a bare `graph.invoke` is not limited.

## MCP

MCP servers are external programs or URLs that offer tools (GitHub, databases, search).
List them under `mcp.servers` and set `mcp.enabled: true`. Their tools join the
lead agent's tool list, and subagents inherit them like any other tool. Secrets stay
out of the YAML with `$ENV_VAR`. A missing variable or an unreachable server skips
that server with a warning; the rest still load. Each tool call opens a short session,
which is simple and safe (DeerFlow pools persistent sessions for speed).

## Memory

`dearflow --memory --user-id ann`. After each turn the user's message and the final
reply are queued; after a quiet period one model call extracts lasting facts
(`"Prefers concise answers"`) into `memory/ann/memory.json`. At the start of each
new message the known facts are added as a hidden `<memory>` block (not in the
system prompt, which keeps it cacheable). Tool calls and hidden messages are never
sent to the extractor. Memory failures are logged, never shown to the chat.

## Run

```bash
cp config.example.yaml config.yaml
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.ilmu.ai/v1"
uv run dearflow --config config.yaml
uv run dearflow --model reasoning --thread-id my-thread
```

## Develop

```bash
uv sync --extra dev
uv run pytest -q
```

## Roadmap

1. ✅ Base: typed config, factories, middleware chain
2. ✅ Prompt + skills
3. ✅ Core tools (clarification, todos; web search deferred)
4. ✅ Context middleware (date, summarization, title)
5. ✅ Subagents (`task` tool)
6. ✅ Memory
7. ✅ MCP
8. ✅ Persistence: SQLite checkpointer + thread/run tables
9. ✅ Streaming: stream modes, run events
10. ✅ FastAPI gateway (SSE endpoint)
11. ✅ Token usage + token budget
12. ✅ PII redaction
13. ✅ Tool receipts + audit

All 13 slices are built. Slices 11-13 are the hardening DeerFlow adds around the core; each is one
middleware plus a small config section. How every slice differs from DeerFlow is in
`docs/deerflow-comparison.md`.
