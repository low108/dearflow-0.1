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
8. Persistence: SQLite checkpointer + thread/run tables
9. Streaming: stream modes, run events
10. FastAPI gateway (SSE endpoint)
11. Token usage + token budget
12. PII redaction
13. Tool receipts + audit

Slices 11-13 are the hardening DeerFlow adds around the core. Each is one middleware
plus a small config section. Design notes are in `docs/deerflow-comparison.md`.
