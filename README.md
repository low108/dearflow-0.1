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
4. Context middleware (date, summarization, title)
5. Subagents (`task` tool)
6. Memory
7. MCP
8. Persistence + streaming
9. FastAPI gateway
