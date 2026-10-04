# DearFlow 0.1

A compact, test-driven reconstruction of DeerFlow's lead-agent architecture. The
project will retain the model, tool, middleware, state, memory, skills, MCP,
subagent, persistence, and streaming concepts while intentionally excluding the
sandbox subsystem and unrestricted shell/filesystem tools.

## Current vertical slice

`assemble_lead_agent(...)` currently:

1. resolves the default or per-run model name;
2. creates the model through an injected adapter;
3. binds injected tools and middleware;
4. compiles a LangGraph agent with a DearFlow `ThreadState`; and
5. returns the graph and effective model as `LeadAgentAssembly`.

The compiled graph supports the complete model -> tools -> model loop.

Configured tools follow DeerFlow's `module.path:attribute` loading convention.
The example enables safe time and arithmetic tools; shell and filesystem tools
are intentionally absent because this project does not include a sandbox.

## Interactive CLI

Copy the example configuration and configure the Ilmu OpenAI-compatible endpoint:

```bash
cp config.example.yaml config.yaml
export OPENAI_API_KEY="your-ilmu-key"
export OPENAI_BASE_URL="https://api.ilmu.ai/v1"
uv run dearflow --config config.yaml
```

The configured model can be overridden by alias or direct model identifier:

```bash
uv run dearflow --config config.yaml --model reasoning
uv run dearflow --model "openai:ilmu-mini-v3.3" --thread-id my-thread
```

Inside the prompt, enter `exit` or `quit` to stop. Conversation checkpoints are
kept in memory for the lifetime of the CLI process.

## Development

```bash
uv sync --extra dev
uv run pytest -q
```

The public construction seam is:

```python
assemble_lead_agent(
    config,
    app_config=app_config,
    dependencies=dependencies,
) -> LeadAgentAssembly
```

## Next slices

- layered application/runtime configuration;
- built-in tool registry and tool authorization;
- ordered middleware assembly without sandbox middleware;
- prompt and skill catalog rendering;
- persistent checkpoints and long-term memory;
- MCP discovery and deferred tools;
- subagent delegation;
- streaming runtime and Gateway endpoints.
