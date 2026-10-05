# Observability with LangSmith

DearFlow uses the LangSmith UI to inspect turns, model usage, tool calls and
middleware changes. This integration adds tracing; it does not add a local
dashboard or store snapshots in conversation history or the run event log.

Export these variables in the same shell that starts the CLI or Gateway:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY='YOUR_LANGSMITH_API_KEY'
export LANGSMITH_PROJECT=dearflow
# Optional: set the endpoint for your account's region, without a trailing slash.
# export LANGSMITH_ENDPOINT='https://eu.api.smith.langchain.com'
uv run dearflow --config config.yaml
# For the HTTP API, launch this instead:
# uv run dearflow-gateway --config config.yaml
```

DearFlow does not automatically load a `.env` file. Keep the API key outside
committed configuration. A key associated with multiple workspaces may also need
`LANGSMITH_WORKSPACE_ID`. See the [LangSmith setup guide](https://docs.langchain.com/langsmith/trace-with-langchain)
for account and regional endpoint settings.

Native LangChain tracing supplies model inputs and outputs, provider token usage,
tool arguments and results, timing, errors and nested agent runs. To inspect the
request passed onward by each middleware hook, add this to `config.yaml` and
restart the process:

```yaml
observability:
  middleware_snapshots: true
  capture_content: true
  max_snapshot_chars: 20000
```

`middleware_snapshots` and `capture_content` default to `false`; the size limit
defaults to `20000`. With snapshots enabled and content disabled, snapshots show
structure, types and string lengths. Enabling snapshots alone does not enable a
remote exporter. `LANGSMITH_TRACING` controls normal environment-based export.

Open your workspace in the [LangSmith UI](https://smith.langchain.com), select the
`dearflow` tracing project, then open a `DearFlow` trace to inspect its tree. Each
service turn has a root trace ID equal to its durable DearFlow `run_id`, with
`thread_id` and `dearflow_run_id` metadata inherited by child runs. The project's
Threads view groups turns using `thread_id`; see [thread configuration](https://docs.langchain.com/langsmith/threads).
Delegated agents appear as `subagent:<name>` beneath the `task` tool, with
`dearflow_agent` metadata identifying their type. Background memory extraction
runs outside the turn's context and can appear as separate traces.
Their token usage is additional to the service turn totals. A middleware hook
completing successfully does not prove that its background extraction produced
valid or useful facts; inspect the extraction response and saved memory too.

**In the trace tree, select Tree settings → Show All.** The Most relevant view
can hide wrapping middleware and its custom snapshot children, even though they
were exported successfully.

Custom spans are named `middleware.<Name>.<hook>`, including async hooks such as
`awrap_model_call`. They carry `dearflow_middleware` and `dearflow_hook` metadata.
On LangChain versions with native middleware spans, those native spans retain
their timing and point to the custom child span for snapshot details.

| Custom span field | Meaning |
| --- | --- |
| `inputs.before` | State or request as the hook receives it. |
| `outputs.calls[i].request` | Request when this hook invokes its downstream handler. |
| `outputs.calls[i].response` | Response immediately returned by that handler, before this hook edits it. |
| `outputs.calls[i].error_type` | Exception type when a downstream attempt fails and the hook handles it. |
| `outputs.result` | Final value returned by a wrapping hook. |
| `outputs.state_update` | Update returned by a before/after state hook. |
| `outputs.handler_calls` / `calls_truncated` | Total downstream attempts and whether some attempt snapshots were omitted. |

Each snapshot has a `value`, `content_captured`, `truncated` and
`redaction_applied` envelope. `redaction_applied` means the snapshot exporter
masked at least one recognized text value or secret field. It describes the
exporter, not a middleware action: with content capture enabled, a PII hook's
input may show `true` and its forwarded request `false` because the hook already
replaced the sensitive text. Both snapshots remain redacted.

Message snapshots include only DearFlow's receipt and receipt-ledger fields from
`additional_kwargs`; arbitrary provider metadata remains excluded. Compare a
receipt hook's downstream tool response with its final result to see the receipt
stamp appear even when the tool content is unchanged.

Compare
`before.value` with each forwarded `request.value` to attribute request changes
to one middleware. Compare a downstream `response.value` with `result.value` for
response edits. State hooks return deltas: `state_update` is not a reconstructed,
merged state. Hooks that short-circuit can have no downstream calls. The first
eight attempts are retained; `handler_calls` still counts all attempts. A failed
hook marks its span as an error and may have no output snapshots. Hook duration
includes time spent in downstream middleware, tools and model calls.

`max_snapshot_chars` bounds each serialized snapshot, not the entire trace.
Truncation and omitted binary content mean snapshots are diagnostic views, not
complete payload archives. Serialization failures produce an unavailable marker.

**The content setting applies only to custom snapshots.** Native LangSmith graph,
model and tool traces can include original, unredacted conversation and tool data.
Enabling remote tracing opts into exporting those traces. Custom snapshots redact
known secret fields and recognized text patterns before export, including when
captured before PII middleware runs. Regex redaction covers known formats; it is
not a guarantee that every sensitive value is removed.

Token totals belong to actual model calls. Provider-reported cache and reasoning
breakdowns may be visible; absent usage is unknown, not evidence of a free call.
DearFlow's existing local totals can show zero when a provider omits usage.
Middleware snapshots do not calculate token deltas or bills. For custom models,
check `ls_model_name` and `ls_provider`, then configure a matching LangSmith model
price entry before relying on cost totals. See [cost tracking](https://docs.langchain.com/langsmith/cost-tracking).

The implementation lives in `src/dearflow/observability/middleware.py` (hook
boundaries and callbacks) and `snapshots.py` (bounded serialization and redaction).
Factories instrument both lead and subagent middleware. Tests in
`tests/test_observability.py`, `tests/test_trace_snapshots.py` and
`tests/test_trace_integration.py` cover routing, retries, errors, cancellation,
redaction, usage and parent correlation. SDK export is tested with an offline
client in sync and async paths. The live demo script verifies exported payloads,
parent relationships and token totals using the configured account. Use
`--verify-only RESULT_FOLDER` to retry that comparison without new model calls.
Finding a trace in the UI is useful evidence, but is not a substitute for the
complete payload comparison. See the README for the live scenarios.
