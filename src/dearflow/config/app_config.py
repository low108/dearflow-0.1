"""Typed application configuration.

Everything the rest of the code needs to know about the user's setup lives
here. Later slices add their own sections (memory, subagents, ...) as new
models on `AppConfig`, so this file is the one place to read for "what can be
configured?".
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_SYSTEM_PROMPT = "You are DearFlow, a helpful general-purpose agent."


class _Config(BaseModel):
    """Base class: reject unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Config):
    """One named model profile, e.g. `fast -> openai:ilmu-mini-v3.3`."""

    model: str = Field(min_length=1)


class ToolConfig(_Config):
    """One tool to load, referenced as `module.path:attribute`."""

    name: str = Field(min_length=1)
    group: str = Field(min_length=1)
    use: str = Field(min_length=1)
    enabled: bool = True

    @field_validator("use")
    @classmethod
    def _must_be_module_attribute(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError(f"must use module.path:attribute syntax, got {value!r}")
        return value


class SkillsConfig(_Config):
    """Where to find skills (folders containing a SKILL.md)."""

    enabled: bool = True
    path: str = "skills"


class SummarizationConfig(_Config):
    """Compress old messages once the conversation gets long."""

    enabled: bool = True
    trigger_tokens: int = Field(default=100_000, gt=0)  # summarize above this many (approx.) tokens
    keep_messages: int = Field(default=20, gt=0)  # recent messages kept verbatim
    model: str | None = None  # alias or id; None = the run's own model


class TitleConfig(_Config):
    """Conversation title. Without `use_model` the title is the first message, shortened."""

    enabled: bool = True
    use_model: bool = False
    max_words: int = Field(default=6, gt=0)


class CustomSubagentConfig(_Config):
    """A subagent type defined in YAML. Unset fields fall back to `SubagentsConfig`."""

    description: str = Field(min_length=1)  # shown to the lead agent
    system_prompt: str = Field(min_length=1)
    tools: list[str] | None = None  # allowlist; None = all of the lead agent's tools
    disallowed_tools: list[str] | None = None  # None = ["task", "ask_clarification"]
    model: str = "inherit"  # alias/id, or "inherit" = the lead agent's model
    max_turns: int | None = Field(default=None, gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)


class SubagentsConfig(_Config):
    """Delegation through the `task` tool. Off by default; see `--subagents`."""

    enabled: bool = False
    max_concurrent: int = Field(default=3, ge=1, le=16)  # task calls per model response
    max_total: int = Field(default=6, ge=1, le=50)  # task calls per user message
    max_turns: int = Field(default=25, gt=0)
    timeout_seconds: int = Field(default=300, gt=0)
    custom_agents: dict[str, CustomSubagentConfig] = Field(default_factory=dict)


class MemoryConfig(_Config):
    """Long-term memory: facts about the user, learned in the background."""

    enabled: bool = False
    storage_path: str = "memory"  # one folder per user inside
    injection_enabled: bool = True  # put known facts in front of each new message
    debounce_seconds: float = Field(default=10, ge=0)  # wait for quiet before extracting
    max_facts: int = Field(default=100, ge=1)
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)  # drop less certain facts
    max_injection_tokens: int = Field(default=1500, gt=0)
    model: str | None = None  # alias for the extraction model; None = default_model


_ENV_REFERENCE = re.compile(r"\$(\w+)|\$\{(\w+)\}")


def expand_env(value: str) -> str:
    """Replace `$NAME` / `${NAME}` with the environment variable. Unset -> error,
    so a missing API key fails at startup, not as a mysterious 401 later."""

    def lookup(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        if name not in os.environ:
            raise ValueError(f"environment variable {name} is not set")
        return os.environ[name]

    return _ENV_REFERENCE.sub(lookup, value)


class McpServerConfig(_Config):
    """One MCP server (an external process or URL that offers tools)."""

    type: Literal["stdio", "http", "sse"] = "stdio"
    enabled: bool = True
    command: str | None = None  # stdio
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None  # http / sse
    headers: dict[str, str] = Field(default_factory=dict)
    tool_name_prefix: bool = True  # `github` server's `search` tool becomes `github_search`
    description: str = ""

    @model_validator(mode="after")
    def _needs_its_transport_fields(self) -> McpServerConfig:
        if self.type == "stdio" and not self.command:
            raise ValueError("stdio servers need `command`")
        if self.type != "stdio" and not self.url:
            raise ValueError(f"{self.type} servers need `url`")
        return self


class McpConfig(_Config):
    """Tools from external MCP servers. Off by default."""

    enabled: bool = False
    load_timeout_seconds: float = Field(default=60, gt=0)  # per server, at startup
    servers: dict[str, McpServerConfig] = Field(default_factory=dict)


class PersistenceConfig(_Config):
    """Where conversations are stored. Off = everything is forgotten on exit."""

    enabled: bool = False
    path: str = "data/dearflow.db"  # SQLite file: checkpoints + thread/run records


class GatewayConfig(_Config):
    """The HTTP API (`dearflow-gateway`)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8001, gt=0, lt=65536)
    api_key: str | None = None  # if set, requests need `Authorization: Bearer <key>`; `$ENV_VAR` allowed
    cors_origins: list[str] = Field(default_factory=list)
    heartbeat_seconds: float = Field(default=15, gt=0)  # idle keep-alive on SSE streams
    retain_seconds: float = Field(default=120, ge=0)  # how long a finished run stays joinable in memory


class ObservabilityConfig(_Config):
    """Extra middleware traces; LangSmith export is controlled by its environment variables."""

    middleware_snapshots: bool = False
    capture_content: bool = False
    max_snapshot_chars: int = Field(default=20_000, ge=256, le=1_000_000)


class TokenUsageConfig(_Config):
    """Count tokens per run and save the totals with the run record."""

    enabled: bool = True


class TokenBudgetConfig(_Config):
    """A per-run token limit (same fields as DeerFlow's). Off by default."""

    enabled: bool = False
    max_tokens: int = Field(default=200_000, ge=1000)  # input + output, all agents in the run
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    warn_threshold: float = Field(default=0.8, ge=0, le=1)  # fraction at which the model is told to wrap up
    hard_stop_threshold: float = Field(default=1.0, ge=0, le=1)  # fraction at which tool calls are removed

    @model_validator(mode="after")
    def _hard_stop_not_before_warning(self) -> TokenBudgetConfig:
        if self.hard_stop_threshold < self.warn_threshold:
            raise ValueError("hard_stop_threshold must be >= warn_threshold")
        return self


class LoopDetectionConfig(_Config):
    """Stop a model from repeating the same tool call indefinitely."""

    enabled: bool = True
    warn_threshold: int = Field(default=3, ge=2)
    hard_limit: int = Field(default=5, ge=3)
    window_size: int = Field(default=20, ge=3)

    @model_validator(mode="after")
    def _hard_limit_after_warning(self) -> LoopDetectionConfig:
        if self.hard_limit <= self.warn_threshold:
            raise ValueError("hard_limit must be greater than warn_threshold")
        if self.window_size < self.hard_limit:
            raise ValueError("window_size must be >= hard_limit")
        return self


class PiiRedactionConfig(_Config):
    """Hide personal data from the model (same options as DeerFlow's). Off by default.

    Covers what the model is shown (user messages, results of outside-content tools),
    the memory queue, titles and summaries. Stored conversations keep the original text.
    """

    enabled: bool = False
    redact_email: bool = True
    redact_api_key: bool = True  # OpenAI sk-, AWS AKIA, GitHub ghp_/github_pat_, Slack xox-, Google AIza
    redact_credit_card: bool = True  # only numbers that pass the Luhn checksum
    redact_phone: bool = True  # +CC international, CN mobile, US formatted
    redact_national_id: bool = True  # CN resident ID and CPF (checksummed), CUIT and RFC (format)
    token_secret: str | None = None  # HMAC key for placeholders, 16+ chars; `$ENV_VAR` allowed
    tool_names: list[str] = Field(default_factory=list)  # extra tools whose results come from outside (MCP tools always do)

    @model_validator(mode="after")
    def _secret_required_when_enabled(self) -> PiiRedactionConfig:
        if self.enabled and not (self.token_secret or "").strip():
            raise ValueError(
                "pii_redaction.enabled needs token_secret: without a secret key, placeholders would be "
                "fingerprints anyone could compute from a guessed value"
            )
        return self


class VerificationConfig(_Config):
    """Tool receipts and the check of subagent reports (same names as DeerFlow's `verification`)."""

    receipts_enabled: bool = True  # stamp a receipt on every tool result
    # Lead agent: show the receipt list to the model on every call ("always"), or only while it
    # is reading a subagent's result ("delegation_only"). Subagents always see it: that is where
    # citations are written.
    receipts_render_mode: Literal["always", "delegation_only"] = "delegation_only"


class ContextConfig(_Config):
    """Middleware that manages what the model sees each turn."""

    timezone: str | None = None  # IANA name such as "Asia/Kuala_Lumpur"; None = machine zone
    date_reminder: bool = True
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    title: TitleConfig = Field(default_factory=TitleConfig)


class AppConfig(_Config):
    """Root configuration object, usually loaded from `config.yaml`."""

    default_model: str = Field(min_length=1)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, min_length=1)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    tools: list[ToolConfig] = Field(default_factory=list)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    pii_redaction: PiiRedactionConfig = Field(default_factory=PiiRedactionConfig)
    token_usage: TokenUsageConfig = Field(default_factory=TokenUsageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    loop_detection: LoopDetectionConfig = Field(default_factory=LoopDetectionConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    plan_mode: bool = False  # enables write_todos; override per run with configurable.is_plan_mode

    @model_validator(mode="after")
    def _budget_needs_usage_counting(self) -> AppConfig:
        if self.token_budget.enabled and not self.token_usage.enabled:
            raise ValueError("token_budget needs token_usage.enabled: the budget is measured from the usage counter")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load and validate a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.model_validate(raw)

    def resolve_model(self, requested: str | None = None) -> str:
        """Turn an alias (`fast`) into a model id; pass unknown names through."""
        selected = requested or self.default_model
        profile = self.models.get(selected)
        return profile.model if profile is not None else selected
