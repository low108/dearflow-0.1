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
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    plan_mode: bool = False  # enables write_todos; override per run with configurable.is_plan_mode

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
