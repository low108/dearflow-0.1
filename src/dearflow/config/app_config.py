"""Typed application configuration.

Everything the rest of the code needs to know about the user's setup lives
here. Later slices add their own sections (memory, subagents, ...) as new
models on `AppConfig`, so this file is the one place to read for "what can be
configured?".
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
