from __future__ import annotations

from collections.abc import Mapping

from dearflow.config.model_config import ModelConfig
from dearflow.config.tool_config import ToolConfig


class AppConfig:
    """Application settings required to construct the lead agent."""

    __slots__ = ("default_model", "system_prompt", "models", "tools")

    def __init__(
        self,
        default_model: str,
        system_prompt: str = "You are DearFlow, a helpful general-purpose agent.",
        models: Mapping[str, ModelConfig] | None = None,
        tools: list[ToolConfig] | None = None,
    ) -> None:
        self.default_model = default_model
        self.system_prompt = system_prompt
        self.models = dict(models or {})
        self.tools = list(tools or [])

    def resolve_model(self, requested_model: str | None = None) -> str:
        """Resolve a configured alias while still accepting direct model identifiers."""

        selected = requested_model or self.default_model
        profile = self.models.get(selected)
        return profile.model if profile is not None else selected
