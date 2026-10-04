from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dearflow.config.app_config import AppConfig
from dearflow.config.model_config import ModelConfig
from dearflow.config.tool_config import ToolConfig


def load_app_config(path: str | Path) -> AppConfig:
    """Load and validate DearFlow application configuration from YAML."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    default_model = _required_string(raw, "default_model")
    system_prompt = raw.get("system_prompt", "You are DearFlow, a helpful general-purpose agent.")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")

    raw_models = raw.get("models", {})
    if not isinstance(raw_models, dict):
        raise ValueError("models must be a mapping")

    models: dict[str, ModelConfig] = {}
    for name, value in raw_models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("model profile names must be non-empty strings")
        if not isinstance(value, dict):
            raise ValueError(f"model profile {name!r} must be a mapping")
        models[name] = ModelConfig(name=name, model=_required_string(value, "model", owner=f"model profile {name!r}"))

    raw_tools = raw.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be a list")
    tools: list[ToolConfig] = []
    for index, value in enumerate(raw_tools):
        owner = f"tool at index {index}"
        if not isinstance(value, dict):
            raise ValueError(f"{owner} must be a mapping")
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{owner} enabled must be a boolean")
        tools.append(
            ToolConfig(
                name=_required_string(value, "name", owner=owner),
                group=_required_string(value, "group", owner=owner),
                use=_required_string(value, "use", owner=owner),
                enabled=enabled,
            )
        )

    return AppConfig(
        default_model=default_model,
        system_prompt=system_prompt,
        models=models,
        tools=tools,
    )


def _required_string(values: dict[str, Any], key: str, *, owner: str = "configuration") -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} requires a non-empty {key}")
    return value.strip()
