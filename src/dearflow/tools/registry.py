"""Collects every tool the agent may use."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable

from langchain_core.tools import BaseTool

from dearflow.config import AppConfig, ToolConfig

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Loads tools declared in config and merges in any extra tools.

    Later slices add tool sources (skills, task delegation, MCP) by passing
    them as `extra_tools`; this class stays the single place that de-duplicates
    and filters them.
    """

    def __init__(self, app_config: AppConfig, extra_tools: Iterable[BaseTool] = ()) -> None:
        self._app_config = app_config
        self._extra_tools = list(extra_tools)

    def get_tools(self, groups: Iterable[str] | None = None) -> list[BaseTool]:
        """Return enabled tools, optionally limited to the named groups."""
        allowed = set(groups) if groups is not None else None
        candidates = [
            self._load(cfg)
            for cfg in self._app_config.tools
            if cfg.enabled and (allowed is None or cfg.group in allowed)
        ]
        # Extra tools are not group-filtered: they are added by code, not config.
        return self.deduplicate([*candidates, *self._extra_tools])

    @staticmethod
    def _load(tool_config: ToolConfig) -> BaseTool:
        module_name, attribute = tool_config.use.split(":", 1)
        loaded = getattr(importlib.import_module(module_name), attribute)
        if not isinstance(loaded, BaseTool):
            raise TypeError(f"{tool_config.use!r} did not resolve to a LangChain BaseTool")
        if loaded.name != tool_config.name:
            logger.warning(
                "Configured tool name %r differs from resolved name %r; using the resolved name",
                tool_config.name,
                loaded.name,
            )
        return loaded

    @staticmethod
    def deduplicate(tools: list[BaseTool]) -> list[BaseTool]:
        unique: dict[str, BaseTool] = {}
        for tool in tools:
            if tool.name in unique:
                logger.warning("Duplicate tool name %r skipped", tool.name)
                continue
            unique[tool.name] = tool
        return list(unique.values())
