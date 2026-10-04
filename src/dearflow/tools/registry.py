from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable

from langchain_core.tools import BaseTool

from dearflow.config import AppConfig

logger = logging.getLogger(__name__)


def get_available_tools(
    app_config: AppConfig,
    *,
    groups: Iterable[str] | None = None,
) -> list[BaseTool]:
    """Load enabled configured tools, optionally restricted to named groups."""

    allowed_groups = set(groups) if groups is not None else None
    configured = [
        tool_config
        for tool_config in app_config.tools
        if tool_config.enabled and (allowed_groups is None or tool_config.group in allowed_groups)
    ]

    unique_tools: list[BaseTool] = []
    seen_names: set[str] = set()
    for tool_config in configured:
        loaded = _resolve_tool(tool_config.use)
        if loaded.name != tool_config.name:
            logger.warning(
                "Configured tool name %r differs from resolved name %r; using the resolved name",
                tool_config.name,
                loaded.name,
            )
        if loaded.name in seen_names:
            logger.warning("Duplicate tool name %r skipped", loaded.name)
            continue
        unique_tools.append(loaded)
        seen_names.add(loaded.name)
    return unique_tools


def _resolve_tool(reference: str) -> BaseTool:
    try:
        module_name, attribute_name = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Tool reference must use module.path:attribute syntax: {reference!r}") from exc

    module = importlib.import_module(module_name)
    loaded = getattr(module, attribute_name)
    if not isinstance(loaded, BaseTool):
        raise TypeError(f"Configured tool {reference!r} did not resolve to a LangChain BaseTool")
    return loaded
