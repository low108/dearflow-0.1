"""Loads tools from MCP servers."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from dearflow.config import McpConfig, McpServerConfig, expand_env

logger = logging.getLogger(__name__)


def build_connection(config: McpServerConfig) -> dict[str, Any]:
    """Config -> the connection dict `langchain-mcp-adapters` expects.

    `$VAR` references in command, args, env, url and headers are filled in from
    the environment here, so secrets stay out of the YAML file.
    """
    if config.type == "stdio":
        connection: dict[str, Any] = {
            "transport": "stdio",
            "command": expand_env(config.command or ""),
            "args": [expand_env(a) for a in config.args],
        }
        if config.env:
            connection["env"] = {k: expand_env(v) for k, v in config.env.items()}
        if config.cwd:
            connection["cwd"] = config.cwd
        return connection

    connection = {"transport": "streamable_http" if config.type == "http" else "sse", "url": expand_env(config.url or "")}
    if config.headers:
        connection["headers"] = {k: expand_env(v) for k, v in config.headers.items()}
    return connection


def run_sync(coro: Coroutine) -> Any:
    """Run a coroutine to completion from sync code, even inside a running loop."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def with_sync_support(tool: BaseTool) -> BaseTool:
    """MCP tools are async-only; this lets `graph.invoke` (sync) call them too."""
    if not isinstance(tool, StructuredTool) or tool.func is not None or tool.coroutine is None:
        return tool
    coroutine = tool.coroutine

    def call(**kwargs: Any) -> Any:
        return run_sync(coroutine(**kwargs))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=call,
        coroutine=coroutine,
        response_format=tool.response_format,
        metadata=tool.metadata,
    )


class McpToolLoader:
    """Connects to each enabled server once and keeps the tools it offered.

    Servers are loaded one by one and independently: a server that is down or
    misconfigured is logged and skipped, and the others still work. Each tool
    call opens its own short session (the adapters' default), which is simple
    and safe; DeerFlow keeps a pool of persistent stdio sessions for speed, at
    the cost of ~800 lines of lifecycle code.
    """

    def __init__(self, config: McpConfig) -> None:
        self._config = config
        self._tools: list[BaseTool] | None = None
        self._lock = threading.Lock()

    def tools(self) -> list[BaseTool]:
        """All tools from enabled servers. Loaded on first call, then cached."""
        if not self._config.enabled:
            return []
        with self._lock:
            if self._tools is None:
                self._tools = run_sync(self._load_all())
            return list(self._tools)

    async def _load_all(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for name, server in self._config.servers.items():
            if not server.enabled:
                continue
            try:
                loaded = await asyncio.wait_for(self._load_one(name, server), self._config.load_timeout_seconds)
            except Exception as exc:  # TimeoutError, bad command, bad env var, refused connection...
                logger.warning("MCP server %r skipped: %s: %s", name, type(exc).__name__, exc)
                continue
            logger.info("MCP server %r: %d tools", name, len(loaded))
            tools.extend(loaded)
        return tools

    @staticmethod
    async def _load_one(name: str, server: McpServerConfig) -> list[BaseTool]:
        client = MultiServerMCPClient({name: build_connection(server)}, tool_name_prefix=server.tool_name_prefix)
        return [with_sync_support(t) for t in await client.get_tools()]
