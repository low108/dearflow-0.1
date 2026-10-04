"""A tiny real MCP server (stdio) used by the tests."""

import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def secret() -> str:
    """Return the DEMO_SECRET environment variable."""
    return os.environ.get("DEMO_SECRET", "unset")


if __name__ == "__main__":
    server.run("stdio")
