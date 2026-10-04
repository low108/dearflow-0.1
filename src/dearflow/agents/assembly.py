from __future__ import annotations

from typing import Any


class LeadAgentAssembly:
    """Result of constructing a lead agent."""

    __slots__ = ("graph", "effective_model")

    def __init__(self, graph: Any, effective_model: str) -> None:
        self.graph = graph
        self.effective_model = effective_model
