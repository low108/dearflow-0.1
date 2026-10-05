"""Public result returned after constructing a lead agent."""

from __future__ import annotations

from typing import Any


class LeadAgent:
    """The compiled graph and the runtime collaborators resolved for it."""

    def __init__(self, graph: Any, model_name: str, usage: Any) -> None:
        self.graph = graph
        self.model_name = model_name
        self.usage = usage
