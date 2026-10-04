"""What a subagent type is."""

from __future__ import annotations

from dataclasses import dataclass

INHERIT = "inherit"
TASK = "task"  # name of the delegation tool

# A subagent must never delegate again or talk to the user.
DEFAULT_DISALLOWED = ("task", "ask_clarification")


@dataclass(frozen=True)
class SubagentSpec:
    """A recipe for one kind of subagent (mirrors DeerFlow's `SubagentConfig`).

    `description` is shown to the LEAD agent so it can choose a type.
    `system_prompt` is the subagent's own instructions.
    `tools=None` means "every tool the lead agent has", minus `disallowed_tools`.
    `model="inherit"` means "same model as the lead agent".
    """

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = DEFAULT_DISALLOWED
    model: str = INHERIT
    max_turns: int = 25
    timeout_seconds: int = 300
