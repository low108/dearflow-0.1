"""Looks up subagent types by name."""

from __future__ import annotations

from collections.abc import Iterable

from dearflow.config import SubagentsConfig
from dearflow.subagents.builtins import BUILTIN_SUBAGENTS
from dearflow.subagents.spec import DEFAULT_DISALLOWED, SubagentSpec


class UnknownSubagentError(KeyError):
    """Raised for an unknown type; the message lists the valid ones."""

    def __str__(self) -> str:  # KeyError would otherwise wrap the message in quotes
        return str(self.args[0])


class SubagentRegistry:
    """Built-in types plus custom ones from config. Custom names win on clash."""

    def __init__(self, custom: Iterable[SubagentSpec] = ()) -> None:
        self._specs: dict[str, SubagentSpec] = {spec.name: spec for spec in (*BUILTIN_SUBAGENTS, *custom)}

    @classmethod
    def from_config(cls, config: SubagentsConfig) -> SubagentRegistry:
        """Builtins plus `subagents.custom_agents`, with global defaults filled in."""
        custom = [
            SubagentSpec(
                name=name,
                description=agent.description,
                system_prompt=agent.system_prompt,
                tools=tuple(agent.tools) if agent.tools is not None else None,
                disallowed_tools=(
                    tuple(agent.disallowed_tools) if agent.disallowed_tools is not None else DEFAULT_DISALLOWED
                ),
                model=agent.model,
                max_turns=agent.max_turns or config.max_turns,
                timeout_seconds=agent.timeout_seconds or config.timeout_seconds,
            )
            for name, agent in config.custom_agents.items()
        ]
        return cls(custom)

    def names(self) -> list[str]:
        return list(self._specs)

    def get(self, name: str) -> SubagentSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownSubagentError(
                f"Unknown subagent type {name!r}. Available types: {', '.join(self._specs)}"
            ) from None

    def describe(self) -> str:
        """One line per type; shown in the task tool description and the prompt."""
        return "\n".join(f"- {spec.name}: {' '.join(spec.description.split())}" for spec in self._specs.values())
