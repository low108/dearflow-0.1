"""The tool that lets the model read a skill's full instructions."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from dearflow.skills.catalog import SkillCatalog


def create_load_skill_tool(catalog: SkillCatalog) -> BaseTool:
    """Build a `load_skill` tool bound to a catalog.

    This is "progressive disclosure": the prompt only lists skill names and
    descriptions; the model calls this tool to pull in the full body of the
    one it actually needs. That keeps the context small.
    """

    def load_skill(
        name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict | None, InjectedState] = None,
    ) -> Command:
        """Load the full instructions of a skill by name. Call this before following a skill."""
        skill = catalog.get(name)
        if skill is None:
            available = ", ".join(catalog.names()) or "none"
            raise ValueError(f"Unknown skill {name!r}. Available skills: {available}")
        message = ToolMessage(
            content=f"# Skill: {skill.name}\n\n{skill.body}",
            tool_call_id=tool_call_id,
            name="load_skill",
        )
        return Command(
            update={
                "messages": [message],
                "skill_context": [
                    {
                        "name": skill.name,
                        "path": str(skill.path),
                        "description": skill.description,
                        "loaded_at": len((state or {}).get("messages", [])),
                    }
                ],
            }
        )

    return StructuredTool.from_function(load_skill)
