"""The tool that lets the model read a skill's full instructions."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from dearflow.skills.catalog import SkillCatalog


def create_load_skill_tool(catalog: SkillCatalog) -> BaseTool:
    """Build a `load_skill` tool bound to a catalog.

    This is "progressive disclosure": the prompt only lists skill names and
    descriptions; the model calls this tool to pull in the full body of the
    one it actually needs. That keeps the context small.
    """

    def load_skill(name: str) -> str:
        """Load the full instructions of a skill by name. Call this before following a skill."""
        skill = catalog.get(name)
        if skill is None:
            available = ", ".join(catalog.names()) or "none"
            raise ValueError(f"Unknown skill {name!r}. Available skills: {available}")
        return f"# Skill: {skill.name}\n\n{skill.body}"

    return StructuredTool.from_function(load_skill)
