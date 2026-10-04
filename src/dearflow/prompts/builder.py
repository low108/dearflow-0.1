"""Assembles the system prompt from its parts."""

from __future__ import annotations

from dearflow.skills import SkillCatalog

_SKILLS_SECTION = """\
<skills>
You have skills: reusable instructions for specific kinds of tasks.
When a task matches a skill's description, call the `load_skill` tool with its
name FIRST, then follow the loaded instructions.

Available skills:
{index}
</skills>"""


class PromptBuilder:
    """Builds the system prompt.

    Each section is a small private method, so adding a section in a later
    slice (memory, date, ...) means adding one method and one line in `build()`.
    """

    def __init__(self, base_prompt: str, skill_catalog: SkillCatalog | None = None) -> None:
        self._base_prompt = base_prompt.strip()
        self._skills = skill_catalog or SkillCatalog()

    def build(self) -> str:
        sections = [self._base_prompt, self._skills_section()]
        return "\n\n".join(section for section in sections if section)

    def _skills_section(self) -> str:
        if not self._skills:
            return ""
        return _SKILLS_SECTION.format(index=self._skills.render_index())
