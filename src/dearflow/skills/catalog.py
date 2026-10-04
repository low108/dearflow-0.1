"""Discovers skills on disk and answers lookups."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from dearflow.skills.skill import Skill, SkillParseError

logger = logging.getLogger(__name__)


class SkillCatalog:
    """An in-memory index of skills, keyed by name."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in self._skills:
                logger.warning("Duplicate skill %r skipped (%s)", skill.name, skill.path)
                continue
            self._skills[skill.name] = skill

    @classmethod
    def discover(cls, root: str | Path) -> SkillCatalog:
        """Find every `SKILL.md` under `root`. Malformed ones are logged and skipped."""
        root_path = Path(root)
        skills: list[Skill] = []
        if root_path.is_dir():
            for skill_file in sorted(root_path.rglob("SKILL.md")):
                try:
                    skills.append(Skill.from_file(skill_file))
                except SkillParseError as exc:
                    logger.warning("Skipping invalid skill: %s", exc)
        return cls(skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def names(self) -> list[str]:
        return list(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def render_index(self) -> str:
        """One line per skill: cheap enough to always include in the prompt."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())
