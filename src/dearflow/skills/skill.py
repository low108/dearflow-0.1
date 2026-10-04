"""A single skill: a named, described block of instructions."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


class SkillParseError(ValueError):
    """Raised when a SKILL.md file is malformed."""


class Skill(BaseModel):
    """A skill loaded from a `SKILL.md` file.

    File format:

        ---
        name: code-review
        description: When to use this skill (shown to the model up front).
        ---
        Full instructions in Markdown (loaded only on demand).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = Field(min_length=1)
    body: str
    path: Path

    @field_validator("name")
    @classmethod
    def _kebab_case(cls, value: str) -> str:
        if not _NAME_PATTERN.match(value):
            raise ValueError(f"skill name must be kebab-case, got {value!r}")
        return value

    @classmethod
    def from_file(cls, path: Path) -> Skill:
        """Parse one SKILL.md file."""
        text = path.read_text(encoding="utf-8")
        match = _FRONTMATTER.match(text)
        if match is None:
            raise SkillParseError(f"{path}: missing '---' frontmatter block")
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise SkillParseError(f"{path}: invalid frontmatter YAML: {exc}") from exc
        if not isinstance(meta, dict):
            raise SkillParseError(f"{path}: frontmatter must be a mapping")
        try:
            return cls(
                name=meta.get("name", ""),
                description=str(meta.get("description", "")).strip(),
                body=match.group(2).strip(),
                path=path,
            )
        except ValueError as exc:
            raise SkillParseError(f"{path}: {exc}") from exc
