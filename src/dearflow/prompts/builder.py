"""Assembles the system prompt from its parts."""

from __future__ import annotations

from dearflow.skills import SkillCatalog
from dearflow.subagents import SubagentRegistry

_SKILLS_SECTION = """\
<skills>
You have skills: reusable instructions for specific kinds of tasks.
When a task matches a skill's description, call the `load_skill` tool with its
name FIRST, then follow the loaded instructions.

Available skills:
{index}
</skills>"""


_CLARIFICATION_SECTION = """\
<clarification_system>
Workflow: CLARIFY, then PLAN, then ACT.
Before working, check whether you can do the task well. If required
information is missing, the request is ambiguous, there are several valid
approaches, or an action is risky or irreversible, call `ask_clarification`
FIRST. Ask one clear question, and do not call any other tool in that turn.
If the request is clear, do not ask: just do it.
</clarification_system>"""


_SUBAGENT_SECTION = """\
<subagent_system>
You can delegate with the `task` tool. A subagent works in its own fresh
context and returns only a final report. Delegating costs extra time and
tokens, so do the work yourself unless delegation clearly wins.

Delegate when:
- two or more INDEPENDENT jobs can run in parallel (launch all `task` calls in
  the same response), or
- one bounded investigation would flood your context with raw material you
  only need summarized.

Do NOT delegate when:
- the job is small, or one step depends on another's result (keep chains together),
- jobs touch the same files or other shared state,
- you need to ask the user something (subagents cannot).

Rules: the subagent knows nothing of this conversation, so write a complete,
self-contained prompt. At most {max_concurrent} `task` calls per response and
{max_total} per user message; extra calls are dropped. Afterwards, check the
reports and write the final answer yourself.{verification}

Subagent types:
{types}
</subagent_system>"""


_VERIFICATION = """

Subagent reports are self-reports, not verified facts. Each result ends with a `citations:` line
checking the receipt ids the subagent cited: "resolved" means those tool calls happened and
succeeded (not that the claims are right); "UNVERIFIED" means it described actions without
citing any. Before relying on a load-bearing claim, check the handle it gives (a path, a URL, an id)."""


class PromptBuilder:
    """Builds the system prompt.

    Each section is a small private method, so adding a section in a later
    slice (memory, date, ...) means adding one method and one line in `build()`.
    """

    def __init__(
        self,
        base_prompt: str,
        skill_catalog: SkillCatalog | None = None,
        *,
        subagents: SubagentRegistry | None = None,
        max_concurrent: int = 3,
        max_total: int = 6,
        receipts: bool = False,
    ) -> None:
        self._receipts = receipts
        self._base_prompt = base_prompt.strip()
        self._skills = skill_catalog or SkillCatalog()
        self._subagents = subagents  # None = delegation is off, no section
        self._max_concurrent = max_concurrent
        self._max_total = max_total

    def build(self) -> str:
        sections = [self._base_prompt, _CLARIFICATION_SECTION, self._skills_section(), self._subagent_section()]
        return "\n\n".join(section for section in sections if section)

    def _skills_section(self) -> str:
        if not self._skills:
            return ""
        return _SKILLS_SECTION.format(index=self._skills.render_index())

    def _subagent_section(self) -> str:
        if self._subagents is None:
            return ""
        return _SUBAGENT_SECTION.format(
            max_concurrent=self._max_concurrent,
            max_total=self._max_total,
            verification=_VERIFICATION if self._receipts else "",
            types=self._subagents.describe(),
        )
