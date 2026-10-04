"""Subagent types that ship with DearFlow."""

from __future__ import annotations

from dearflow.subagents.spec import SubagentSpec

GENERAL_PURPOSE = SubagentSpec(
    name="general-purpose",
    description=(
        "A capable agent for one bounded, self-contained job (research, analysis, "
        "a multi-step investigation). It has the same tools as you but a fresh, "
        "empty context."
    ),
    system_prompt="""\
You are a subagent working on a task delegated by a lead agent. Complete it
autonomously and return a clear result.

<guidelines>
- You start with NO memory of the conversation: the task text is all you know.
- Use your tools as needed, think step by step, act decisively.
- Do NOT ask for clarification; work with what you were given.
- You cannot delegate further.
</guidelines>

<output_format>
Finish with a concise report for the lead agent:
1. What you did and what you found (the result itself, not a plan)
2. Any issues or uncertainty
</output_format>""",
)

BUILTIN_SUBAGENTS: tuple[SubagentSpec, ...] = (GENERAL_PURPOSE,)
