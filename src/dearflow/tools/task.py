"""The `task` tool: how the lead agent delegates work to a subagent."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from dearflow.subagents import SubagentExecutor, SubagentRegistry, UnknownSubagentError
from dearflow.subagents.spec import TASK

_DESCRIPTION = """\
Delegate a bounded task to a subagent that works in its own fresh context and
returns only its final report.

The subagent knows NOTHING about this conversation: put everything it needs in
`prompt` (goal, relevant facts, expected output).

Subagent types:
{types}

Several `task` calls in one response run in parallel, so only launch them
together when the jobs are independent."""


class TaskInput(BaseModel):
    prompt: str = Field(description="Complete, self-contained instructions for the subagent.")
    subagent_type: str = Field(description="Which subagent type to use, e.g. 'general-purpose'.")
    description: str = Field(default="", description="A 3-5 word label for the job (for logs and UIs).")


def create_task_tool(executor: SubagentExecutor, registry: SubagentRegistry) -> BaseTool:
    """Build the `task` tool. Both sync and async callers are supported."""

    def task(prompt: str, subagent_type: str, description: str = "") -> str:
        try:
            spec = registry.get(subagent_type)
        except UnknownSubagentError as exc:
            return f"Error: {exc}"
        return executor.run(spec, prompt).render()

    async def atask(prompt: str, subagent_type: str, description: str = "") -> str:
        try:
            spec = registry.get(subagent_type)
        except UnknownSubagentError as exc:
            return f"Error: {exc}"
        return (await executor.arun(spec, prompt)).render()

    return StructuredTool.from_function(
        func=task,
        coroutine=atask,
        name=TASK,
        description=_DESCRIPTION.format(types=registry.describe()),
        args_schema=TaskInput,
    )
