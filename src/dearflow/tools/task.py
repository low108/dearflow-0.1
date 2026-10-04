"""The `task` tool: how the lead agent delegates work to a subagent."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_stream_writer
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


def _emit(payload: dict) -> None:
    """Send a progress event to `custom` stream listeners. Harmless if nobody listens."""
    try:
        get_stream_writer()(payload)
    except RuntimeError:  # called outside a graph run
        pass


def create_task_tool(executor: SubagentExecutor, registry: SubagentRegistry) -> BaseTool:
    """Build the `task` tool. Both sync and async callers are supported."""

    def task(prompt: str, subagent_type: str, description: str = "") -> str:
        try:
            spec = registry.get(subagent_type)
        except UnknownSubagentError as exc:
            return f"Error: {exc}"
        _emit({"type": "subagent.start", "subagent_type": spec.name, "description": description})
        result = executor.run(spec, prompt)
        _emit({"type": "subagent.end", "subagent_type": spec.name, "status": result.status.value})
        return result.render()

    async def atask(prompt: str, subagent_type: str, description: str = "") -> str:
        try:
            spec = registry.get(subagent_type)
        except UnknownSubagentError as exc:
            return f"Error: {exc}"
        _emit({"type": "subagent.start", "subagent_type": spec.name, "description": description})
        result = await executor.arun(spec, prompt)
        _emit({"type": "subagent.end", "subagent_type": spec.name, "status": result.status.value})
        return result.render()

    return StructuredTool.from_function(
        func=task,
        coroutine=atask,
        name=TASK,
        description=_DESCRIPTION.format(types=registry.describe()),
        args_schema=TaskInput,
    )
