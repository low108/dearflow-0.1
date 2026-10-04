"""The `task` tool: how the lead agent delegates work to a subagent."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import BaseModel, Field

from dearflow.receipts import render_verdict, verify_receipt_citations
from dearflow.subagents import SubagentExecutor, SubagentRegistry, SubagentResult, SubagentStatus, UnknownSubagentError
from dearflow.subagents.spec import TASK

_DESCRIPTION = """\
Delegate a bounded task to a subagent that works in its own fresh context and
returns only its final report.

The subagent knows NOTHING about this conversation: put everything it needs in
`prompt` (goal, relevant facts, expected output).

Subagent types:
{types}

Several `task` calls in one response run in parallel, so only launch them
together when the jobs are independent.

Reading the result: a subagent's report is a SELF-REPORT, not a verified fact. When receipts are
on, the report is followed by a `citations:` line. "N resolved" means the receipt ids the
subagent cited exist and succeeded (the calls happened; nothing about whether the claims are
right). "UNVERIFIED" means it described actions but cited no receipt. Check anything your
answer depends on."""


class TaskInput(BaseModel):
    prompt: str = Field(description="Complete, self-contained instructions for the subagent.")
    subagent_type: str = Field(description="Which subagent type to use, e.g. 'general-purpose'.")
    description: str = Field(default="", description="A 3-5 word label for the job (for logs and UIs).")
    tool_call_id: Annotated[str, InjectedToolCallId]  # filled in by the framework, hidden from the model


def _emit(payload: dict) -> None:
    """Send a progress event to `custom` stream listeners. Harmless if nobody listens."""
    try:
        get_stream_writer()(payload)
    except RuntimeError:  # called outside a graph run
        pass


def _as_command(result: SubagentResult, tool_call_id: str) -> Command:
    """The lead's tool result: the report, a citation check, and the evidence kept beside it.

    The receipts and the check travel in the message's `additional_kwargs`, so they are saved
    with the conversation (an audit can read them later) without using up model context.
    """
    content = result.render()
    extra: dict = {"subagent_status": result.status.value}
    if result.status is SubagentStatus.COMPLETED and result.receipts is not None:
        verdict = verify_receipt_citations(result.text, result.receipts)
        extra.update(receipt_verdict=verdict, tool_receipts=result.receipts)
        if line := render_verdict(verdict):
            content = f"{content}\n\n{line}"
    failed = result.status in (SubagentStatus.FAILED, SubagentStatus.TIMED_OUT)
    message = ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=TASK,
        status="error" if failed else "success",
        additional_kwargs=extra,
    )
    return Command(update={"messages": [message]})


def create_task_tool(executor: SubagentExecutor, registry: SubagentRegistry) -> BaseTool:
    """Build the `task` tool. Both sync and async callers are supported."""

    def task(prompt: str, subagent_type: str, tool_call_id: str, description: str = "") -> Command | str:
        try:
            spec = registry.get(subagent_type)
        except UnknownSubagentError as exc:
            return f"Error: {exc}"
        _emit({"type": "subagent.start", "subagent_type": spec.name, "description": description})
        result = executor.run(spec, prompt)
        _emit({"type": "subagent.end", "subagent_type": spec.name, "status": result.status.value})
        return _as_command(result, tool_call_id)

    async def atask(prompt: str, subagent_type: str, tool_call_id: str, description: str = "") -> Command | str:
        try:
            spec = registry.get(subagent_type)
        except UnknownSubagentError as exc:
            return f"Error: {exc}"
        _emit({"type": "subagent.start", "subagent_type": spec.name, "description": description})
        result = await executor.arun(spec, prompt)
        _emit({"type": "subagent.end", "subagent_type": spec.name, "status": result.status.value})
        return _as_command(result, tool_call_id)

    return StructuredTool.from_function(
        func=task,
        coroutine=atask,
        name=TASK,
        description=_DESCRIPTION.format(types=registry.describe()),
        args_schema=TaskInput,
    )
