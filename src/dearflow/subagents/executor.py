"""Runs one subagent to completion."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from dearflow.middlewares.messages import text_of
from dearflow.middlewares.tool_error_handling import ToolErrorHandlingMiddleware
from dearflow.subagents.spec import INHERIT, SubagentSpec
from dearflow.usage import CALLER_KEY, SUBAGENT

logger = logging.getLogger(__name__)

NO_RESPONSE = "No response generated"


class SubagentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    MAX_TURNS = "max_turns"


@dataclass(frozen=True)
class SubagentResult:
    status: SubagentStatus
    text: str

    def render(self) -> str:
        """The string the lead agent receives as the `task` tool result."""
        if self.status is SubagentStatus.COMPLETED:
            return f"Task Succeeded. Result: {self.text}"
        if self.status is SubagentStatus.MAX_TURNS:
            return f"Task stopped at its turn limit. Partial result: {self.text}"
        if self.status is SubagentStatus.TIMED_OUT:
            return f"Task timed out. {self.text}"
        return f"Task failed. Error: {self.text}"


def filter_tools(tools: Sequence[BaseTool], allowed: Sequence[str] | None, disallowed: Sequence[str]) -> list[BaseTool]:
    """Allowlist first (if any), then the denylist always wins."""
    kept = [t for t in tools if allowed is None or t.name in allowed]
    return [t for t in kept if t.name not in disallowed]


def final_text(state: dict[str, Any] | None) -> str:
    """The last AI message's text: a subagent's answer is whatever it said last."""
    for message in reversed((state or {}).get("messages", [])):
        if isinstance(message, AIMessage) and text_of(message).strip():
            return text_of(message)
    return NO_RESPONSE


class SubagentExecutor:
    """Builds a fresh agent for each task and runs it.

    Isolation is the point: the subagent sees only the task text (no lead
    conversation), has its own system prompt and a filtered tool set, and
    returns only its last message. The lead agent's context stays small.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        default_model: BaseChatModel,
        model_resolver,
        extra_middleware: Sequence[Any] = (),
    ) -> None:
        self._tools = list(tools)
        self._extra_middleware = list(extra_middleware)  # e.g. the token budget, shared with the lead agent
        self._default_model = default_model
        self._resolve_model = model_resolver  # alias -> chat model, for specs with their own model

    def _build(self, spec: SubagentSpec):
        model = self._default_model if spec.model == INHERIT else self._resolve_model(spec.model)
        return create_agent(
            model=model,
            tools=filter_tools(self._tools, spec.tools, spec.disallowed_tools),
            middleware=[ToolErrorHandlingMiddleware(), *self._extra_middleware],
            system_prompt=spec.system_prompt,
        )

    @staticmethod
    def _run_config(spec: SubagentSpec) -> dict[str, Any]:
        # One turn = a model step + a tools step, plus the final answer step.
        # The tag lets the usage counter tell subagent model calls from the lead agent's.
        return {"recursion_limit": spec.max_turns * 2 + 1, "metadata": {CALLER_KEY: SUBAGENT}}

    @staticmethod
    def _inputs(prompt: str) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": prompt}]}

    # Streaming "values" keeps the latest state, so if the turn limit cuts the
    # run short we can still return the partial work.

    def run(self, spec: SubagentSpec, prompt: str) -> SubagentResult:
        last: dict[str, Any] | None = None
        try:
            for last in self._build(spec).stream(self._inputs(prompt), self._run_config(spec), stream_mode="values"):
                pass
        except GraphRecursionError:
            return SubagentResult(SubagentStatus.MAX_TURNS, final_text(last))
        except Exception as exc:
            logger.exception("Subagent %s failed", spec.name)
            return SubagentResult(SubagentStatus.FAILED, f"{type(exc).__name__}: {exc}")
        return SubagentResult(SubagentStatus.COMPLETED, final_text(last))

    async def arun(self, spec: SubagentSpec, prompt: str) -> SubagentResult:
        last: dict[str, Any] | None = None

        async def consume() -> None:
            nonlocal last
            graph = self._build(spec)
            async for last in graph.astream(self._inputs(prompt), self._run_config(spec), stream_mode="values"):
                pass

        try:
            await asyncio.wait_for(consume(), timeout=spec.timeout_seconds)
        except TimeoutError:
            return SubagentResult(
                SubagentStatus.TIMED_OUT, f"Stopped after {spec.timeout_seconds}s. Partial result: {final_text(last)}"
            )
        except GraphRecursionError:
            return SubagentResult(SubagentStatus.MAX_TURNS, final_text(last))
        except Exception as exc:
            logger.exception("Subagent %s failed", spec.name)
            return SubagentResult(SubagentStatus.FAILED, f"{type(exc).__name__}: {exc}")
        return SubagentResult(SubagentStatus.COMPLETED, final_text(last))
