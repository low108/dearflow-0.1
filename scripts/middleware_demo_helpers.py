"""Seeded fixture scenarios; all continuation model calls use the real provider.

Import only: this module never reads credentials, writes app configuration or
starts network calls by itself. Run labels explicitly distinguish test seeds
from history actually produced by the live model.
"""

from __future__ import annotations

from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.models import ModelFactory
from dearflow.persistence import Persistence
from dearflow.service import RunResult, RunService


class ForcedCalculatorChatOpenAI(ChatOpenAI):
    """Real ChatOpenAI with test-only provider-enforced tool choice.

    Normal unbound calls (e.g. summarization/memory) remain ordinary real calls.
    Forced tool choice is disclosed in trace metadata and scenario report.
    """

    fixture_tool_choice: str = Field(default="calculator", exclude=True)

    def bind_tools(self, tools, **kwargs):
        names = {convert_to_openai_tool(tool)["function"]["name"] for tool in tools}
        # Structured memory extraction binds its own schema and explicit choice.
        # Force only the scenario agent's tools, preserving all explicit choices.
        if kwargs.get("tool_choice") is None and (
            self.fixture_tool_choice in names
            or self.fixture_tool_choice == "none"
            and bool(names & {"calculator", "task", "ask_clarification"})
        ):
            kwargs["tool_choice"] = self.fixture_tool_choice
            kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, **kwargs)


class ForcedCalculatorFactory(ModelFactory):
    def __init__(self, forced_tool: str = "calculator"):
        if forced_tool not in {"calculator", "task", "none"}:
            raise ValueError("Supported test tool choices: calculator, task, none")
        self.forced_tool = forced_tool

    def create(self, model_name: str):
        provider, name = model_name.split(":", 1)
        if provider != "openai":
            raise ValueError(
                "This test-only fixture supports the configured OpenAI-compatible provider."
            )
        return ForcedCalculatorChatOpenAI(
            model=name,
            temperature=0,
            max_tokens=256,
            timeout=60,
            max_retries=1,
            fixture_tool_choice=self.forced_tool,
        )


def with_overrides(config: AppConfig, **overrides) -> AppConfig:
    """Validated deep merge: retains every enabled middleware unless changed."""

    def merge(base, changes):
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    return AppConfig.model_validate(merge(config.model_dump(), overrides))


def run_seeded(
    service: RunService,
    *,
    label: str,
    state: dict[str, Any],
    thread_id: str,
    user_id: str,
    fixture_details: str,
) -> RunResult:
    """Run arbitrary starting history without inserting a new turn boundary.

    Uses the app's durable run wrapper and real graph so token accounting,
    callbacks, checkpoints and LangSmith parenting are unchanged. No fake usage
    or model outputs are generated. The seed AI/tool messages are fixture data.
    """
    with service._tracked(label, thread_id, user_id) as run:
        run.config["recursion_limit"] = 40
        run.config["tags"] = ["middleware-live-demo", "seeded-history-fixture"]
        run.config["metadata"].update(
            {
                "scenario": label,
                "seeded_history": True,
                "fixture_details": fixture_details,
                "continuation_model": "real-provider",
            }
        )
        run.state = service._agent.graph.invoke(state, run.config)
    return RunResult(run.run_id, thread_id, run.state, run.usage)


def loop_history(repetitions: int = 3) -> dict[str, Any]:
    """Three seeded identical completed calls; next actual request gets warning.

    With configured warn=3/hard=5 and ForcedCalculatorFactory, matching next real
    calls are fourth (executes) and fifth (blocked before execution). Arguments
    remain genuine model output, so final caller must verify the stop_reason.
    """
    messages = [
        HumanMessage(
            content=(
                "This is a bounded middleware loop fixture. Continue the calculation "
                'by calling calculator with left=17, right=23, operation="multiply". '
                "Use these exact arguments whenever you call calculator. "
                "There is no other task and do not create todos."
            )
        )
    ]
    for index in range(repetitions):
        call_id = f"seed_loop_{index}"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"left": 17, "right": 23, "operation": "multiply"},
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="391.0", tool_call_id=call_id, name="calculator"),
            ]
        )
    return {"messages": messages}


def interrupted_history() -> dict[str, Any]:
    """Missing result plus orphan result. Neither is a claimed live tool call."""
    return {
        "messages": [
            HumanMessage(
                content="Fixture history: an earlier time lookup was interrupted."
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "current_time",
                        "args": {"timezone_name": "Europe/Paris"},
                        "id": "seed_interrupted_time",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Fixture orphan result: no associated call survives.",
                tool_call_id="seed_orphan_call",
                name="calculator",
            ),
            HumanMessage(
                content=(
                    "The previous lookup was interrupted. Use current_time once for "
                    "Europe/Paris and report the actual current time. Do not use the "
                    "orphan result. No delegation or todo list is necessary."
                )
            ),
        ]
    }


def summarization_history() -> dict[str, Any]:
    """Fabricated long benign history, explicitly marked before real summary."""
    messages = []
    topics = [
        "budget",
        "agenda",
        "guests",
        "venue",
        "transport",
        "catering",
        "slides",
        "follow-up",
    ]
    for index, topic in enumerate(topics):
        paragraph = (
            f"Seeded workshop planning record {index + 1}: the {topic} discussion "
            "concerned a fictional educational event called Cedar Workshop. "
            "The agreed outcome is a small practical session with clear examples, "
            "accessible explanations, time for questions, and concise notes. "
            "The fictional contact is sample.person@example.com. "
            "Participants prefer a 30-minute morning session and green notebooks. "
        )
        messages.extend(
            [
                HumanMessage(content=paragraph * 3),
                AIMessage(
                    content=(
                        f"Fixture response {index + 1}: recorded the {topic} decision. "
                        "The Cedar Workshop lasts 30 minutes and uses green notebooks. "
                        "This message is seeded history, not a prior provider response."
                    )
                ),
            ]
        )
    messages.append(
        HumanMessage(
            content=(
                "Using the earlier workshop details, state its name, duration and "
                "notebook color in one sentence. Do not call tools or create todos."
            )
        )
    )
    return {"messages": messages}


class RecordingModelFactory(ModelFactory):
    """Attach an optional demo callback to every real model this factory builds."""

    def __init__(self, delegate, callback):
        self.delegate, self.callback = delegate, callback

    def create(self, model_name):
        model = self.delegate.create(model_name)
        model.callbacks = [*(model.callbacks or []), self.callback]
        return model


def build_service(
    config: AppConfig,
    *,
    forced_calculator: bool = False,
    forced_tool: str | None = None,
    model_callback=None,
):
    """Temporary in-memory checkpoints; caller owns memory folder in config."""
    if forced_tool is None and forced_calculator:
        forced_tool = "calculator"
    db = Persistence.in_memory()
    models = (
        ForcedCalculatorFactory(forced_tool)
        if forced_tool is not None
        else ModelFactory()
    )
    if model_callback is not None:
        models = RecordingModelFactory(models, model_callback)
    factory = LeadAgentFactory(
        config,
        checkpointer=db.checkpointer,
        model_factory=models,
    )
    agent = factory.build({"configurable": {"is_plan_mode": True}})
    return factory, RunService(agent, db), db


def task_limit_history() -> dict[str, Any]:
    """Three seeded earlier delegations; real fourth task must hit max_total=3.

    Build with forced_tool='task'. The seeded task reports are explicitly
    fixtures, and do not claim actual subagent executions or tool receipts.
    """
    messages = [
        HumanMessage(
            content=(
                "This is a bounded delegation-limit fixture. Three earlier delegations "
                "below are seeded test history. Request one more general-purpose "
                "subagent to state the result of 2 + 2 without using tools. "
                'Use description="Fourth delegation". Do not create todos.'
            )
        )
    ]
    for index in range(3):
        call_id = f"seed_task_{index}"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "prompt": "State the result of 2 + 2 without tools.",
                                "subagent_type": "general-purpose",
                                "description": f"Seeded delegation {index + 1}",
                            },
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=(
                        "Seeded fixture result: 2 + 2 = 4. This is synthetic earlier "
                        "history; no actual subagent was executed for this message."
                    ),
                    tool_call_id=call_id,
                    name="task",
                ),
            ]
        )
    return {"messages": messages}


def todo_history() -> dict[str, Any]:
    """Pending todo fixture with real model forced to return text only.

    Build with forced_tool='none'. Todo before_model first restores the missing
    visible list; after_model should redirect twice, then allow final text with
    the todo still pending. This demonstrates bounded reminders, not completion.
    """
    return {
        "messages": [
            HumanMessage(
                content=(
                    "This is a bounded todo-reminder fixture. Reply with the word Done "
                    "and no other text. Do not call write_todos or any other tool. "
                    "The pending todo is seeded test state."
                )
            )
        ],
        "todos": [
            {
                "content": "Fixture pending task: verify the workshop checklist",
                "status": "pending",
            }
        ],
    }
