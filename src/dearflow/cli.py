"""Interactive terminal chat."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence

from langgraph.checkpoint.memory import InMemorySaver

from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.models import ModelFactory

ChatRunner = Callable[..., None]


def run_chat(
    *,
    app_config: AppConfig,
    thread_id: str,
    model_name: str | None = None,
    model_factory: ModelFactory | None = None,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    """Run a checkpointed chat loop until the user types exit/quit."""
    factory = LeadAgentFactory(
        app_config,
        model_factory=model_factory,
        checkpointer=InMemorySaver(),
    )
    agent = factory.build({"configurable": {"model_name": model_name}})
    run_config = {"configurable": {"thread_id": thread_id}}

    write(f"DearFlow CLI — model: {agent.model_name}")
    write("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            message = read("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue

        result = agent.graph.invoke({"messages": [{"role": "user", "content": message}]}, run_config)
        content = result["messages"][-1].content
        write(f"DearFlow: {content if isinstance(content, str) else str(content)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DearFlow interactive CLI.")
    parser.add_argument(
        "--config",
        default=os.environ.get("DEARFLOW_CONFIG", "config.yaml"),
        help="YAML configuration path (default: DEARFLOW_CONFIG or config.yaml).",
    )
    parser.add_argument("--model", help="Model alias or identifier; overrides default_model.")
    parser.add_argument("--thread-id", default="cli", help="Conversation checkpoint thread ID.")
    parser.add_argument("--system-prompt", help="Override the configured system prompt.")
    parser.add_argument("--plan", action="store_true", help="Plan mode: enable the write_todos task list.")
    parser.add_argument("--subagents", action="store_true", help="Enable the `task` tool for delegating to subagents.")
    return parser


def main(argv: Sequence[str] | None = None, *, chat_runner: ChatRunner = run_chat) -> int:
    args = _build_parser().parse_args(argv)
    app_config = AppConfig.from_yaml(args.config)
    if args.system_prompt:
        app_config = app_config.model_copy(update={"system_prompt": args.system_prompt})

    if args.plan:
        app_config = app_config.model_copy(update={"plan_mode": True})

    if args.subagents:
        app_config = app_config.model_copy(update={"subagents": app_config.subagents.model_copy(update={"enabled": True})})

    chat_runner(app_config=app_config, thread_id=args.thread_id, model_name=args.model)
    return 0
