from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from dearflow.agents import assemble_lead_agent, build_agent_dependencies
from dearflow.config import AppConfig, load_app_config

ModelFactory = Callable[[str], BaseChatModel]
ChatRunner = Callable[..., None]
DEFAULT_SYSTEM_PROMPT = "You are DearFlow, a helpful general-purpose agent."


def _create_model(model_name: str) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    return init_chat_model(model_name)


def run_chat(
    *,
    model_name: str,
    thread_id: str,
    model_factory: ModelFactory = _create_model,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    app_config: AppConfig | None = None,
) -> None:
    """Run an interactive, checkpointed terminal conversation."""

    base_config = app_config or AppConfig(default_model=model_name)
    session_config = AppConfig(
        default_model=base_config.default_model,
        system_prompt=system_prompt,
        models=base_config.models,
        tools=base_config.tools,
    )
    assembly = assemble_lead_agent(
        {"configurable": {"model_name": model_name}},
        app_config=session_config,
        dependencies=build_agent_dependencies(
            session_config,
            create_model=model_factory,
            checkpointer=InMemorySaver(),
        ),
    )
    run_config = {"configurable": {"thread_id": thread_id}}

    write(f"DearFlow CLI — model: {assembly.effective_model}")
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

        result = assembly.graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            run_config,
        )
        content = result["messages"][-1].content
        write(f"DearFlow: {content if isinstance(content, str) else str(content)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DearFlow interactive CLI.")
    parser.add_argument(
        "--config",
        default=os.environ.get("DEARFLOW_CONFIG"),
        help="YAML application configuration path. Defaults to DEARFLOW_CONFIG.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEARFLOW_MODEL"),
        help="LangChain model identifier, for example openai:gpt-4.1-mini. Defaults to DEARFLOW_MODEL.",
    )
    parser.add_argument("--thread-id", default="cli", help="Conversation checkpoint thread ID.")
    parser.add_argument(
        "--system-prompt",
        help="Override the configured system prompt for this CLI session.",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, chat_runner: ChatRunner = run_chat) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    app_config = load_app_config(args.config) if args.config else None
    if app_config is None and not args.model:
        parser.error("provide --config/DEARFLOW_CONFIG or --model/DEARFLOW_MODEL")

    if app_config is not None:
        model_name = app_config.resolve_model(args.model)
        system_prompt = args.system_prompt or app_config.system_prompt
    else:
        model_name = args.model
        system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    chat_runner(
        model_name=model_name,
        thread_id=args.thread_id,
        system_prompt=system_prompt,
        app_config=app_config,
    )
    return 0
