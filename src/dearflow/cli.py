"""Interactive terminal chat."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence

from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.models import ModelFactory
from dearflow.persistence import Persistence
from dearflow.service import RunService

ChatRunner = Callable[..., None]


def run_chat(
    *,
    app_config: AppConfig,
    thread_id: str,
    user_id: str = "default",
    model_name: str | None = None,
    model_factory: ModelFactory | None = None,
    stream: bool = False,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Run a chat loop until the user types exit/quit.

    With `stream=True` the answer is printed as it is produced (through `emit`,
    which writes without a newline), along with tool and subagent activity.

    With `persistence.enabled`, the conversation is saved: start again with the
    same `--thread-id` and it continues where it stopped.
    """
    persistence = Persistence.from_config(app_config.persistence)
    factory = LeadAgentFactory(app_config, model_factory=model_factory, checkpointer=persistence.checkpointer)
    agent = factory.build({"configurable": {"model_name": model_name}})
    service = RunService(agent, persistence, track_usage=app_config.token_usage.enabled)

    try:
        write(f"DearFlow CLI — model: {agent.model_name}")
        _show_resumed_thread(service, persistence, thread_id, write)
        write("Type 'exit' or 'quit' to stop.")
        turn = _stream_turn if stream else _plain_turn
        _chat_loop(service, thread_id, user_id, read, write, emit or _stdout, turn)
    finally:
        factory.flush_memory()
        persistence.close()


def _show_resumed_thread(service: RunService, persistence: Persistence, thread_id: str, write) -> None:
    record = persistence.threads.get(thread_id)
    if record is None:
        return
    turns = sum(1 for m in service.history(thread_id) if m.type == "human")
    write(f"Resuming thread {thread_id!r}{f' — {record.title}' if record.title else ''} ({turns} earlier messages)")


def _stdout(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _plain_turn(service: RunService, message: str, thread_id: str, user_id: str, write, emit) -> None:
    result = service.invoke(message, thread_id=thread_id, user_id=user_id)
    write(f"DearFlow: {result.text}")
    _show_usage(result.usage, lambda text: write(f"  {text}"))


def _stream_turn(service: RunService, message: str, thread_id: str, user_id: str, write, emit) -> None:
    """Print one turn live. Tokens come from the model node; everything else is a status line."""
    started = False  # have we begun the "DearFlow: ..." line?

    def say(text: str) -> None:
        nonlocal started
        if not started:
            emit("DearFlow: ")
            started = True
        emit(text)

    def note(text: str) -> None:
        nonlocal started
        if started:
            emit("\n")
            started = False
        write(f"  {text}")

    for event in service.stream(message, thread_id=thread_id, user_id=user_id, modes=["messages", "updates", "custom"]):
        data = event.data
        if event.event == "messages":
            item = data["message"]
            is_model_text = data["node"] == "model" and item["type"] == "ai" and not item.get("tool_calls")
            is_question = item["type"] == "tool" and item.get("name") == "ask_clarification"
            if (is_model_text or is_question) and isinstance(item["content"], str) and item["content"]:
                say(item["content"])
        elif event.event == "updates":
            for call in _tool_calls(data):
                note(f"→ {call['name']}")
        elif event.event == "custom" and data.get("type") == "subagent.start":
            note(f"⇢ subagent {data['subagent_type']}: {data.get('description') or 'working'}")
        elif event.event == "error":
            note(f"error: {data['message']}")
        elif event.event == "end":
            _show_usage(data.get("usage") or {}, note)
    if started:
        emit("\n")


def _show_usage(usage: dict, note) -> None:
    """One line about the bill, only when the model reported token counts."""
    if usage.get("total_tokens"):
        note(f"tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
    if usage.get("stop_reason") == "token_capped":
        note("stopped early: token budget reached")


def _tool_calls(update: dict) -> list[dict]:
    """Tool calls the model just decided to make, from an `updates` chunk."""
    messages = (update.get("model") or {}).get("messages") or []
    return [call for message in messages for call in message.get("tool_calls", [])]


def _chat_loop(service: RunService, thread_id: str, user_id: str, read, write, emit, turn) -> None:
    while True:
        try:
            message = read("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue

        turn(service, message, thread_id, user_id, write, emit)


def list_threads(app_config: AppConfig, user_id: str, write: Callable[[str], None] = print) -> None:
    """Print the user's saved conversations, newest first."""
    persistence = Persistence.from_config(app_config.persistence)
    try:
        records = persistence.threads.list(user_id)
        if not records:
            write("No saved threads.")
        for record in records:
            write(f"{record.thread_id}\t{record.updated_at[:19]}\t{record.title or '(untitled)'}")
    finally:
        persistence.close()


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
    parser.add_argument("--memory", action="store_true", help="Enable long-term memory (facts saved under memory.storage_path).")
    parser.add_argument("--user-id", default="default", help="Whose memory to use.")
    parser.add_argument("--persist", action="store_true", help="Save conversations to persistence.path so they survive restarts.")
    parser.add_argument("--list-threads", action="store_true", help="List saved conversations and exit.")
    parser.add_argument("--no-stream", action="store_true", help="Wait for the full answer instead of printing it live.")
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

    if args.memory:
        app_config = app_config.model_copy(update={"memory": app_config.memory.model_copy(update={"enabled": True})})

    if args.persist or args.list_threads:
        app_config = app_config.model_copy(update={"persistence": app_config.persistence.model_copy(update={"enabled": True})})

    if args.list_threads:
        list_threads(app_config, args.user_id)
        return 0

    chat_runner(app_config=app_config, thread_id=args.thread_id, user_id=args.user_id, model_name=args.model, stream=not args.no_stream)
    return 0
