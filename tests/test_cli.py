from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from dearflow.cli import main, run_chat


def test_run_chat_assembles_the_requested_model_and_handles_a_turn():
    requested_models: list[str] = []
    model = FakeMessagesListChatModel(responses=[AIMessage(content="Hello from the CLI")])
    user_input = iter(["Hello", "quit"])
    output: list[str] = []

    run_chat(
        model_name="test:model",
        thread_id="cli-test",
        model_factory=lambda name: requested_models.append(name) or model,
        read=lambda _prompt: next(user_input),
        write=output.append,
    )

    assert requested_models == ["test:model"]
    assert output == [
        "DearFlow CLI — model: test:model",
        "Type 'exit' or 'quit' to stop.",
        "DearFlow: Hello from the CLI",
    ]


def test_main_uses_yaml_configuration_for_the_cli_session(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
default_model: fast
system_prompt: Answer briefly.
models:
  fast:
    model: openai:ilmu-mini-v3.3
tools:
  - name: calculator
    group: math
    use: dearflow.tools.builtins:calculator
""".strip(),
        encoding="utf-8",
    )
    sessions: list[dict] = []

    exit_code = main(
        ["--config", str(config_path), "--thread-id", "configured-thread"],
        chat_runner=lambda **settings: sessions.append(settings),
    )

    assert exit_code == 0
    assert len(sessions) == 1
    assert sessions[0]["model_name"] == "openai:ilmu-mini-v3.3"
    assert sessions[0]["thread_id"] == "configured-thread"
    assert sessions[0]["system_prompt"] == "Answer briefly."
    assert [tool.name for tool in sessions[0]["app_config"].tools] == ["calculator"]
