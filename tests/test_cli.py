from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.cli import main, run_chat
from dearflow.config import AppConfig


def test_run_chat_handles_a_turn_then_quits():
    model = ToolCapableFakeModel(responses=[AIMessage(content="Hello from the CLI")])
    models = FakeModelFactory(model)
    inputs = iter(["Hello", "quit"])
    output: list[str] = []

    run_chat(
        app_config=AppConfig(default_model="test:model"),
        thread_id="t",
        model_factory=models,
        read=lambda _p: next(inputs),
        write=output.append,
    )

    assert models.requested == ["test:model"]
    assert output == [
        "DearFlow CLI — model: test:model",
        "Type 'exit' or 'quit' to stop.",
        "DearFlow: Hello from the CLI",
    ]


def test_main_loads_yaml_and_applies_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        'default_model: fast\nsystem_prompt: Answer briefly.\nmodels:\n  fast: {model: "p:m"}\n',
        encoding="utf-8",
    )
    sessions: list[dict] = []

    code = main(
        ["--config", str(path), "--thread-id", "t1", "--system-prompt", "Be formal."],
        chat_runner=lambda **kw: sessions.append(kw),
    )

    assert code == 0
    assert sessions[0]["thread_id"] == "t1"
    assert sessions[0]["app_config"].system_prompt == "Be formal."
    assert sessions[0]["model_name"] is None


def test_plan_flag_enables_plan_mode(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("default_model: m\n", encoding="utf-8")
    sessions: list[dict] = []

    main(["--config", str(path), "--plan"], chat_runner=lambda **kw: sessions.append(kw))

    assert sessions[0]["app_config"].plan_mode is True
