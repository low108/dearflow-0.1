from dearflow.config import load_app_config


def test_load_app_config_reads_model_profiles_and_system_prompt(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
default_model: fast
system_prompt: You are a concise Malaysian assistant.
models:
  fast:
    model: openai:ilmu-mini-v3.3
  reasoning:
    model: openai:ilmu-v3.1
tools:
  - name: current_time
    group: utility
    use: dearflow.tools.builtins:current_time
  - name: disabled_tool
    group: utility
    use: dearflow.tools.builtins:current_time
    enabled: false
""".strip(),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.default_model == "fast"
    assert config.system_prompt == "You are a concise Malaysian assistant."
    assert config.resolve_model() == "openai:ilmu-mini-v3.3"
    assert config.resolve_model("reasoning") == "openai:ilmu-v3.1"
    assert [tool.name for tool in config.tools] == ["current_time", "disabled_tool"]
    assert config.tools[0].group == "utility"
    assert config.tools[0].enabled is True
    assert config.tools[1].enabled is False
