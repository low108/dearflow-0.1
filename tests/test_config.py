import pytest
from pydantic import ValidationError

from dearflow.config import AppConfig, ModelConfig


def test_from_yaml_reads_models_tools_and_prompt(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
default_model: fast
system_prompt: You are concise.
models:
  fast: {model: "openai:ilmu-mini-v3.3"}
  reasoning: {model: "openai:ilmu-v3.1"}
tools:
  - {name: current_time, group: utility, use: "dearflow.tools.builtins:current_time"}
  - {name: disabled_tool, group: utility, use: "dearflow.tools.builtins:current_time", enabled: false}
""",
        encoding="utf-8",
    )

    config = AppConfig.from_yaml(path)

    assert config.system_prompt == "You are concise."
    assert config.resolve_model() == "openai:ilmu-mini-v3.3"
    assert config.resolve_model("reasoning") == "openai:ilmu-v3.1"
    assert config.resolve_model("raw:model-id") == "raw:model-id"
    assert [t.enabled for t in config.tools] == [True, False]


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        AppConfig(default_model="x", typo_field=1)


def test_tool_reference_must_contain_an_attribute():
    with pytest.raises(ValidationError):
        AppConfig(default_model="x", tools=[{"name": "t", "group": "g", "use": "no_colon"}])


def test_alias_resolution_uses_model_profiles():
    config = AppConfig(default_model="fast", models={"fast": ModelConfig(model="p:m")})
    assert config.resolve_model() == "p:m"
