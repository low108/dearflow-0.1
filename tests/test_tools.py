from dearflow.config import AppConfig, ToolConfig
from dearflow.tools import get_available_tools


def test_get_available_tools_loads_enabled_configured_tools_and_filters_groups():
    config = AppConfig(
        default_model="test-model",
        tools=[
            ToolConfig(
                "current_time",
                "utility",
                "dearflow.tools.builtins:current_time",
            ),
            ToolConfig(
                "disabled_time",
                "utility",
                "dearflow.tools.builtins:current_time",
                enabled=False,
            ),
            ToolConfig(
                "calculator",
                "math",
                "dearflow.tools.builtins:calculator",
            ),
        ],
    )

    all_tools = get_available_tools(config)
    utility_tools = get_available_tools(config, groups={"utility"})

    assert [tool.name for tool in all_tools] == ["current_time", "calculator"]
    assert [tool.name for tool in utility_tools] == ["current_time"]
