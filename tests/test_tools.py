from dearflow.config import AppConfig
from dearflow.tools import ToolRegistry

TIME = "dearflow.tools.builtins:current_time"
CALC = "dearflow.tools.builtins:calculator"


def _config():
    return AppConfig(
        default_model="m",
        tools=[
            {"name": "current_time", "group": "utility", "use": TIME},
            {"name": "disabled", "group": "utility", "use": TIME, "enabled": False},
            {"name": "calculator", "group": "math", "use": CALC},
        ],
    )


def test_registry_loads_enabled_tools_and_filters_groups():
    registry = ToolRegistry(_config())

    assert [t.name for t in registry.get_tools()] == ["current_time", "calculator"]
    assert [t.name for t in registry.get_tools(groups={"utility"})] == ["current_time"]


def test_registry_merges_extra_tools_and_skips_duplicates():
    from dearflow.tools.builtins import calculator

    registry = ToolRegistry(_config(), extra_tools=[calculator])

    assert [t.name for t in registry.get_tools()] == ["current_time", "calculator"]
