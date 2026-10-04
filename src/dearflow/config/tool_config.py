class ToolConfig:
    """Configuration for one dynamically loaded tool."""

    __slots__ = ("name", "group", "use", "enabled")

    def __init__(self, name: str, group: str, use: str, enabled: bool = True) -> None:
        self.name = name
        self.group = group
        self.use = use
        self.enabled = enabled
