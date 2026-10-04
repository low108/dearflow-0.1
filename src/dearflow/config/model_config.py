class ModelConfig:
    """One named model profile from the application configuration."""

    __slots__ = ("name", "model")

    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model

