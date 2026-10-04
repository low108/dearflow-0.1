"""Which stream modes a caller may ask for."""

from __future__ import annotations

# public name -> LangGraph name. DeerFlow also offers debug/tasks/checkpoints; not needed to learn streaming.
SUPPORTED_MODES: dict[str, str] = {
    "values": "values",  # full state after every step
    "updates": "updates",  # only what each node changed
    "messages": "messages",  # LLM tokens as they are produced
    "messages-tuple": "messages",  # DeerFlow / LangGraph Platform spelling of the same thing
    "custom": "custom",  # progress events emitted by tools (e.g. subagent start/finish)
}

DEFAULT_MODES = ("values",)


class UnsupportedStreamModeError(ValueError):
    def __init__(self, modes: list[str]) -> None:
        self.modes = tuple(dict.fromkeys(modes))
        super().__init__(f"Unsupported stream mode(s): {', '.join(self.modes)}. Supported: {', '.join(SUPPORTED_MODES)}")


def normalize_modes(raw: list[str] | str | None) -> list[str]:
    """Validate requested modes and map them to LangGraph names. No silent fallback:
    an unknown mode is an error, not a quietly different stream."""
    if raw is None:
        requested = list(DEFAULT_MODES)
    else:
        requested = [raw] if isinstance(raw, str) else (list(raw) or list(DEFAULT_MODES))
    unknown = [m for m in requested if m not in SUPPORTED_MODES]
    if unknown:
        raise UnsupportedStreamModeError(unknown)
    return list(dict.fromkeys(SUPPORTED_MODES[m] for m in requested))
