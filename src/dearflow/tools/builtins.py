from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool
def current_time(timezone_name: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone such as Asia/Kuala_Lumpur."""

    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc
    return datetime.now(timezone).isoformat()


@tool
def calculator(
    left: float,
    right: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> float:
    """Perform one arithmetic operation on two numbers."""

    if operation == "add":
        return left + right
    if operation == "subtract":
        return left - right
    if operation == "multiply":
        return left * right
    if right == 0:
        raise ValueError("Cannot divide by zero")
    return left / right

