"""The `ask_clarification` tool.

The function body never does real work. `ClarificationMiddleware` intercepts
the call, shows the question to the user and ends the run; the tool exists so
the model sees a proper schema and description.

`return_direct=True` is what ends the run: LangChain skips the usual "tool
result goes back to the model" step when every called tool is return_direct.
"""

from __future__ import annotations

from typing import Literal, Required, TypedDict

from langchain_core.tools import tool

ASK_CLARIFICATION = "ask_clarification"

ClarificationType = Literal[
    "missing_info",
    "ambiguous_requirement",
    "approach_choice",
    "risk_confirmation",
    "suggestion",
]


class ClarificationFormField(TypedDict, total=False):
    """One field in a structured clarification form."""

    name: Required[str]
    label: str
    type: Literal["text", "textarea", "number", "select", "multi_select", "checkbox", "date"]
    required: bool
    options: list[str]
    placeholder: str


@tool(ASK_CLARIFICATION, return_direct=True)
def ask_clarification(
    question: str,
    clarification_type: ClarificationType,
    context: str | None = None,
    options: list[str] | None = None,
    fields: list[ClarificationFormField] | None = None,
) -> str:
    """Ask the user a question and wait for their answer before continuing.

    Use it when you cannot proceed well without the user: required information
    is missing (missing_info), the request can be read several ways
    (ambiguous_requirement), there are several valid approaches
    (approach_choice), an action is risky or irreversible (risk_confirmation),
    or you want to suggest something and get approval (suggestion).

    Ask ONE clear question. Give `options` for one choice, or `fields` to
    collect several values in one form. Do not call other tools in the same
    turn: the run stops here and the user's reply starts the next turn.
    """
    return "Clarification request processed by middleware"
