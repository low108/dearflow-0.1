"""Asks a model what is worth remembering."""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, StringConstraints, ValidationError

from dearflow.memory.exchange import Exchange
from dearflow.memory.store import CATEGORIES, Fact
from dearflow.text import text_of

logger = logging.getLogger(__name__)

_PROMPT = """\
Extract lasting facts about the USER from this conversation.

<conversation>
{conversation}
</conversation>

{output_format}

Rules:
- Only clear, specific facts about the user (preferences, expertise, background, goals, corrections).
- Use one category from {categories} and a numeric confidence from 0.0 to 1.0.
- Explicit statements get confidence 0.9+, implied ones 0.6-0.8.
- Skip anything vague, temporary, or about the assistant.
- Skip instructions limited to the current task, such as which tools to call, calculations to run,
  cities to look up, and demo constraints. A one-turn request is not a lasting preference or goal.
- Write each fact as one short third-person sentence, e.g. "Prefers concise answers".
- If there is nothing worth remembering, return {{"facts": []}}."""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_JSON_FORMAT = 'Return ONLY JSON: {"facts": [{"content": "...", "category": "preference", "confidence": 0.9}]}'
_TOOL_FORMAT = "Submit the extracted facts using the MemoryFacts tool."


class _MemoryFact(BaseModel):
    """One lasting fact explicitly grounded in the user's conversation."""

    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    category: Literal[
        "preference", "knowledge", "context", "behavior", "goal", "correction"
    ]
    confidence: float = Field(ge=0, le=1)


class MemoryFacts(BaseModel):
    """Record lasting user facts; use an empty list when nothing should be remembered."""

    facts: list[_MemoryFact]


def format_conversation(exchanges: list[Exchange], *, limit: int = 2000) -> str:
    lines: list[str] = []
    for ex in exchanges:
        lines.append(f"User: {ex.user[:limit]}")
        if ex.assistant:
            lines.append(f"Assistant: {ex.assistant[:limit]}")
    return "\n".join(lines)


def parse_facts(raw: str, *, threshold: float) -> list[Fact]:
    """Turn the model's reply into facts. Bad JSON or bad items are skipped, never raised."""
    facts, _ = _parse_facts(raw, threshold=threshold)
    return facts


def _parse_facts(raw: str, *, threshold: float) -> tuple[list[Fact], bool]:
    """Return valid facts and whether the response had malformed JSON or items."""
    text = _FENCE.sub("", _THINK.sub("", raw).strip())
    try:
        response = json.loads(text)
    except ValueError:
        logger.warning("Memory extraction returned unparseable output")
        return [], True
    if not isinstance(response, dict) or not isinstance(response.get("facts"), list):
        logger.warning("Memory extraction returned an invalid facts schema")
        return [], True
    facts: list[Fact] = []
    malformed = False
    for item in response["facts"]:
        try:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("content"), str)
                or not item["content"].strip()
            ):
                raise ValueError("Fact content must be a non-empty string")
            fact = Fact(
                content=item["content"].strip(),
                category=item.get("category")
                if item.get("category") in CATEGORIES
                else "context",
                confidence=float(item.get("confidence", 0.8)),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            malformed = True
            continue
        if fact.content and fact.confidence >= threshold:
            facts.append(fact)
    return facts, malformed


class FactExtractor:
    def __init__(
        self, model: BaseChatModel, *, confidence_threshold: float = 0.7
    ) -> None:
        self._model = model
        self._threshold = confidence_threshold
        self._structured = None
        with_structured_output = getattr(model, "with_structured_output", None)
        if with_structured_output is not None:
            try:
                self._structured = with_structured_output(
                    MemoryFacts, method="function_calling", include_raw=True
                )
            except NotImplementedError:
                pass  # Models explicitly lacking this capability keep JSON extraction.

    def _attempt(self, prompt: str) -> tuple[list[Fact], bool]:
        if self._structured is None:
            return _parse_facts(
                text_of(self._model.invoke(prompt)), threshold=self._threshold
            )
        result = self._structured.invoke(prompt)
        if result.get("parsing_error") is not None:
            return [], True
        try:
            batch = MemoryFacts.model_validate(result.get("parsed"))
        except ValidationError:
            return [], True
        return [
            Fact(**fact.model_dump())
            for fact in batch.facts
            if fact.confidence >= self._threshold
        ], False

    def extract(self, exchanges: list[Exchange]) -> list[Fact]:
        prompt = _PROMPT.format(
            conversation=format_conversation(exchanges),
            categories="|".join(CATEGORIES),
            output_format=_TOOL_FORMAT
            if self._structured is not None
            else _JSON_FORMAT,
        )
        facts, malformed = self._attempt(prompt)
        if malformed:
            logger.warning("Retrying malformed memory extraction once")
            facts, _ = self._attempt(
                prompt
                + "\n\nThe previous extraction did not satisfy the required facts schema. "
                "Extract the facts again from the conversation above. "
                "Use an empty facts array if there are no lasting facts.\n"
                + (_TOOL_FORMAT if self._structured is not None else _JSON_FORMAT)
            )
        return facts
