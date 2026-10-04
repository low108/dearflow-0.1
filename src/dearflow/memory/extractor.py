"""Asks a model what is worth remembering."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from dearflow.memory.store import CATEGORIES, Fact
from dearflow.text import text_of

logger = logging.getLogger(__name__)

_PROMPT = """\
Extract lasting facts about the USER from this conversation.

<conversation>
{conversation}
</conversation>

Return ONLY JSON: {{"facts": [{{"content": "...", "category": "{categories}", "confidence": 0.0-1.0}}]}}

Rules:
- Only clear, specific facts about the user (preferences, expertise, background, goals, corrections).
- Explicit statements get confidence 0.9+, implied ones 0.6-0.8.
- Skip anything vague, temporary, or about the assistant.
- Write each fact as one short third-person sentence, e.g. "Prefers concise answers".
- If there is nothing worth remembering, return {{"facts": []}}."""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass(frozen=True)
class Exchange:
    """One user message and the assistant's final reply to it."""

    user: str
    assistant: str = ""


def format_conversation(exchanges: list[Exchange], *, limit: int = 2000) -> str:
    lines: list[str] = []
    for ex in exchanges:
        lines.append(f"User: {ex.user[:limit]}")
        if ex.assistant:
            lines.append(f"Assistant: {ex.assistant[:limit]}")
    return "\n".join(lines)


def parse_facts(raw: str, *, threshold: float) -> list[Fact]:
    """Turn the model's reply into facts. Bad JSON or bad items are skipped, never raised."""
    text = _FENCE.sub("", _THINK.sub("", raw).strip())
    try:
        items = json.loads(text).get("facts", [])
    except (ValueError, AttributeError):
        logger.warning("Memory extraction returned unparseable output")
        return []
    facts: list[Fact] = []
    for item in items if isinstance(items, list) else []:
        try:
            fact = Fact(
                content=str(item["content"]).strip(),
                category=item.get("category") if item.get("category") in CATEGORIES else "context",
                confidence=float(item.get("confidence", 0.8)),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if fact.content and fact.confidence >= threshold:
            facts.append(fact)
    return facts


class FactExtractor:
    def __init__(self, model: BaseChatModel, *, confidence_threshold: float = 0.7) -> None:
        self._model = model
        self._threshold = confidence_threshold

    def extract(self, exchanges: list[Exchange]) -> list[Fact]:
        prompt = _PROMPT.format(conversation=format_conversation(exchanges), categories="|".join(CATEGORIES))
        reply = self._model.invoke(prompt)
        return parse_facts(text_of(reply), threshold=self._threshold)
