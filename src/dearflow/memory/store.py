"""Facts about the user, kept in one JSON file per user."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

CATEGORIES = ("preference", "knowledge", "context", "behavior", "goal", "correction")
_CHARS_PER_TOKEN = 4  # rough; good enough for a budget


class Fact(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    content: str
    category: str = "context"
    confidence: float = Field(default=0.8, ge=0, le=1)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def _key(content: str) -> str:
    """Normalized text used to spot duplicates."""
    return re.sub(r"\s+", " ", content).strip().casefold()


class MemoryStore:
    """Loads, merges and saves facts. Safe to call from several threads.

    DeerFlow keeps summaries in `memory.json` and each fact as a Markdown file.
    One JSON file per user is enough to show the idea.
    """

    def __init__(self, root: str | Path, *, max_facts: int = 100) -> None:
        self._root = Path(root)
        self._max_facts = max_facts
        self._lock = threading.Lock()

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id) or "default"
        return self._root / safe / "memory.json"

    def load(self, user_id: str) -> list[Fact]:
        path = self.path_for(user_id)
        if not path.exists():
            return []
        return [Fact.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8")).get("facts", [])]

    def add(self, user_id: str, new_facts: list[Fact]) -> list[Fact]:
        """Merge facts: duplicates keep the higher confidence; the cap drops the weakest."""
        with self._lock:
            by_key = {_key(f.content): f for f in self.load(user_id)}
            for fact in new_facts:
                existing = by_key.get(_key(fact.content))
                if existing is None:
                    by_key[_key(fact.content)] = fact
                elif fact.confidence > existing.confidence:
                    existing.confidence = fact.confidence
            kept = sorted(by_key.values(), key=lambda f: (f.confidence, f.created_at), reverse=True)
            kept = kept[: self._max_facts]
            self._save(user_id, kept)
            return kept

    def delete(self, user_id: str, fact_id: str) -> bool:
        with self._lock:
            facts = self.load(user_id)
            remaining = [f for f in facts if f.id != fact_id]
            if len(remaining) == len(facts):
                return False
            self._save(user_id, remaining)
            return True

    def render_context(self, user_id: str, *, max_tokens: int) -> str:
        """Facts as prompt text, most confident first, cut to a token budget."""
        facts = sorted(self.load(user_id), key=lambda f: (f.confidence, f.created_at), reverse=True)
        lines: list[str] = []
        used = 0
        for fact in facts:
            line = f"- [{fact.category}] {fact.content}"
            used += len(line) // _CHARS_PER_TOKEN + 1
            if used > max_tokens:
                break
            lines.append(line)
        return "What you know about this user:\n" + "\n".join(lines) if lines else ""

    def _save(self, user_id: str, facts: list[Fact]) -> None:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"facts": [f.model_dump() for f in facts]}, indent=2), encoding="utf-8")
        os.replace(temp, path)  # atomic: a crash never leaves half a file
