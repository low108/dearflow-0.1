"""Debounced background memory updates."""

from __future__ import annotations

import logging
import threading

from dearflow.memory.extractor import Exchange, FactExtractor
from dearflow.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """The one object the rest of the app talks to.

    `enqueue` is cheap and never blocks the chat. Exchanges pile up per user and
    a timer restarts on each new one (debounce), so a burst of quick turns costs
    ONE extraction call, and no exchange is lost. `flush` runs pending work now
    (used on exit and in tests). Failures are logged; memory never breaks a chat.
    """

    def __init__(
        self,
        store: MemoryStore,
        extractor: FactExtractor,
        *,
        debounce_seconds: float = 10,
        max_injection_tokens: int = 1500,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._debounce = debounce_seconds
        self._max_injection_tokens = max_injection_tokens
        self._pending: dict[str, list[Exchange]] = {}
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def context(self, user_id: str) -> str:
        """Prompt text for this user, or "" when nothing is known."""
        try:
            return self._store.render_context(user_id, max_tokens=self._max_injection_tokens)
        except Exception:
            logger.warning("Could not read memory", exc_info=True)
            return ""

    def enqueue(self, user_id: str, exchanges: list[Exchange]) -> None:
        if not exchanges:
            return
        with self._lock:
            self._pending.setdefault(user_id, []).extend(exchanges)
            if self._timer is not None:
                self._timer.cancel()
            if self._debounce > 0:
                self._timer = threading.Timer(self._debounce, self.flush)
                self._timer.daemon = True
                self._timer.start()

    def flush(self) -> None:
        """Extract and save everything pending, now."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            batch, self._pending = self._pending, {}
        for user_id, exchanges in batch.items():
            try:
                facts = self._extractor.extract(exchanges)
                if facts:
                    self._store.add(user_id, facts)
            except Exception:
                logger.warning("Memory update failed for user %s", user_id, exc_info=True)
