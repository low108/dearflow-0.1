"""Summarization that never shows personal data to the summary model."""

from __future__ import annotations

import logging

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AnyMessage

from dearflow.pii import Redactor

logger = logging.getLogger(__name__)


class RedactingSummarizationMiddleware(SummarizationMiddleware):
    """LangChain's summarization, with personal data redacted first.

    The summary model is called from inside this middleware, so it bypasses
    `PiiRedactionMiddleware` (which only wraps the main model call). Without this, old
    messages would reach the summary model raw, and the summary that replaces them would
    carry the data into every later request.
    """

    def __init__(self, *args, redactor: Redactor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._redactor = redactor

    def _redacted(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        result = []
        for message in messages:
            try:
                content, changed = self._redactor.redact_content(message.content)
            except Exception:
                logger.warning("PII redaction failed before summarizing; using the original", exc_info=True)
                content, changed = message.content, False
            result.append(message.model_copy(update={"content": content}) if changed else message)
        return result

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        return super()._create_summary(self._redacted(messages_to_summarize))

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        return await super()._acreate_summary(self._redacted(messages_to_summarize))
