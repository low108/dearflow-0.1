"""Turns a model identifier into a LangChain chat model."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


class ModelFactory:
    """Creates chat models for the single configured provider.

    The provider (OpenAI-compatible, e.g. Ilmu) is selected by the
    `provider:model` identifier and the usual OPENAI_API_KEY / OPENAI_BASE_URL
    environment variables. Tests subclass or replace this to return fakes.
    """

    def create(self, model_name: str) -> BaseChatModel:
        return init_chat_model(model_name)
