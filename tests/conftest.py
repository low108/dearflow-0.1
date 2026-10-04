"""Shared test doubles."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from dearflow.models import ModelFactory


class ToolCapableFakeModel(FakeMessagesListChatModel):
    """Fake chat model that accepts bind_tools (needed when an agent has tools)."""

    def bind_tools(self, _tools, *, tool_choice=None, **_kwargs):
        return self


class FakeModelFactory(ModelFactory):
    """Returns a preset model and records which names were requested."""

    def __init__(self, model):
        self.model = model
        self.requested: list[str] = []

    def create(self, model_name):
        self.requested.append(model_name)
        return self.model
