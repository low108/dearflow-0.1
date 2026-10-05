"""Shared test doubles."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from dearflow.models import ModelFactory


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test in an empty directory so the repo's real skills/ never leak in."""
    monkeypatch.chdir(tmp_path)


class ToolCapableFakeModel(FakeMessagesListChatModel):
    """Fake chat model that accepts bind_tools and remembers which tools it got."""

    bound_tool_names: list = []

    def bind_tools(self, tools, *, tool_choice=None, **_kwargs):
        self.bound_tool_names = [getattr(t, "name", str(t)) for t in tools]
        return self

    def with_structured_output(self, *args, **kwargs):
        raise NotImplementedError(
            "This fake returns scripted message text, not structured output."
        )


class FakeModelFactory(ModelFactory):
    """Returns a preset model and records which names were requested."""

    def __init__(self, model):
        self.model = model
        self.requested: list[str] = []

    def create(self, model_name):
        self.requested.append(model_name)
        return self.model
