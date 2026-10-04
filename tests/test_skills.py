import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig
from dearflow.prompts import PromptBuilder
from dearflow.skills import Skill, SkillCatalog, SkillParseError, create_load_skill_tool


def write_skill(root, folder, name, description="Does things.", body="Step 1."):
    path = root / folder / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n", encoding="utf-8")
    return path


def test_skill_parses_frontmatter_and_body(tmp_path):
    skill = Skill.from_file(write_skill(tmp_path, "a", "my-skill", "Use for X.", "Do this."))

    assert (skill.name, skill.description, skill.body) == ("my-skill", "Use for X.", "Do this.")


@pytest.mark.parametrize("content", ["no frontmatter", "---\nname: Bad Name\ndescription: x\n---\nbody", "---\nname: ok\n---\nbody"])
def test_invalid_skill_files_raise(tmp_path, content):
    path = tmp_path / "SKILL.md"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SkillParseError):
        Skill.from_file(path)


def test_catalog_discovers_nested_skills_and_skips_invalid(tmp_path):
    write_skill(tmp_path, "public/one", "one")
    write_skill(tmp_path, "custom/two", "two")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("garbage", encoding="utf-8")

    catalog = SkillCatalog.discover(tmp_path)

    assert sorted(catalog.names()) == ["one", "two"]
    assert SkillCatalog.discover(tmp_path / "missing").names() == []


def test_prompt_lists_skills_but_not_their_bodies(tmp_path):
    write_skill(tmp_path, "a", "alpha", "Use for alpha.", "SECRET BODY")
    prompt = PromptBuilder("Base.", SkillCatalog.discover(tmp_path)).build()

    assert prompt.startswith("Base.")
    assert "- alpha: Use for alpha." in prompt
    assert "SECRET BODY" not in prompt
    assert "<skills>" not in PromptBuilder("Base.").build()


def test_load_skill_tool_returns_body_or_lists_available(tmp_path):
    write_skill(tmp_path, "a", "alpha", body="Alpha steps.")
    tool = create_load_skill_tool(SkillCatalog.discover(tmp_path))

    assert "Alpha steps." in tool.invoke({"name": "alpha"})
    with pytest.raises(ValueError, match="Available skills: alpha"):
        tool.invoke({"name": "nope"})


def test_agent_sees_skill_index_and_loads_a_skill_on_demand(tmp_path):
    write_skill(tmp_path, "a", "alpha", "Use for alpha.", "Alpha steps.")
    config = AppConfig(default_model="m", skills={"path": str(tmp_path)})
    call = {"name": "load_skill", "args": {"name": "alpha"}, "id": "c1", "type": "tool_call"}
    model = ToolCapableFakeModel(responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="Done.")])
    seen_prompts: list[str] = []
    original = model._generate

    def spy(messages, *args, **kwargs):
        seen_prompts.append(next(m.content for m in messages if isinstance(m, SystemMessage)))
        return original(messages, *args, **kwargs)

    object.__setattr__(model, "_generate", spy)

    agent = LeadAgentFactory(config, model_factory=FakeModelFactory(model)).build()
    result = agent.graph.invoke({"messages": [{"role": "user", "content": "help"}]})

    assert "- alpha: Use for alpha." in seen_prompts[0]
    assert any(isinstance(m, ToolMessage) and "Alpha steps." in m.content for m in result["messages"])


def test_disabled_skills_add_no_tool_or_prompt_section(tmp_path):
    write_skill(tmp_path, "a", "alpha")
    config = AppConfig(default_model="m", skills={"enabled": False, "path": str(tmp_path)})
    model = ToolCapableFakeModel(responses=[AIMessage(content="hi")])

    agent = LeadAgentFactory(config, model_factory=FakeModelFactory(model)).build()
    agent.graph.invoke({"messages": [{"role": "user", "content": "x"}]})

    assert "load_skill" not in model.bound_tool_names
