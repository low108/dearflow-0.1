import re
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from conftest import FakeModelFactory, ToolCapableFakeModel
from dearflow.agents import LeadAgentFactory
from dearflow.config import AppConfig, PiiRedactionConfig
from dearflow.mcp import McpToolLoader
from dearflow.memory import Exchange, MemoryManager
from dearflow.middlewares import MemoryMiddleware, MiddlewareChain, RedactingSummarizationMiddleware, TitleMiddleware
from dearflow.persistence import Persistence
from dearflow.pii import DETECTORS, make_redactor, placeholder
from dearflow.pii.redactor import cn_resident_id_valid, cpf_valid, luhn_valid
from dearflow.service import RunService
from dearflow.tools import ToolRegistry

SECRET = "a-long-enough-secret-key"
EMAIL = "ann@example.com"
TOKEN = re.compile(r"\[[A-Z_]+_[a-z]{27}\]")


def redactor(**overrides):
    return make_redactor(PiiRedactionConfig(enabled=True, token_secret=SECRET, **overrides))


def cn_id(body="11010519491231002"):
    """A Chinese resident ID with a correct check character."""
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    return body + "10X98765432"[sum(int(d) * w for d, w in zip(body, weights)) % 11]


# --- detectors ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, category",
    [
        (f"write to {EMAIL} today", "EMAIL"),
        ("key sk-abcdefghijklmnopqrstuvwxyz1234", "API_KEY"),
        ("aws AKIAIOSFODNN7EXAMPLE", "API_KEY"),
        ("gh ghp_" + "a" * 36, "API_KEY"),
        ("slack xoxb-1234567890-abcdef", "API_KEY"),
        ("card 4111 1111 1111 1111 ok", "CREDIT_CARD"),
        ("card 4111111111111111", "CREDIT_CARD"),
        ("call +60 12-345 6789", "PHONE"),
        ("call 13812345678", "PHONE"),
        ("call (415) 555-2671", "PHONE"),
        (f"身份证{cn_id()}", "NATIONAL_ID"),
        ("cpf 529.982.247-25", "NATIONAL_ID"),
        ("cuit 20-12345678-3", "NATIONAL_ID"),
        ("rfc ABCD800101XY1", "NATIONAL_ID"),
    ],
)
def test_each_kind_of_personal_data_is_replaced(text, category):
    result = redactor().redact(text)

    assert f"[{category}_" in result and TOKEN.search(result)
    assert not any(ch.isdigit() for ch in TOKEN.search(result).group(0))  # tokens carry no digit runs


@pytest.mark.parametrize(
    "text",
    [
        "card 4111 1111 1111 1112",  # fails Luhn
        "order 1234567890123456",  # 16 digits, fails Luhn
        f"id {cn_id()[:-1]}0" if cn_id()[-1] != "0" else "id 110105194912310021",  # wrong check character
        "cpf 111.111.111-11",  # repeated digits
        "call 12345",  # too short for a phone
        "the quick brown fox",
    ],
)
def test_lookalikes_that_fail_their_checksum_are_left_alone(text):
    assert redactor().redact(text) == text


def test_checksum_helpers():
    assert luhn_valid("4111 1111 1111 1111") and not luhn_valid("4111 1111 1111 1112")
    assert cn_resident_id_valid(cn_id()) and not cn_resident_id_valid("110105194912310022")
    assert cpf_valid("529.982.247-25") and not cpf_valid("529.982.247-26")


def test_detector_order_national_id_wins_over_card():
    # Find a valid resident ID whose 18 digits ALSO pass Luhn. It must be claimed as an ID, not eaten as a card.
    for serial in range(1000):
        candidate = cn_id(f"11010519491231{serial:03d}")
        if candidate[-1].isdigit() and cn_resident_id_valid(candidate) and luhn_valid(candidate):
            assert redactor().redact(candidate).startswith("[NATIONAL_ID_")
            return
    pytest.fail("expected at least one ID that is both checksum-valid and Luhn-valid in 1000 tries")


def test_detectors_run_in_the_documented_order():
    assert [d.name for d in DETECTORS] == ["email", "api_key", "national_id", "credit_card", "phone"]


def test_detectors_can_be_switched_off_one_by_one():
    text = f"{EMAIL} and 13812345678"

    only_phone = redactor(redact_email=False).redact(text)
    assert EMAIL in only_phone and "[PHONE_" in only_phone
    assert redactor(redact_email=False, redact_phone=False, redact_api_key=False, redact_credit_card=False, redact_national_id=False).redact(text) == text


# --- placeholders ---------------------------------------------------------------------------


def test_placeholder_is_stable_keyed_and_value_derived():
    one = placeholder("email", EMAIL, SECRET)

    assert one == placeholder("email", EMAIL, SECRET)  # same value -> same token, anywhere, any time
    assert one != placeholder("email", "bob@example.com", SECRET)
    assert one != placeholder("email", EMAIL, "another-long-enough-secret")  # different deployment, different token
    assert one != placeholder("phone", EMAIL, SECRET)
    assert re.fullmatch(r"\[EMAIL_[a-z]{27}\]", one)
    assert redactor().redact(f"{EMAIL} ... {EMAIL}").count(one) == 2  # the model can tell it is the same address


def test_secret_rules():
    with pytest.raises(ValueError, match="token_secret"):
        PiiRedactionConfig(enabled=True)
    with pytest.raises(ValueError, match="16"):
        make_redactor(PiiRedactionConfig(enabled=True, token_secret="short"))
    assert make_redactor(PiiRedactionConfig()) is None  # off: no redactor at all
    with pytest.raises(ValueError):
        placeholder("email", EMAIL, "")


def test_redact_content_handles_blocks_and_never_mutates_its_input():
    blocks = [{"type": "text", "text": f"mail {EMAIL}"}, {"type": "image_url", "image_url": {"url": "http://x"}}, "plain"]
    before = [dict(b) if isinstance(b, dict) else b for b in blocks]
    result, changed = redactor().redact_content(blocks)

    assert changed and "[EMAIL_" in result[0]["text"] and result[1:] == blocks[1:]
    assert blocks == before
    assert redactor().redact_content("nothing to hide") == ("nothing to hide", False)
    assert redactor().redact_content(None) == (None, False)


# --- what the model sees -----------------------------------------------------------------------


class Recording(ToolCapableFakeModel):
    requests: list = []

    def _generate(self, messages, *args, **kwargs):
        self.requests.append([(m.type, m.content if isinstance(m.content, str) else str(m.content)) for m in messages])
        return super()._generate(messages, *args, **kwargs)


@tool
def outside_tool() -> str:
    """Pretends to fetch a web page."""
    return f"Contact: {EMAIL}"


@tool
def local_tool() -> str:
    """Reads something local."""
    return f"Contact: {EMAIL}"


def call(name, id="c1", **args):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def build(responses, *, tools=(), pii=None, **config):
    model = Recording(responses=responses, requests=[])
    pii = pii or {"enabled": True, "token_secret": SECRET}
    app = AppConfig(default_model="fake", pii_redaction=pii, **config)
    persistence = Persistence.in_memory()
    factory = LeadAgentFactory(
        app,
        model_factory=FakeModelFactory(model),
        tool_registry=ToolRegistry(app, extra_tools=list(tools)),
        checkpointer=persistence.checkpointer,
    )
    return RunService(factory.build(), persistence), persistence, model, factory


def test_the_model_sees_placeholders_but_the_conversation_keeps_the_original():
    service, persistence, model, _ = build([AIMessage(content="Noted")])
    result = service.invoke(f"My email is {EMAIL}", thread_id="t1")

    sent = " ".join(content for request in model.requests for kind, content in request if kind == "human")
    assert EMAIL not in sent and "[EMAIL_" in sent
    assert any(m.content == f"My email is {EMAIL}" for m in result.state["messages"])  # the user's original is stored
    assert [m.content for m in service.history("t1")][0] == f"My email is {EMAIL}"  # and shown back to them


def test_every_turn_redacts_the_whole_history_the_same_way():
    service, _, model, _ = build([AIMessage(content="one"), AIMessage(content="two")])
    service.invoke(f"mail {EMAIL}", thread_id="t1")
    service.invoke("and again", thread_id="t1")

    first_token = TOKEN.search(str(model.requests[0])).group(0)
    assert first_token in str(model.requests[1])  # same token on the second call
    assert EMAIL not in str(model.requests)


def test_only_outside_tools_are_redacted():
    tools = [outside_tool, local_tool]
    service, _, model, _ = build(
        [
            AIMessage(content="", tool_calls=[call("outside_tool", "o1"), call("local_tool", "l1")]),
            AIMessage(content="done"),
        ],
        tools=tools,
        pii={"enabled": True, "token_secret": SECRET, "tool_names": ["outside_tool"]},
    )
    result = service.invoke("fetch", thread_id="t1")

    results = {m.name: m.content for m in result.state["messages"] if isinstance(m, ToolMessage)}
    assert "[EMAIL_" in results["outside_tool"] and EMAIL not in results["outside_tool"]
    assert results["local_tool"] == f"Contact: {EMAIL}"  # left alone: it is the user's own data


def test_mcp_tool_results_are_redacted_automatically(monkeypatch):
    server = str(Path(__file__).parent / "support" / "mcp_demo_server.py")
    mcp = {"enabled": True, "servers": {"demo": {"command": sys.executable, "args": [server], "env": {"DEMO_SECRET": EMAIL}}}}
    tools = McpToolLoader(AppConfig(default_model="x", mcp=mcp).mcp).tools()
    assert all(t.metadata["dearflow_source"] == "mcp" for t in tools)

    model = Recording(responses=[AIMessage(content="", tool_calls=[call("demo_secret")]), AIMessage(content="ok")], requests=[])
    app = AppConfig(default_model="fake", mcp=mcp, pii_redaction={"enabled": True, "token_secret": SECRET})
    factory = LeadAgentFactory(app, model_factory=FakeModelFactory(model))
    factory.build().graph.invoke({"messages": [{"role": "user", "content": "go"}]})

    second_call = str(model.requests[1])
    assert "[EMAIL_" in second_call and EMAIL not in second_call


def test_subagents_are_covered_too():
    service, _, model, _ = build(
        [
            AIMessage(content="", tool_calls=[call("task", prompt=f"Write to {EMAIL}", subagent_type="general-purpose")]),
            AIMessage(content="sub done"),
            AIMessage(content="final"),
        ],
        subagents={"enabled": True},
    )
    service.invoke("delegate", thread_id="t1")

    subagent_call = model.requests[1]  # the subagent's first model call sees the task prompt as its user message
    assert any("[EMAIL_" in content and EMAIL not in content for kind, content in subagent_call if kind == "human")


def test_a_failure_in_redaction_lets_the_text_through_and_logs(monkeypatch, caplog):
    service, _, model, factory = build([AIMessage(content="ok")])
    monkeypatch.setattr(factory._redactor, "redact_content", lambda content: (_ for _ in ()).throw(RuntimeError("boom")))
    result = service.invoke(f"mail {EMAIL}", thread_id="t1")

    assert result.text == "ok" and "PII redaction failed" in caplog.text
    assert EMAIL in str(model.requests)  # fails open, like DeerFlow


def test_redaction_is_off_by_default():
    model = Recording(responses=[AIMessage(content="ok")], requests=[])
    factory = LeadAgentFactory(AppConfig(default_model="fake"), model_factory=FakeModelFactory(model))
    factory.build().graph.invoke({"messages": [{"role": "user", "content": f"mail {EMAIL}"}]})

    assert EMAIL in str(model.requests) and factory._redactor is None


# --- the other places text travels ---------------------------------------------------------------------


def test_memory_queue_never_gets_personal_data():
    queued = []
    manager = MemoryManager.__new__(MemoryManager)
    manager.enqueue = lambda user, exchanges: queued.append(exchanges[0])
    middleware = MemoryMiddleware(manager, redact=redactor().redact)
    state = {"messages": [HumanMessage(content=f"I am {EMAIL}", id="u1"), AIMessage(content=f"Hi {EMAIL}")]}
    middleware.after_agent(state, None)

    assert isinstance(queued[0], Exchange) and EMAIL not in queued[0].user + queued[0].assistant
    assert "[EMAIL_" in queued[0].user and "[EMAIL_" in queued[0].assistant


def test_titles_are_redacted_before_they_are_shortened():
    long_prefix = "please help me with something quite long and detailed, my address is "
    state = {"messages": [HumanMessage(content=long_prefix + EMAIL, id="u1"), AIMessage(content="sure")]}
    plain = TitleMiddleware(max_chars=200).after_agent(state, None)["title"]
    safe = TitleMiddleware(max_chars=200, redact=redactor().redact).after_agent(state, None)["title"]

    assert EMAIL in plain and EMAIL not in safe and "[EMAIL_" in safe


def test_the_summary_model_only_sees_redacted_history():
    seen = []

    class Summarizer(ToolCapableFakeModel):
        def invoke(self, prompt, *args, **kwargs):
            seen.append(str(prompt))
            return AIMessage(content="summary")

    middleware = RedactingSummarizationMiddleware(Summarizer(responses=[]), redactor=redactor(), trigger=("tokens", 10), keep=("messages", 1))
    summary = middleware._create_summary([HumanMessage(content=f"my mail is {EMAIL}"), AIMessage(content="ok")])

    assert summary == "summary" and EMAIL not in seen[0] and "[EMAIL_" in seen[0]


def test_default_chain_puts_redaction_just_before_clarification():
    chain = MiddlewareChain.default(redactor=redactor())
    names = [type(m).__name__ for m in chain.build()]

    assert names[-2:] == ["PiiRedactionMiddleware", "ClarificationMiddleware"]
    assert "PiiRedactionMiddleware" not in [type(m).__name__ for m in MiddlewareChain.default().build()]


def test_secret_from_the_environment(monkeypatch):
    monkeypatch.setenv("PII_SECRET", SECRET)
    cfg = AppConfig(default_model="fake", pii_redaction={"enabled": True, "token_secret": "$PII_SECRET"})
    factory = LeadAgentFactory(cfg, model_factory=FakeModelFactory(Recording(responses=[], requests=[])))
    assert factory._redactor.redact(EMAIL) == placeholder("email", EMAIL, SECRET)

    monkeypatch.delenv("PII_SECRET")
    with pytest.raises(ValueError, match="PII_SECRET"):
        LeadAgentFactory(cfg, model_factory=FakeModelFactory(Recording(responses=[], requests=[])))
