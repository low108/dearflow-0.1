"""The demo verifier must reject no-op or incomplete middleware evidence."""

import importlib
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT))
examine = importlib.import_module("middleware_demo").examine


def span(name, before=None, output=None):
    return {
        "id": "example",
        "name": "middleware." + name,
        "inputs": {"before": before or {"value": {}}},
        "outputs": output or {},
    }


def test_present_no_op_hooks_are_not_claimed_as_demonstrated():
    spans = [
        span(name, output={"state_update": {"value": None}, "calls": []})
        for name in (
            "MemoryMiddleware.before_agent",
            "TitleMiddleware.after_agent",
            "SubagentLimitMiddleware.after_model",
            "LoopDetectionMiddleware.after_model",
            "TokenBudgetMiddleware.after_model",
            "SummarizationMiddleware.before_model",
        )
    ]
    assert examine(spans) == {}


def test_receipt_claim_requires_actual_response_boundary_change():
    from dearflow.receipts import RECEIPT_KEY

    msg = {"additional_kwargs": {RECEIPT_KEY: {"status": "success"}}}
    s = span(
        "ToolReceiptMiddleware.wrap_tool_call",
        output={"result": {"value": msg}, "calls": [{"response": {"value": msg}}]},
    )
    assert "receipt_stamped" not in examine([s])
    s["outputs"]["calls"][0]["response"]["value"] = {"content": "391"}
    assert "receipt_stamped" in examine([s])


def test_pii_proof_requires_boundary_marker_transition_and_placeholder():
    s = span(
        "PiiRedactionMiddleware.wrap_model_call",
        before={"redaction_applied": True},
        output={
            "calls": [
                {
                    "request": {
                        "redaction_applied": True,
                        "value": {"messages": [{"content": "[EMAIL_test]"}]},
                    }
                }
            ]
        },
    )
    assert "pii_masked" not in examine([s])
    s["outputs"]["calls"][0]["request"]["redaction_applied"] = False
    assert "pii_masked" in examine([s])


def test_clarification_pass_through_does_not_count_as_interception():
    s = span(
        "ClarificationMiddleware.wrap_tool_call",
        before={"value": {"tool_call": {"name": "ask_clarification"}}},
        output={"handler_calls": 1, "result": {"value": {"content": "question"}}},
    )
    assert "clarification_intercepted" not in examine([s])
    s["outputs"]["handler_calls"] = 0
    assert "clarification_intercepted" in examine([s])


def test_remote_verification_waits_for_root_and_model_token_indexing(monkeypatch):
    from types import SimpleNamespace

    module = importlib.import_module("middleware_demo")
    monkeypatch.setattr(module, "wait_for_all_tracers", lambda: None)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    snapshot = SimpleNamespace(
        id="snapshot",
        inputs={"before": {}},
        outputs={"calls": []},
        parent_run_id="native",
        run_type="chain",
        total_tokens=0,
    )
    root = SimpleNamespace(
        id="root",
        inputs={},
        outputs={},
        parent_run_id=None,
        run_type="chain",
        total_tokens=7,
        end_time=True,
    )
    model = SimpleNamespace(
        id="model",
        inputs={},
        outputs={},
        parent_run_id="snapshot",
        run_type="llm",
        total_tokens=7,
    )

    class Client:
        count = 0

        def flush(self):
            pass

        def list_runs(self, **kwargs):
            self.count += 1
            return [snapshot] if self.count == 1 else [root, snapshot, model]

        def get_run_url(self, **kwargs):
            return "https://smith.langchain.com/example"

    demo = object.__new__(module.Demo)
    demo.client, demo.project = Client(), "test"
    record = {"run_id": "root", "usage": {"total_tokens": 7}}
    demo.verify_remote(
        record,
        [
            {
                "id": "snapshot",
                "inputs": {"before": {}},
                "outputs": {"calls": []},
                "parent": "native",
            }
        ],
    )
    assert demo.client.count == 2
    assert record["langsmith_status"] == "verified"
    assert record["tokens_match"] is True


def test_remote_verification_allows_sdk_cursor_pagination(monkeypatch):
    """A total trace cap must not be sent as the API's per-page limit."""
    from types import SimpleNamespace

    module = importlib.import_module("middleware_demo")
    monkeypatch.setattr(module, "wait_for_all_tracers", lambda: None)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    root = SimpleNamespace(
        id="root",
        inputs={},
        outputs={},
        parent_run_id=None,
        run_type="chain",
        total_tokens=7,
        end_time=True,
    )
    model = SimpleNamespace(
        id="model",
        inputs={},
        outputs={},
        parent_run_id="root",
        run_type="llm",
        total_tokens=7,
    )

    class Client:
        def flush(self):
            pass

        def list_runs(self, **kwargs):
            assert "limit" not in kwargs, "let the SDK follow all server-sized pages"
            yield root
            yield model

        def get_run_url(self, **kwargs):
            return "https://smith.langchain.com/example"

    demo = object.__new__(module.Demo)
    demo.client, demo.project = Client(), "test"
    record = {
        "run_id": "root",
        "usage": {"total_tokens": 7},
        "export_error": "OldTransientError",
    }
    demo.verify_remote(record, [])
    assert record["langsmith_status"] == "verified"
    assert "export_error" not in record


def test_remote_verification_reports_http_status_without_exception_payload(monkeypatch):
    import requests
    from langsmith.utils import LangSmithError

    module = importlib.import_module("middleware_demo")
    monkeypatch.setattr(module, "wait_for_all_tracers", lambda: None)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    class Client:
        def flush(self):
            pass

        def list_runs(self, **kwargs):
            response = requests.Response()
            response.status_code = 422
            try:
                raise requests.HTTPError(
                    "Secret request: sk-private-example", response=response
                )
            except requests.HTTPError:
                raise LangSmithError("Server rejected sk-private-example")

    demo = object.__new__(module.Demo)
    demo.client, demo.project = Client(), "test"
    record = {"run_id": "root", "usage": {"total_tokens": 7}}
    demo.verify_remote(record, [])
    assert record["export_error"] == "LangSmithError"
    assert record["export_http_status"] == 422
    assert "sk-private-example" not in str(record)


def test_verify_only_reuses_saved_evidence_without_model_credentials(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    module = importlib.import_module("middleware_demo")
    monkeypatch.setattr(module, "wait_for_all_tracers", lambda: None)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-key")
    data = {
        "model": "existing-model",
        "project": "existing-project",
        "synthetic_user": "existing-user",
        "observed_middleware": ["TitleMiddleware"],
        "scenarios": [
            {
                "scenario": "example",
                "run_id": "root",
                "passed": True,
                "usage": {"total_tokens": 7},
            }
        ],
    }
    (tmp_path / "results.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "example.snapshots.json").write_text("[]", encoding="utf-8")
    root = SimpleNamespace(
        id="root",
        inputs={},
        outputs={},
        parent_run_id=None,
        run_type="chain",
        total_tokens=7,
        end_time=True,
    )
    model = SimpleNamespace(
        id="model",
        inputs={},
        outputs={},
        parent_run_id="root",
        run_type="llm",
        total_tokens=7,
    )

    class Client:
        def flush(self):
            pass

        def list_runs(self, **kwargs):
            assert kwargs["project_name"] == "existing-project"
            return [root, model]

        def get_run_url(self, **kwargs):
            return "https://smith.langchain.com/example"

        def close(self):
            pass

    monkeypatch.setattr(module, "Client", Client)
    monkeypatch.setattr(
        module,
        "build_service",
        lambda *_args, **_kwargs: pytest.fail("verify-only must not call a provider"),
    )
    monkeypatch.setattr(
        sys, "argv", ["middleware_demo.py", "--verify-only", str(tmp_path)]
    )
    assert module.main() == 0
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["model"] == "existing-model"
    assert saved["scenarios"][0]["langsmith_status"] == "verified"
    assert (tmp_path / "report.md").exists()


@pytest.mark.parametrize(
    ("before", "after", "expected_pass"),
    [
        (
            "Prefers green notebooks and concise bullet points.",
            "Prefers green notebooks and concise bullet points.",
            False,
        ),
        ("", "Favorite notebook colour is green.", False),
        ("", "Prefers concise bullet points.", False),
        ("", "Likes green notebooks. Prefers short bulleted responses.", True),
        (
            "",
            "Favourite note book color: green. Likes succinct bullet-point answers.",
            True,
        ),
    ],
)
def test_memory_learn_requires_newly_persisted_both_preferences(
    tmp_path, monkeypatch, before, after, expected_pass
):
    from types import SimpleNamespace

    module = importlib.import_module("middleware_demo")
    current = [before]
    memory = SimpleNamespace(context=lambda user: current[0])
    factory = SimpleNamespace(
        memory=memory, flush_memory=lambda: current.__setitem__(0, after)
    )
    graph = SimpleNamespace(with_config=lambda **kwargs: graph)
    result = SimpleNamespace(
        run_id="11111111-1111-4111-8111-111111111111",
        thread_id="example",
        text="Acknowledged",
        usage={"total_tokens": 7},
        state={},
    )
    service = SimpleNamespace(
        _agent=SimpleNamespace(graph=graph),
        invoke=lambda *args, **kwargs: result,
        receipts=lambda thread: [],
    )
    monkeypatch.setattr(
        module,
        "build_service",
        lambda *args, **kwargs: (factory, service, SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(module, "with_overrides", lambda config, **kwargs: config)
    demo = object.__new__(module.Demo)
    demo.config, demo.folder, demo.prefix, demo.user = (
        object(),
        tmp_path,
        "suite",
        "synthetic-user",
    )
    demo.redactor = SimpleNamespace(redact=lambda text: text)
    demo.results, demo.observed = [], set()
    demo.verify_remote, demo.write = lambda *args: None, lambda: None
    demo.run(
        "memory-learn",
        prompt="Remember my preferences",
        expected=["memory_persisted", "memory_preferences_persisted"],
    )
    record = demo.results[0]
    assert record["passed"] is expected_pass
    assert record["memory_before"] == before
    assert record["memory_after"] == after
    if before == after:
        assert "memory_persisted" not in record["checks"]


@pytest.mark.parametrize(
    ("mode", "expected_names"),
    [
        ("--memory-only", ["memory-learn", "full-flow"]),
        ("--memory-smoke", ["memory-learn", "memory-recall"]),
    ],
)
def test_memory_only_runs_learning_and_injection_in_fresh_folder(
    tmp_path, monkeypatch, mode, expected_names
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    module = importlib.import_module("middleware_demo")
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LANGSMITH_API_KEY"):
        monkeypatch.setenv(key, "fake-test-value")
    yes = SimpleNamespace(enabled=True)
    config = SimpleNamespace(
        plan_mode=True,
        memory=yes,
        pii_redaction=yes,
        token_budget=yes,
        subagents=yes,
        loop_detection=yes,
        context=SimpleNamespace(date_reminder=True, summarization=yes, title=yes),
        verification=SimpleNamespace(receipts_enabled=True),
        observability=SimpleNamespace(middleware_snapshots=True, capture_content=True),
    )
    monkeypatch.setattr(module.AppConfig, "from_yaml", lambda _: config)
    monkeypatch.setattr(
        module,
        "Client",
        lambda: SimpleNamespace(flush=lambda: None, close=lambda: None),
    )
    monkeypatch.setattr(module, "tracing_context", lambda **kwargs: nullcontext())
    names = []

    class Demo:
        def __init__(self, cfg, folder, client, project):
            assert folder.is_dir()
            assert not (folder / "memory").exists()
            self.results = []

        def run(self, name, **kwargs):
            names.append(name)
            if name == "memory-learn":
                assert "memory_preferences_persisted" in kwargs["expected"]
            if mode == "--memory-smoke":
                assert kwargs["forced_tool"] == "none"
                if name == "memory-recall":
                    assert "memory_preferences_injected" in kwargs["expected"]
                    assert "memory_preferences_recalled" in kwargs["expected"]
                    assert not any(
                        word in kwargs["prompt"].lower()
                        for word in ("green", "concise", "bullet")
                    )
            self.results.append(
                {"passed": True, "langsmith_status": "verified", "tokens_match": True}
            )

    monkeypatch.setattr(module, "Demo", Demo)
    output = tmp_path / "fresh"
    monkeypatch.setattr(
        sys, "argv", ["middleware_demo.py", mode, "--output", str(output)]
    )
    assert module.main() == 0
    assert names == expected_names
    with pytest.raises(FileExistsError):
        module.main()


def test_forced_scenario_tool_does_not_override_structured_memory_schema():
    from pydantic import BaseModel
    from middleware_demo_helpers import ForcedCalculatorChatOpenAI

    class MemoryFacts(BaseModel):
        facts: list[str]

    model = ForcedCalculatorChatOpenAI(
        model="test", api_key="not-a-real-key", fixture_tool_choice="none"
    )
    memory = model.bind_tools([MemoryFacts], tool_choice="MemoryFacts")
    assert memory.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "MemoryFacts"},
    }
    assert memory.kwargs["tools"][0]["function"]["name"] == "MemoryFacts"
    unnamed_memory = model.bind_tools([MemoryFacts])
    assert "tool_choice" not in unnamed_memory.kwargs
    scenario = model.bind_tools(
        [
            {
                "type": "function",
                "function": {"name": "calculator", "parameters": {"type": "object"}},
            }
        ],
        tool_choice=None,
    )
    assert scenario.kwargs["tool_choice"] == "none"
    assert scenario.kwargs["parallel_tool_calls"] is False


def test_memory_diagnostics_capture_only_flush_calls_and_redact_replies():
    import json
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from dearflow.pii import DETECTORS, Redactor

    module = importlib.import_module("middleware_demo")
    recorder = module.MemoryEvidence(Redactor(DETECTORS, "synthetic-demo-secret"))
    recorder.on_chat_model_start({}, [], run_id="lead")
    assert recorder.attempts == {}
    recorder.enabled = True
    recorder.on_chat_model_start({}, [], run_id="first")
    reply = AIMessage(
        content="demo.person@example.com",
        invalid_tool_calls=[
            {"name": "MemoryFacts", "args": "bad json", "id": "call", "error": "parse"}
        ],
    )
    recorder.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=reply)]]), run_id="first"
    )
    recorder.on_chat_model_start({}, [], run_id="retry")
    recorder.on_llm_error(
        ValueError("Private request includes demo.person@example.com"), run_id="retry"
    )
    saved = json.dumps(recorder.attempts)
    assert "demo.person@example.com" not in saved
    assert "Private request" not in saved
    assert "[EMAIL_" in saved
    assert "invalid_tool_calls" in saved
    assert recorder.attempts["first"]["status"] == "model_completed"
    assert recorder.attempts["retry"] == {
        "run_id": "retry",
        "attempt": 2,
        "status": "model_error",
        "error": "ValueError",
    }


def test_recording_factory_callback_survives_structured_extraction_binding():
    import json
    from types import SimpleNamespace
    import httpx
    from langchain_openai import ChatOpenAI
    from dearflow.memory import Exchange, FactExtractor
    from dearflow.pii import DETECTORS, Redactor
    from middleware_demo_helpers import RecordingModelFactory

    module = importlib.import_module("middleware_demo")
    recorder = module.MemoryEvidence(Redactor(DETECTORS, "synthetic-demo-secret"))
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        tool_name = body["tools"][0]["function"]["name"]
        return httpx.Response(
            200,
            json={
                "id": "offline-extraction",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "facts-call",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": '{"facts": []}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(
            model="test",
            api_key="offline-key",
            base_url="https://offline.invalid/v1",
            http_client=client,
            max_retries=0,
        )
        factory = RecordingModelFactory(
            SimpleNamespace(create=lambda _: model), recorder
        )
        extractor = FactExtractor(factory.create("openai:test"))
        recorder.enabled = True
        assert extractor.extract([Exchange("Hello")]) == []
    assert len(requests) == len(recorder.attempts) == 1
    attempt = next(iter(recorder.attempts.values()))
    assert attempt["status"] == "model_completed"
    reply = attempt["reply"]["value"][0]
    assert reply["tool_calls"][0]["args"] == {"facts": []}
    assert reply["usage"]["total_tokens"] == 30
