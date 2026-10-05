import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from dearflow.memory import Exchange, FactExtractor


VALID_FACTS = {
    "facts": [
        {
            "content": "Prefers green notebooks",
            "category": "preference",
            "confidence": 0.9,
        },
        {
            "content": "Prefers concise bullet points",
            "category": "preference",
            "confidence": 0.9,
        },
    ]
}
# Exact second invalid extraction returned by the real provider, as tool arguments.
INVALID_FACTS = '{"facts":[{"content":"Prefers green notebook color","category":"preference","confidence":0.9},{"content": Prefers concise bullet points for communication,"category":"behavior","confidence":0.9}]}'


def chat_model(*arguments):
    requests = []
    replies = iter(arguments)

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
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "facts-call",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": next(replies),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )

    model = ChatOpenAI(
        model="test-model",
        api_key="offline-test-key",
        base_url="https://offline.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    )
    return model, requests


def test_openai_extraction_forces_a_validated_facts_tool():
    model, requests = chat_model(json.dumps(VALID_FACTS))

    facts = FactExtractor(model).extract(
        [Exchange("I like green notebooks and concise bullet points.")]
    )

    assert [fact.content for fact in facts] == [
        item["content"] for item in VALID_FACTS["facts"]
    ]
    assert len(requests) == 1
    tool = requests[0]["tools"][0]["function"]
    assert requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": tool["name"]},
    }
    assert "facts" in tool["parameters"]["properties"]
    assert requests[0]["parallel_tool_calls"] is False


def test_invalid_structured_arguments_retry_once():
    model, requests = chat_model(INVALID_FACTS, json.dumps(VALID_FACTS))

    assert len(FactExtractor(model).extract([Exchange("I like green notebooks.")])) == 2
    assert len(requests) == 2


def test_repeated_invalid_structured_arguments_stop_after_one_retry():
    model, requests = chat_model(INVALID_FACTS, INVALID_FACTS)

    assert FactExtractor(model).extract([Exchange("I like green notebooks.")]) == []
    assert len(requests) == 2


def test_empty_structured_facts_are_a_successful_extraction():
    model, requests = chat_model('{"facts": []}')

    assert FactExtractor(model).extract([Exchange("Hello.")]) == []
    assert len(requests) == 1


@pytest.mark.parametrize(
    "item",
    [
        {"content": " ", "category": "preference", "confidence": 0.9},
        {"content": "Green notebooks", "category": "made_up", "confidence": 0.9},
        {"content": "Green notebooks", "category": "preference", "confidence": 2},
    ],
)
def test_structured_schema_validation_triggers_retry(item):
    model, requests = chat_model(json.dumps({"facts": [item]}), json.dumps(VALID_FACTS))

    assert len(FactExtractor(model).extract([Exchange("I like green notebooks.")])) == 2
    assert len(requests) == 2


def test_structured_low_confidence_facts_do_not_retry():
    low_confidence = {
        "facts": [
            {
                "content": "Maybe likes green notebooks",
                "category": "preference",
                "confidence": 0.2,
            }
        ]
    }
    model, requests = chat_model(json.dumps(low_confidence))

    assert FactExtractor(model).extract([Exchange("Green notebooks?")]) == []
    assert len(requests) == 1


class UnsupportedModel:
    def __init__(self):
        self.capability_checks = 0
        self.calls = 0

    def with_structured_output(self, *args, **kwargs):
        self.capability_checks += 1
        raise NotImplementedError

    def invoke(self, prompt):
        self.calls += 1
        return AIMessage(content=json.dumps(VALID_FACTS))


def test_explicitly_unsupported_models_keep_json_extraction():
    model = UnsupportedModel()
    extractor = FactExtractor(model)

    assert len(extractor.extract([Exchange("I like green.")])) == 2
    assert len(extractor.extract([Exchange("I like green.")])) == 2
    assert model.capability_checks == 1
    assert model.calls == 2


def test_unexpected_structured_setup_errors_are_not_silently_downgraded():
    class BrokenModel(UnsupportedModel):
        def with_structured_output(self, *args, **kwargs):
            raise ValueError("Invalid configuration")

    with pytest.raises(ValueError, match="Invalid configuration"):
        FactExtractor(BrokenModel())
