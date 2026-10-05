import json

import pytest
from langchain_core.messages import AIMessage

from dearflow.memory import Exchange, FactExtractor


# Exact malformed JSON returned by the live demo's first extraction.
MALFORMED_LIVE_REPLY = """```json
{"facts": [{"content": "Prefers the color green for notebooks", "category": "preference", "confidence": 1.0}, {"content": Prefers concise bullet points for communication", "category": "preference", "confidence": 1.0}, {"content": Prefers brief and organized responses", "category": "behavior", "confidence": 0.8}]}
```"""
VALID_REPLY = '{"facts": [{"content": "Prefers green notebooks", "category": "preference", "confidence": 0.95}]}'


class ExtractionModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)


def test_extraction_regenerates_live_malformed_json_once():
    model = ExtractionModel(MALFORMED_LIVE_REPLY, VALID_REPLY)

    facts = FactExtractor(model).extract(
        [Exchange("My favorite notebook color is green.")]
    )

    assert [fact.content for fact in facts] == ["Prefers green notebooks"]
    assert len(model.prompts) == 2
    assert "My favorite notebook color is green." in model.prompts[1]
    assert "previous extraction" in model.prompts[1].lower()


@pytest.mark.parametrize(
    "reply",
    [
        "[]",
        "{}",
        '{"facts": "wrong type"}',
        '{"facts": [{"nope": 1}]}',
        '{"facts": [{"content": null}]}',
        '{"facts": [{"content": " "}]}',
        '{"facts": [{"content": "Green notebooks", "confidence": 2}]}',
    ],
)
def test_extraction_retries_invalid_schema(reply):
    model = ExtractionModel(reply, VALID_REPLY)

    assert FactExtractor(model).extract([Exchange("I like green notebooks.")])
    assert len(model.prompts) == 2


@pytest.mark.parametrize(
    "reply",
    [
        '{"facts": []}',
        '{"facts": [{"content": "Might like green", "confidence": 0.2}]}',
    ],
)
def test_extraction_does_not_retry_when_no_facts_qualify(reply):
    model = ExtractionModel(reply)

    assert FactExtractor(model).extract([Exchange("Hello.")]) == []
    assert len(model.prompts) == 1


def test_extraction_stops_after_one_invalid_response_retry():
    model = ExtractionModel(MALFORMED_LIVE_REPLY, MALFORMED_LIVE_REPLY)

    assert FactExtractor(model).extract([Exchange("I like green notebooks.")]) == []
    assert len(model.prompts) == 2


def test_extraction_valid_response_does_not_retry():
    model = ExtractionModel(VALID_REPLY)

    assert FactExtractor(model).extract([Exchange("I like green notebooks.")])
    assert len(model.prompts) == 1


def test_extraction_does_not_retry_provider_errors():
    model = ExtractionModel(RuntimeError("Provider unavailable"))

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        FactExtractor(model).extract([Exchange("I like green notebooks.")])

    assert len(model.prompts) == 1


def test_extraction_prompt_provides_a_valid_json_example():
    model = ExtractionModel('{"facts": []}')
    FactExtractor(model).extract([Exchange("Hello.")])

    example = next(
        line.removeprefix("Return ONLY JSON: ")
        for line in model.prompts[0].splitlines()
        if line.startswith("Return ONLY JSON: ")
    )
    assert isinstance(json.loads(example)["facts"], list)
