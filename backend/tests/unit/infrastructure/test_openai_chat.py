from types import SimpleNamespace

import pytest
from openai import OpenAIError

from codeinsight.application.context_assembler import ContextAssembler
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)
from codeinsight.infrastructure.openai_chat import OpenAIChatModel, parse_model_answer


class FakeCompletions:
    def __init__(self, *, error: OpenAIError | None = None) -> None:
        self.kwargs = None
        self.error = error

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=('{"outcome":"answered","answer":"Uses E1.","citations":["E1"]}')
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=6),
        )


def _fake_client(completions: FakeCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_parse_answered_json() -> None:
    result = parse_model_answer(
        '{"outcome":"answered","answer":"Uses checkout.","citations":["E1"]}',
        model="test-model",
        input_tokens=10,
        output_tokens=5,
    )

    assert result.outcome == "answered"
    assert result.evidence_ids == ("E1",)


def test_parse_answered_json_tolerates_markdown_wrapper() -> None:
    result = parse_model_answer(
        '```json\n{"outcome":"answered","answer":"Uses E1.","citations":["E1"]}\n```',
        model="test-model",
        input_tokens=10,
        output_tokens=5,
    )

    assert result.answer == "Uses E1."


def test_parse_deduplicates_evidence_ids() -> None:
    result = parse_model_answer(
        '{"outcome":"answered","answer":"Uses checkout.","citations":["E1","E1"]}',
        model="test-model",
        input_tokens=None,
        output_tokens=None,
    )

    assert result.evidence_ids == ("E1",)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"outcome":"unknown","answer":"x","citations":[]}',
        '{"outcome":"answered","answer":"","citations":["E1"]}',
        '{"outcome":"answered","answer":"x","citations":[]}',
        '{"outcome":"insufficient_evidence","answer":"x","citations":["E1"]}',
        '{"outcome":"answered","answer":"x","citations":["E1"],"extra":1}',
    ],
)
def test_invalid_model_payload_raises(content: str) -> None:
    with pytest.raises(ModelResponseError):
        parse_model_answer(content, model="test-model", input_tokens=None, output_tokens=None)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"CODEINSIGHT_MODEL": "test-model"}, "必须配置 CODEINSIGHT_API_KEY"),
        ({"CODEINSIGHT_API_KEY": "secret"}, "必须配置 CODEINSIGHT_MODEL"),
    ],
)
def test_missing_configuration_raises(environment: dict[str, str], message: str) -> None:
    with pytest.raises(ModelConfigurationError, match=message):
        OpenAIChatModel.from_environment(environment)


def test_environment_uses_project_gateway_base_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return _fake_client(FakeCompletions())

    monkeypatch.setattr("codeinsight.infrastructure.openai_chat.OpenAI", fake_openai)
    model = OpenAIChatModel.from_environment(
        {
            "CODEINSIGHT_API_KEY": "test-key",
            "CODEINSIGHT_MODEL": "test-model",
        }
    )

    assert model.model == "test-model"
    assert captured["base_url"] == "https://api.frontier-intelligence.tech/v1"


def test_environment_rejects_non_project_chat_endpoint() -> None:
    with pytest.raises(ModelConfigurationError, match="api.frontier-intelligence.tech/v1"):
        OpenAIChatModel.from_environment(
            {
                "CODEINSIGHT_API_KEY": "test-key",
                "CODEINSIGHT_MODEL": "test-model",
                "CODEINSIGHT_BASE_URL": "https://api.deepseek.com/v1",
            }
        )


def test_generate_maps_messages_and_usage() -> None:
    completions = FakeCompletions()
    model = OpenAIChatModel(client=_fake_client(completions), model="test-model")  # type: ignore[arg-type]

    result = model.generate("system", "user")

    assert completions.kwargs == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 0,
    }
    assert result.input_tokens == 11
    assert result.output_tokens == 6


def test_complete_returns_raw_content_and_usage() -> None:
    completions = FakeCompletions()
    model = OpenAIChatModel(client=_fake_client(completions), model="test-model")  # type: ignore[arg-type]

    result = model.complete("system", "user")

    assert result.content.startswith('{"outcome":"answered"')
    assert result.model == "test-model"
    assert result.input_tokens == 11
    assert result.output_tokens == 6


def test_context_assembler_estimate_is_carried_with_completion() -> None:
    completions = FakeCompletions()
    model = OpenAIChatModel(
        client=_fake_client(completions),
        model="test-model",
        context_assembler=ContextAssembler(),
    )  # type: ignore[arg-type]

    result = model.complete("system", "user")

    assert result.estimated_input_tokens is not None
    assert result.estimated_input_tokens > 0
    assert "[user_code_task]" in completions.kwargs["messages"][1]["content"]


def test_sdk_error_becomes_safe_model_call_error() -> None:
    completions = FakeCompletions(error=OpenAIError("request included secret-value"))
    model = OpenAIChatModel(client=_fake_client(completions), model="test-model")  # type: ignore[arg-type]

    with pytest.raises(ModelCallError, match="^模型请求失败$") as raised:
        model.generate("system", "user")

    assert "secret-value" not in str(raised.value)
