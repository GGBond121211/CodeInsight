"""Tests for the OpenAI-compatible embedding adapter without network calls."""

from types import SimpleNamespace

import pytest

from codeinsight.domain.errors import ModelConfigurationError, ModelResponseError
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel


class _FakeEmbeddings:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _SequencedEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 1.0])
                for index, _ in enumerate(kwargs["input"])
            ],
            usage=SimpleNamespace(total_tokens=len(kwargs["input"]) * 2),
        )


def _fake_client(embeddings):
    return SimpleNamespace(embeddings=embeddings)


def test_embedding_configuration_requires_explicit_model() -> None:
    with pytest.raises(ModelConfigurationError, match="CODEINSIGHT_API_KEY"):
        OpenAIEmbeddingModel.from_environment({})
    with pytest.raises(ModelConfigurationError, match="CODEINSIGHT_EMBEDDING_MODEL"):
        OpenAIEmbeddingModel.from_environment({"CODEINSIGHT_API_KEY": "test-key"})


def test_embedding_configuration_can_use_a_separate_provider(monkeypatch) -> None:
    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("codeinsight.infrastructure.embeddings.OpenAI", _FakeOpenAI)

    model = OpenAIEmbeddingModel.from_environment(
        {
            "CODEINSIGHT_EMBEDDING_API_KEY": "embedding-key",
            "CODEINSIGHT_EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "CODEINSIGHT_EMBEDDING_MODEL": "qwen3.7-text-embedding",
        }
    )

    assert model.model == "qwen3.7-text-embedding"
    assert captured["api_key"] == "embedding-key"
    assert captured["base_url"] == "https://embedding.example/v1"


def test_embedding_adapter_maps_vectors_and_usage() -> None:
    embeddings = _FakeEmbeddings(
        SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ],
            usage=SimpleNamespace(total_tokens=7),
        )
    )
    model = OpenAIEmbeddingModel(client=_fake_client(embeddings), model="test-embedding")  # type: ignore[arg-type]

    result = model.embed(("first", "second"))

    assert result.model == "test-embedding"
    assert result.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert result.input_tokens == 7
    assert embeddings.calls == [{"model": "test-embedding", "input": ["first", "second"]}]


def test_embedding_adapter_rejects_incomplete_response() -> None:
    embeddings = _FakeEmbeddings(
        SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])],
            usage=None,
        )
    )
    model = OpenAIEmbeddingModel(client=_fake_client(embeddings), model="test-embedding")  # type: ignore[arg-type]

    with pytest.raises(ModelResponseError, match="count"):
        model.embed(("first", "second"))


def test_embedding_adapter_batches_large_repository_inputs_in_order() -> None:
    embeddings = _SequencedEmbeddings()
    model = OpenAIEmbeddingModel(  # type: ignore[arg-type]
        client=_fake_client(embeddings), model="test-embedding"
    )
    texts = tuple(f"chunk-{index}" for index in range(35))

    result = model.embed(texts)

    assert [len(call["input"]) for call in embeddings.calls] == [16, 16, 3]
    assert len(result.vectors) == 35
    assert result.input_tokens == 70
    assert result.vectors[0] == (0.0, 1.0)
    assert result.vectors[16] == (0.0, 1.0)
    assert result.vectors[32] == (0.0, 1.0)
