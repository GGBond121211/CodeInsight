import json
from unittest.mock import patch

import pytest

from codeinsight.domain.errors import ModelResponseError
from codeinsight.infrastructure.reranker import (
    DEFAULT_RERANK_MODEL,
    OpenAITextReranker,
    _parse_results,
    _rerank_endpoint,
)


def test_endpoint_converts_bailian_embedding_path() -> None:
    assert _rerank_endpoint("https://example.com/compatible-mode/v1") == (
        "https://example.com/compatible-api/v1/reranks"
    )


def test_environment_uses_embedding_key_and_default_model() -> None:
    model = OpenAITextReranker.from_environment(
        {
            "CODEINSIGHT_EMBEDDING_API_KEY": "secret",
            "CODEINSIGHT_EMBEDDING_BASE_URL": "https://example.com/compatible-mode/v1",
        }
    )
    assert model.model == DEFAULT_RERANK_MODEL
    assert model.endpoint.endswith("/compatible-api/v1/reranks")


def test_parse_results_validates_index_and_score() -> None:
    result = _parse_results(
        {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]},
        document_count=2,
        top_n=2,
    )
    assert [(item.index, item.relevance_score) for item in result] == [(1, 0.9), (0, 0.2)]
    with pytest.raises(ModelResponseError, match="重复 index"):
        _parse_results(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
            document_count=2,
            top_n=2,
        )


def test_rerank_posts_top_level_bailian_payload() -> None:
    reranker = OpenAITextReranker(
        api_key="secret",
        endpoint="https://example.com/compatible-api/v1/reranks",
        model="qwen3.7-text-rerank",
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"results": [{"index": 1, "relevance_score": 0.9}]}
            ).encode()

    with patch("codeinsight.infrastructure.reranker.urlopen", return_value=Response()) as mocked:
        result = reranker.rerank("checkout", ("generic", "checkout implementation"), top_n=1)

    assert result[0].index == 1
    payload = json.loads(mocked.call_args.args[0].data.decode())
    assert payload["query"] == "checkout"
    assert payload["documents"] == ["generic", "checkout implementation"]
    assert payload["top_n"] == 1
    assert payload["return_documents"] is False
    assert reranker.last_call is not None
    assert reranker.last_call.document_count == 2
    assert reranker.last_call.status == "ok"


def test_rerank_does_not_silently_fallback_on_http_error() -> None:
    from urllib.error import HTTPError

    reranker = OpenAITextReranker(
        api_key="secret", endpoint="https://example.com/reranks", model="test"
    )
    error = HTTPError("https://example.com/reranks", 429, "rate limit", {}, None)
    with patch("codeinsight.infrastructure.reranker.urlopen", side_effect=error):
        with pytest.raises(RuntimeError, match="Rerank 请求失败"):
            reranker.rerank("q", ("d",), top_n=1)
