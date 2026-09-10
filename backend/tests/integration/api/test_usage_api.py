from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.domain.answer import ModelAnswer, ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.model_gateway import GatewayRequest, ModelGateway
from codeinsight.infrastructure.otel import Telemetry
from codeinsight.infrastructure.provider_adapters import ProviderResponse
from codeinsight.infrastructure.reranker import RerankResult

BACKEND_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


class FakeModel:
    def complete(self, _system_prompt: str, _user_prompt: str) -> ModelCompletion:
        return ModelCompletion(
            '{"language":"en","normalized_question":"Where is checkout?",'
            '"subquestions":[{"question":"Where is checkout?","intent":"symbol_lookup",'
            '"retrieval_mode":"bm25"}],"execution_route":"linear","confidence":0.9}',
            "fake-router",
            5,
            2,
        )

    def generate(self, _system_prompt: str, _user_prompt: str) -> ModelAnswer:
        return ModelAnswer("answered", "checkout is defined here.", ("E1",), "fake-answer", 8, 3)


class FakeEmbedding:
    @staticmethod
    def embed(texts):
        return EmbeddingBatch(
            "fake",
            tuple((1.0, 0.0) for _ in texts),
            len(texts),
            tuple(SparseEmbedding((1,), (1.0,)) for _ in texts),
            "dense-sparse-v1",
        )


class FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


class Provider:
    def invoke(self, _request):
        return ProviderResponse(
            content="ok",
            tool_calls=(),
            model="deepseek-v4-flash",
            input_tokens=10,
            output_tokens=4,
            cached_input_tokens=6,
            finish_reason="stop",
            cache_miss_tokens=4,
            usage_source="provider_native",
        )


def test_main_api_usage_routes_read_the_same_injected_gateway() -> None:
    gateway = ModelGateway(provider=Provider(), telemetry=Telemetry())
    gateway.complete(
        GatewayRequest(
            request_id="request-from-api-test",
            tenant_id="tenant-a",
            user_id="user-a",
            scene="explain",
            prompt_version="test-v1",
            messages=({"role": "user", "content": "same"},),
            estimated_input_tokens=10,
            reserved_output_tokens=10,
        )
    )

    client = TestClient(
        create_app(
            FakeModel,
            FakeEmbedding,
            reranker_factory=FakeReranker,
            usage_gateway_factory=lambda: gateway,
        )
    )  # type: ignore[arg-type]

    summary = client.get("/api/v1/usage/summary")
    assert summary.status_code == 200
    assert summary.json()["provider_attempts"] == 1
    assert summary.json()["cache_read_tokens"] == 6
    assert summary.json()["cache_miss_tokens"] == 4
    assert summary.json()["usage_coverage"] == 1.0

    calls = client.get("/api/v1/usage/calls?limit=1")
    assert calls.status_code == 200
    detail = calls.json()["data"][0]
    assert detail["request_id"] == "request-from-api-test"
    assert detail["cache_write_tokens"] is None
    assert detail["cache_read_tokens"] == 6
    assert detail["usage_source"] == "provider_native"
    assert "content" not in detail
