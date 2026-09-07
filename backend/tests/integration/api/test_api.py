from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.domain.answer import ModelAnswer, ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch
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
        vectors = []
        for _ in texts:
            vectors.append((1.0, 0.0))
        return EmbeddingBatch("fake", tuple(vectors), len(texts))


class FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


def test_health_auto_answer_and_usage_are_the_registered_product_routes() -> None:
    client = TestClient(
        create_app(FakeModel, FakeEmbedding, reranker_factory=FakeReranker)
    )  # type: ignore[arg-type]

    assert client.get("/api/v1/health").status_code == 200
    auto = client.post(
        "/api/v1/auto/answer",
        json={"repository_root": str(FIXTURE_ROOT), "question": "Where is checkout?"},
    )
    assert auto.status_code == 200
    assert auto.json()["outcome"] == "answered"

    for path in ("/api/v1/search", "/api/v1/answer", "/api/v1/agent/answer"):
        assert client.post(path, json={}).status_code == 404


def test_openapi_exposes_current_product_routes() -> None:
    client = TestClient(
        create_app(FakeModel, FakeEmbedding, reranker_factory=FakeReranker)
    )  # type: ignore[arg-type]
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/api/v1/health",
        "/api/v1/auto/answer",
        "/api/v1/usage/summary",
        "/api/v1/usage/calls",
        "/api/v2/change/preview",
        "/api/v2/change/approve",
        "/api/v2/change/apply",
        "/api/v2/change/rollback",
        "/api/v2/change/{run_id}/cancel",
        "/api/v2/change/{run_id}/events",
        "/api/v2/change/{run_id}/patches/{patch_id}",
        "/api/v2/chat/sessions",
        "/api/v2/chat/sessions/{session_id}",
        "/api/v2/chat/turns",
        "/api/v2/chat/turns/{turn_id}",
        "/api/v2/chat/runs/{run_id}",
        "/api/v2/chat/turns/{turn_id}/approve",
        "/api/v2/chat/turns/{turn_id}/cancel",
        "/api/v2/chat/turns/{turn_id}/events",
    }
