"""HTTP contract tests for the optional Smart Answer endpoint."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.domain.agent import AgentEvent, AgentRepositoryAnswer
from codeinsight.domain.answer import (
    AnswerCitation,
    ModelAnswer,
    ModelCompletion,
    RepositoryAnswer,
    SubQuestionAnswer,
)
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.reranker import RerankResult

BACKEND_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_ROOT = BACKEND_ROOT / "backend" / "tests" / "fixtures" / "sample_repo"


class _FakeEmbedding:
    @staticmethod
    def embed(texts):
        vectors = []
        for _ in texts:
            vectors.append((1.0, 0.0))
        return EmbeddingBatch(
            "fake",
            tuple(vectors),
            len(texts),
            tuple(SparseEmbedding((1,), (1.0,)) for _ in texts),
            "dense-sparse-v1",
        )


class _FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


class _FakeAutoModel:
    def __init__(self, router_content: str) -> None:
        self.router_content = router_content
        self.router_calls = 0
        self.answer_calls = 0

    def complete(self, _system: str, _user: str) -> ModelCompletion:
        self.router_calls += 1
        return ModelCompletion(self.router_content, "fake-router", 7, 3)

    def generate(self, _system: str, _user: str) -> ModelAnswer:
        self.answer_calls += 1
        return ModelAnswer(
            "answered",
            "checkout validates input.",
            ("E1",),
            "fake-answer",
            12,
            5,
        )


def _router_payload() -> str:
    return json.dumps(
        {
            "language": "en",
            "normalized_question": "Where is checkout validation implemented?",
            "subquestions": [
                {
                    "question": "Where is checkout validation implemented?",
                    "intent": "implementation",
                    "retrieval_mode": "hybrid",
                }
            ],
            "execution_route": "linear",
            "confidence": 0.92,
        }
    )


def _model_factory(model):
    def factory():
        return model

    return factory


def test_auto_answer_returns_plan_subquestion_and_router_usage() -> None:
    model = _FakeAutoModel(_router_payload())
    client = TestClient(
        create_app(_model_factory(model), _FakeEmbedding, reranker_factory=_FakeReranker)
    )  # type: ignore[arg-type]

    response = client.post(
        "/api/v1/auto/answer",
        json={
            "repository_root": str(FIXTURE_ROOT),
            "question": "Where is checkout validation implemented?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["execution_route"] == "linear"
    assert payload["plan"]["subquestions"][0]["retrieval_mode"] == "hybrid"
    assert payload["subquestions"][0]["outcome"] == "answered"
    assert payload["router_usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert payload["usage"] == {"input_tokens": 12, "output_tokens": 5}
    assert payload["embedding_input_tokens"] > 0
    assert model.router_calls == 1
    assert model.answer_calls == 1


def test_auto_answer_invalid_router_output_uses_linear_hybrid_fallback() -> None:
    model = _FakeAutoModel("not-json")
    client = TestClient(
        create_app(_model_factory(model), _FakeEmbedding, reranker_factory=_FakeReranker)
    )  # type: ignore[arg-type]

    response = client.post(
        "/api/v1/auto/answer",
        json={
            "repository_root": str(FIXTURE_ROOT),
            "question": "Where is checkout defined?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["execution_route"] == "linear"
    assert payload["plan"]["retrieval_modes"] == ["hybrid"]
    assert payload["fallback_reason"] == "router_invalid_or_low_confidence"
    assert model.answer_calls == 1


def test_agent_route_returns_each_independent_subquestion_answer(monkeypatch) -> None:
    router_payload = json.dumps(
        {
            "language": "en",
            "normalized_question": "Explain validation and pricing.",
            "subquestions": [
                {
                    "question": "How is input validated?",
                    "intent": "implementation",
                    "retrieval_mode": "hybrid",
                },
                {
                    "question": "How is price computed?",
                    "intent": "data_flow",
                    "retrieval_mode": "hybrid",
                },
            ],
            "execution_route": "agent",
            "confidence": 0.95,
        }
    )
    model = _FakeAutoModel(router_payload)
    validation = SubQuestionAnswer(
        "How is input validated?",
        "implementation",
        "hybrid",
        "answered",
        "Validation answer.",
        (AnswerCitation("E1", "src/validation.py", 1, 4),),
    )
    pricing = SubQuestionAnswer(
        "How is price computed?",
        "data_flow",
        "hybrid",
        "insufficient_evidence",
        "Pricing evidence is insufficient.",
        (),
    )
    combined = RepositoryAnswer(
        "partially_answered",
        "1. How is input validated?\nValidation answer.\n\n"
        "2. How is price computed?\nPricing evidence is insufficient.",
        validation.citations,
        "auto",
        "fake-agent",
        "citation-agent-v3",
        30,
        10,
    )
    def fake_run_citation_agent(*_args, **_kwargs):
        return AgentRepositoryAnswer(
            combined,
            0,
            (AgentEvent(1, "finalize", "Combined independent answers."),),
            30,
            10,
            4,
            (validation, pricing),
        )

    monkeypatch.setattr(
        "codeinsight.api.routes.run_citation_agent",
        fake_run_citation_agent,
    )
    client = TestClient(
        create_app(_model_factory(model), _FakeEmbedding, reranker_factory=_FakeReranker)
    )  # type: ignore[arg-type]

    response = client.post(
        "/api/v1/auto/answer",
        json={
            "repository_root": str(FIXTURE_ROOT),
            "question": "Explain validation and pricing.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "partially_answered"
    outcomes = []
    answers = []
    for item in payload["subquestions"]:
        outcomes.append(item["outcome"])
        answers.append(item["answer"])
    assert outcomes == [
        "answered",
        "insufficient_evidence",
    ]
    assert answers == [
        "Validation answer.",
        "Pricing evidence is insufficient.",
    ]
    assert payload["subquestions"][1]["citations"] == []
