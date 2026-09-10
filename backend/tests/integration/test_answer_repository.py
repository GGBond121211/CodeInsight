from pathlib import Path

import pytest

from codeinsight.application.answer_repository import answer_repository
from codeinsight.application.search_repository import search_repository
from codeinsight.domain.answer import INSUFFICIENT_EVIDENCE, ModelAnswer
from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.reranker import RerankResult

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


def _fake_embed(texts):
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
    def rerank(self, query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


def _generated_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    evidence_ids: tuple[str, ...] = ("E1",),
) -> ModelAnswer:
    assert "仓库证据" in user_prompt
    assert "citations" in system_prompt
    return ModelAnswer(
        outcome="answered",
        answer="checkout is defined in the service module.",
        evidence_ids=evidence_ids,
        model="fake-model",
        input_tokens=20,
        output_tokens=8,
    )


def test_answer_defaults_to_hybrid_and_maps_retrieved_evidence() -> None:
    expected = search_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        retrieval_mode="hybrid",
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
    )[0].chunk

    result = answer_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        generate=_generated_answer,
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
    )

    assert result.retrieval_mode == "hybrid"
    assert result.citations[0].relative_path == expected.relative_path
    assert result.citations[0].start_line == expected.start_line
    assert result.citations[0].end_line == expected.end_line


def test_answer_preserves_parser_deduplication_order() -> None:
    def generate(system: str, user: str) -> ModelAnswer:
        return _generated_answer(system, user, evidence_ids=("E2", "E1"))

    result = answer_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        generate=generate,
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
    )

    citation_ids = []
    for citation in result.citations:
        citation_ids.append(citation.evidence_id)
    assert citation_ids == ["E2", "E1"]


def test_answer_rejects_unknown_evidence_id() -> None:
    def generate(system: str, user: str) -> ModelAnswer:
        return _generated_answer(system, user, evidence_ids=("E99",))

    with pytest.raises(ModelResponseError, match="未知的 evidence ID：E99"):
        answer_repository(
            FIXTURE_ROOT,
            "Where is checkout defined?",
            generate=generate,
            semantic_embed=_fake_embed,
            reranker=_FakeReranker(),
        )


def test_no_results_are_insufficient_without_calling_model(monkeypatch) -> None:
    called = False

    def generate(system_prompt: str, user_prompt: str) -> ModelAnswer:
        nonlocal called
        called = True
        return _generated_answer(system_prompt, user_prompt)

    def empty_search(*args, **kwargs):
        return ()

    monkeypatch.setattr(
        "codeinsight.application.answer_repository.search_repository",
        empty_search,
    )

    result = answer_repository(
        FIXTURE_ROOT, "nothing", generate=generate, semantic_embed=_fake_embed
    )

    assert result.outcome == INSUFFICIENT_EVIDENCE
    assert result.citations == ()
    assert result.model is None
    assert called is False


def test_model_insufficient_result_has_no_citations() -> None:
    def insufficient(system_prompt: str, user_prompt: str) -> ModelAnswer:
        assert "仓库证据" in user_prompt
        return ModelAnswer(
            outcome=INSUFFICIENT_EVIDENCE,
            answer="The evidence does not identify a provider.",
            evidence_ids=(),
            model="fake-model",
            input_tokens=18,
            output_tokens=7,
        )

    result = answer_repository(
        FIXTURE_ROOT,
        "Which payment processor is used?",
        generate=insufficient,
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
    )

    assert result.outcome == INSUFFICIENT_EVIDENCE
    assert result.citations == ()


def test_answer_supports_explicit_sparse_retrieval() -> None:
    result = answer_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        generate=_generated_answer,
        retrieval_mode="sparse",
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
    )

    assert result.retrieval_mode == "sparse"
