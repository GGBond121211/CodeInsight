"""Integration tests for multi-question evidence orchestration."""

from pathlib import Path

from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.query_router import QueryRouterResult
from codeinsight.domain.answer import ModelAnswer
from codeinsight.domain.query_plan import QueryPlan, SubQuestion
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.domain.source import SourceChunk

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


def _router_result() -> QueryRouterResult:
    plan = QueryPlan(
        original_question="checkout 先校验什么，然后支付用什么？",
        language="mixed",
        normalized_question="Explain checkout validation and payment provider.",
        subquestions=(
            SubQuestion("How does checkout validate input?", "implementation", "bm25"),
            SubQuestion("Which payment provider is used?", "unknown", "bm25"),
        ),
        retrieval_modes=("bm25",),
        execution_route="linear",
        confidence=0.9,
    )
    return QueryRouterResult(plan, False, None, "fake-router", 5, 2, 1.0)


def _fake_generate(system_prompt: str, user_prompt: str) -> ModelAnswer:
    if "Which payment provider" in user_prompt:
        return ModelAnswer(
            "insufficient_evidence",
            "The repository does not identify a payment provider.",
            (),
            "fake-answer",
            10,
            4,
        )
    return ModelAnswer(
        "answered",
        "checkout calls validate_request before reservation.",
        ("E1",),
        "fake-answer",
        20,
        8,
    )


def _fake_embed(texts):
    vectors = []
    for _ in texts:
        vectors.append((1.0, 0.0))
    return EmbeddingBatch("fake", tuple(vectors), len(texts))


def test_auto_answer_keeps_answerable_subquestion_when_another_is_unsupported() -> None:
    result = auto_answer_repository(
        FIXTURE_ROOT,
        router_result=_router_result(),
        generate=_fake_generate,
        semantic_embed=_fake_embed,
    )

    assert result.outcome == "partially_answered"
    assert len(result.subquestions) == 2
    assert result.subquestions[0].outcome == "answered"
    assert result.subquestions[1].outcome == "insufficient_evidence"
    assert result.subquestions[0].citations[0].evidence_id == "E1"
    assert result.router_input_tokens == 5
    assert result.input_tokens == 30


def test_auto_answer_fallback_plan_stops_without_answer_model() -> None:
    plan = QueryPlan(
        original_question="?",
        language="unknown",
        normalized_question="?",
        subquestions=(),
        retrieval_modes=(),
        execution_route="insufficient",
        confidence=0.0,
        fallback_reason="router_invalid_or_low_confidence",
    )
    router_result = QueryRouterResult(plan, True, plan.fallback_reason, None, 2, 1, 0.5)
    called = False

    def generate(_system, _user):
        nonlocal called
        called = True
        raise AssertionError("answer model must not be called")

    result = auto_answer_repository(FIXTURE_ROOT, router_result=router_result, generate=generate)

    assert called is False
    assert result.outcome == "insufficient_evidence"
    assert result.fallback_reason == "router_invalid_or_low_confidence"


def test_auto_answer_reports_embedding_input_tokens(monkeypatch) -> None:
    plan = QueryPlan(
        original_question="semantic question",
        language="en",
        normalized_question="semantic question",
        subquestions=(SubQuestion("semantic question", "semantic", "hybrid"),),
        retrieval_modes=("hybrid",),
        execution_route="linear",
        confidence=0.9,
    )
    router_result = QueryRouterResult(plan, False, None, "fake-router", 1, 1, 1.0)
    chunk = SourceChunk("src/shop/service.py", 10, 19, "def checkout():")

    def fake_embed(texts):
        vectors = []
        for _ in texts:
            vectors.append((1.0, 0.0))
        return EmbeddingBatch("fake", tuple(vectors), 5)

    def fake_search(_root, _question, **kwargs):
        assert kwargs["semantic_embed"] is None
        return (RankedChunk(chunk, 1.0, 1, "hybrid_match"),)

    monkeypatch.setattr(
        "codeinsight.application.auto_answer_repository.search_repository",
        fake_search,
    )

    def generate_answer(_system: str, _user: str) -> ModelAnswer:
        return ModelAnswer(
            "answered",
            "The semantic result is supported.",
            ("E1",),
            "fake-answer",
            2,
            1,
        )

    result = auto_answer_repository(
        FIXTURE_ROOT,
        router_result=router_result,
        generate=generate_answer,
        semantic_embed=fake_embed,
    )

    # 一次批处理构建共享仓库索引，另一次查询 Embedding 为子问题生成；
    # 具体索引批大小取决于 fixture。
    assert result.embedding_input_tokens > 5


def test_each_subquestion_keeps_its_own_candidate_capacity(monkeypatch) -> None:
    first = SourceChunk("src/first.py", 1, 3, "first evidence")
    second = SourceChunk("src/second.py", 1, 3, "second evidence")
    calls: list[tuple[str, int]] = []

    def fake_retrieve(_root, question, *, limit, **_kwargs):
        calls.append((question, limit))
        chunk = first if question == "first question" else second
        return (RankedChunk(chunk, 1.0, 1, "hybrid_match"),)

    monkeypatch.setattr(
        "codeinsight.application.auto_answer_repository.retrieve_subquestion_evidence",
        fake_retrieve,
    )
    def fake_build_repository_index(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(
        "codeinsight.application.auto_answer_repository.build_repository_semantic_index",
        fake_build_repository_index,
    )
    plan = QueryPlan(
        original_question="first and second",
        language="en",
        normalized_question="first and second",
        subquestions=(
            SubQuestion("first question", "implementation", "bm25"),
            SubQuestion("second question", "call_flow", "bm25"),
        ),
        retrieval_modes=("bm25",),
        execution_route="linear",
        confidence=0.9,
    )
    def generate_answer(_system: str, user: str) -> ModelAnswer:
        evidence_ids = ("E1",)
        if "first question" not in user:
            evidence_ids = ("E2",)
        return ModelAnswer(
            "answered",
            "supported",
            evidence_ids,
            "answer",
            1,
            1,
        )

    result = auto_answer_repository(
        FIXTURE_ROOT,
        router_result=QueryRouterResult(plan, False, None, "router", 1, 1, 1.0),
        generate=generate_answer,
        limit=5,
        semantic_embed=_fake_embed,
    )

    assert calls == [("first question", 5), ("second question", 5)]
    citation_paths = []
    for item in result.subquestions:
        citation_paths.append(item.citations[0].relative_path)
    assert citation_paths == [
        "src/first.py",
        "src/second.py",
    ]


def test_auto_answer_passes_model_identity_through_usage_wrapper(monkeypatch) -> None:
    captured = {}

    class Embedder:
        model = "persistent-model"

        def embed(self, texts):
            vectors = []
            for _ in texts:
                vectors.append((1.0, 0.0))
            return EmbeddingBatch(self.model, tuple(vectors), len(texts))

    def fake_build(_root, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "codeinsight.application.auto_answer_repository.build_repository_semantic_index",
        fake_build,
    )
    def empty_retrieve(*_args, **_kwargs):
        return ()

    monkeypatch.setattr(
        "codeinsight.application.auto_answer_repository.retrieve_subquestion_evidence",
        empty_retrieve,
    )
    plan = QueryPlan(
        original_question="question",
        language="en",
        normalized_question="question",
        subquestions=(SubQuestion("question", "semantic", "bm25"),),
        retrieval_modes=("bm25",),
        execution_route="linear",
        confidence=0.9,
    )

    auto_answer_repository(
        FIXTURE_ROOT,
        router_result=QueryRouterResult(plan, False, None, "router", 1, 1, 1.0),
        generate=_fake_generate,
        semantic_embed=Embedder().embed,
    )

    assert captured["semantic_model"] == "persistent-model"
