from pathlib import Path

from codeinsight.application.agent_answer_repository import agent_answer_repository
from codeinsight.domain.answer import ModelCompletion
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


class CompletionSequence:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.index = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> ModelCompletion:
        content = self.contents[self.index]
        self.index += 1
        return ModelCompletion(content, "fake-model", 12, 5)


class _FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, 1.0) for index in range(min(top_n, len(documents))))


def test_agent_answer_maps_reviewed_evidence_from_real_retrieval() -> None:
    complete = CompletionSequence(
        '{"outcome":"answered","answer":"checkout validates input.","citations":["E1","E2"]}',
        '{"verdict":"pass","supported_citations":["E1","E2"],"feedback":"Both are direct."}',
    )

    result = agent_answer_repository(
        FIXTURE_ROOT,
        "How does checkout validate input?",
        complete=complete,
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
    )

    assert result.result.outcome == "answered"
    assert result.result.retrieval_mode == "hybrid"
    assert result.result.prompt_version == "citation-agent-v3"
    assert result.result.citations
    for citation in result.result.citations:
        assert citation.relative_path.startswith("src/shop/")
    assert result.revisions == 0
