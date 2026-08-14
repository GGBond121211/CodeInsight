"""Tests for the QueryPlan-aware LangGraph retrieval state."""

from pathlib import Path

from codeinsight.agent.workflow import run_citation_agent
from codeinsight.domain.answer import ModelCompletion
from codeinsight.domain.query_plan import QueryPlan, SubQuestion
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.domain.source import SourceChunk

BACKEND_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


class _CompletionSequence:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, _system: str, user: str) -> ModelCompletion:
        self.prompts.append(user)
        content = self.contents[self.calls]
        self.calls += 1
        return ModelCompletion(content, "fake-agent", 10, 4)


def test_query_plan_agent_retrieves_each_subquestion_and_deduplicates_state() -> None:
    plan = QueryPlan(
        original_question="checkout 怎么校验，价格怎么算？",
        language="mixed",
        normalized_question="Explain validation and pricing.",
        subquestions=(
            SubQuestion("How does checkout validate input?", "implementation", "bm25"),
            SubQuestion("How does checkout compute the total?", "data_flow", "bm25"),
        ),
        retrieval_modes=("bm25",),
        execution_route="agent",
        confidence=0.9,
    )
    calls: list[tuple[str, str]] = []

    def search(_root, question, *, retrieval_mode, **_kwargs):
        calls.append((question, retrieval_mode))
        return (
            RankedChunk(
                SourceChunk("src/shop/service.py", 10, 19, "checkout evidence"),
                1.0,
                1,
            ),
        )

    complete = _CompletionSequence(
        '{"outcome":"answered","answer":"Validation is in checkout.","citations":["E1"]}',
        '{"outcome":"insufficient_evidence","answer":"The total formula is not shown.",'
        '"citations":[]}',
        '{"verdict":"pass","supported_citations":["E1"],"feedback":"Supported."}',
    )

    result = run_citation_agent(
        FIXTURE_ROOT,
        plan.original_question,
        complete=complete,
        retrieval_mode="auto",
        search=search,
        query_plan=plan,
        semantic_embed=lambda texts: EmbeddingBatch(
            "fake", tuple((1.0, 0.0) for _ in texts), len(texts)
        ),
    )

    assert calls == [
        ("How does checkout validate input?", "bm25"),
        ("How does checkout compute the total?", "bm25"),
    ]
    assert result.result.retrieval_mode == "auto"
    assert result.result.outcome == "partially_answered"
    assert result.result.citations[0].evidence_id == "E1"
    assert [item.outcome for item in result.subquestions] == [
        "answered",
        "insufficient_evidence",
    ]
    assert result.subquestions[0].answer == "Validation is in checkout."
    assert result.subquestions[1].answer == "The total formula is not shown."
    assert result.subquestions[1].citations == ()
    assert "How does checkout compute the total?" not in complete.prompts[0]
    assert "How does checkout validate input?" not in complete.prompts[1]
    assert "subquestions" in result.events[0].summary
    assert complete.calls == 3


def test_query_plan_critic_revises_only_failed_subquestion() -> None:
    plan = QueryPlan(
        original_question="Explain validation and pricing.",
        language="en",
        normalized_question="Explain validation and pricing.",
        subquestions=(
            SubQuestion("How is input validated?", "implementation", "bm25"),
            SubQuestion("How is price computed?", "data_flow", "bm25"),
        ),
        retrieval_modes=("bm25",),
        execution_route="agent",
        confidence=0.9,
    )

    def search(_root, question, *, retrieval_mode, **_kwargs):
        line = 1 if "validated" in question else 10
        path = "src/validation.py" if "validated" in question else "src/pricing.py"
        return (
            RankedChunk(
                SourceChunk(path, line, line + 2, f"evidence for {question}"),
                1.0,
                1,
            ),
        )

    complete = _CompletionSequence(
        '{"outcome":"answered","answer":"Validation answer.","citations":["E1"]}',
        '{"outcome":"answered","answer":"Weak price answer.","citations":["E2"]}',
        '{"verdict":"pass","supported_citations":["E1"],"feedback":"Supported."}',
        '{"verdict":"revise","supported_citations":["E2"],"feedback":"Be precise."}',
        '{"outcome":"answered","answer":"Precise price answer.","citations":["E2"]}',
        '{"verdict":"pass","supported_citations":["E2"],"feedback":"Supported."}',
    )

    result = run_citation_agent(
        FIXTURE_ROOT,
        plan.original_question,
        complete=complete,
        retrieval_mode="auto",
        search=search,
        query_plan=plan,
        semantic_embed=lambda texts: EmbeddingBatch(
            "fake", tuple((1.0, 0.0) for _ in texts), len(texts)
        ),
    )

    assert result.revisions == 1
    assert [item.answer for item in result.subquestions] == [
        "Validation answer.",
        "Precise price answer.",
    ]
    assert sum("How is input validated?" in prompt for prompt in complete.prompts) == 2
    assert sum("How is price computed?" in prompt for prompt in complete.prompts) == 4
    assert complete.calls == 6
