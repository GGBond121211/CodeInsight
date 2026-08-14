"""Tests for the public Smart Answer query plan contract."""

import pytest

from codeinsight.domain.query_plan import QueryPlan, SubQuestion


def _plan() -> QueryPlan:
    first = SubQuestion("What does checkout validate?", "implementation", "bm25")
    second = SubQuestion("How does checkout call inventory?", "call_flow", "bm25")
    return QueryPlan(
        original_question="checkout 先校验什么，然后库存怎么走？",
        language="mixed",
        normalized_question="Explain checkout validation and inventory flow.",
        subquestions=(first, second),
        retrieval_modes=("bm25",),
        execution_route="agent",
        confidence=0.91,
    )


def test_query_plan_preserves_original_and_serializes_public_fields() -> None:
    plan = _plan()
    payload = plan.to_dict()

    assert payload["original_question"].startswith("checkout")
    assert payload["execution_route"] == "agent"
    assert payload["subquestions"][1]["retrieval_mode"] == "bm25"
    assert "path" not in payload
    assert "evidence" not in payload


def test_plan_requires_unique_matching_retrieval_modes() -> None:
    subquestion = SubQuestion("Where is checkout?", "symbol_lookup", "bm25")
    with pytest.raises(ValueError, match="must match"):
        QueryPlan(
            original_question="Where is checkout?",
            language="en",
            normalized_question="Where is checkout?",
            subquestions=(subquestion,),
            retrieval_modes=("lexical",),
            execution_route="linear",
            confidence=0.8,
        )


def test_plan_rejects_invalid_route_and_confidence() -> None:
    with pytest.raises(ValueError, match="execution route"):
        QueryPlan(
            original_question="question",
            language="en",
            normalized_question="question",
            subquestions=(),
            retrieval_modes=(),
            execution_route="tool-loop",
            confidence=0.8,
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        QueryPlan(
            original_question="question",
            language="en",
            normalized_question="question",
            subquestions=(),
            retrieval_modes=(),
            execution_route="insufficient",
            confidence=1.1,
        )
