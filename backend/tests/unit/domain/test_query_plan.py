"""公开 Smart Answer 查询计划契约测试。"""

import pytest

from codeinsight.domain.query_plan import QueryPlan, SubQuestion


def _plan() -> QueryPlan:
    first = SubQuestion("What does checkout validate?", "implementation", "hybrid")
    second = SubQuestion("How does checkout call inventory?", "call_flow", "hybrid")
    return QueryPlan(
        original_question="checkout 先校验什么，然后库存怎么走？",
        language="mixed",
        normalized_question="Explain checkout validation and inventory flow.",
        subquestions=(first, second),
        retrieval_modes=("hybrid",),
        execution_route="agent",
        confidence=0.91,
    )


def test_query_plan_preserves_original_and_serializes_public_fields() -> None:
    plan = _plan()
    payload = plan.to_dict()

    assert payload["original_question"].startswith("checkout")
    assert payload["execution_route"] == "agent"
    assert payload["subquestions"][1]["retrieval_mode"] == "hybrid"
    assert "path" not in payload
    assert "evidence" not in payload


def test_plan_requires_unique_matching_retrieval_modes() -> None:
    subquestion = SubQuestion("Where is checkout?", "symbol_lookup", "hybrid")
    with pytest.raises(ValueError, match="必须与 subquestion 的 retrieval_mode 匹配"):
        QueryPlan(
            original_question="Where is checkout?",
            language="en",
            normalized_question="Where is checkout?",
            subquestions=(subquestion,),
            retrieval_modes=("dense",),
            execution_route="linear",
            confidence=0.8,
        )


def test_plan_rejects_invalid_route_and_confidence() -> None:
    with pytest.raises(ValueError, match="execution_route 不受支持"):
        QueryPlan(
            original_question="question",
            language="en",
            normalized_question="question",
            subquestions=(),
            retrieval_modes=(),
            execution_route="tool-loop",
            confidence=0.8,
        )
    with pytest.raises(ValueError, match="0 和 1 之间"):
        QueryPlan(
            original_question="question",
            language="en",
            normalized_question="question",
            subquestions=(),
            retrieval_modes=(),
            execution_route="insufficient",
            confidence=1.1,
        )
