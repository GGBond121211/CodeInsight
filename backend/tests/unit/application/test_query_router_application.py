"""严格 QueryPlan 解析和 Router 回退测试。"""

import json

from codeinsight.application.query_router import parse_query_plan, route_question
from codeinsight.domain.answer import ModelCompletion
from codeinsight.domain.errors import ModelCallError, ModelResponseError


def _completion(payload: dict) -> ModelCompletion:
    return ModelCompletion(json.dumps(payload), "fake-router", 11, 5)


def _invalid_completion(_system: str, _user: str) -> ModelCompletion:
    return ModelCompletion("not-json", "fake-router", 3, 1)


def _payload() -> dict:
    return {
        "language": "mixed",
        "normalized_question": "Explain checkout validation and inventory flow.",
        "subquestions": [
            {
                "question": "What does checkout validate?",
                "intent": "implementation",
                "retrieval_mode": "bm25",
            },
            {
                "question": "How does checkout call inventory?",
                "intent": "call_flow",
                "retrieval_mode": "bm25",
            },
        ],
        "execution_route": "agent",
        "confidence": 0.91,
    }


def test_parse_query_plan_preserves_original_question() -> None:
    plan = parse_query_plan("checkout 先校验什么，然后库存怎么走？", json.dumps(_payload()))

    assert plan.original_question.startswith("checkout")
    assert plan.language == "mixed"
    assert plan.retrieval_modes == ("bm25",)
    assert plan.execution_route == "agent"


def test_parse_query_plan_tolerates_model_json_wrapper_and_aliases() -> None:
    payload = {
        "language": "zh-en",
        "normalized_question": "Explain checkout validation and inventory flow.",
        "subquestions": [
            {
                "question": "Trace the checkout call graph.",
                "intent": "call_graph",
                "retrieval_mode": "sparse",
            },
            {
                "question": "Find semantic business rules.",
                "intent": "business_logic",
                "retrieval_mode": "bm25",
            },
        ],
        "execution_route": "linear",
        "confidence": "0.80",
    }

    plan = parse_query_plan("中英混合问题", f"```json\n{json.dumps(payload)}\n```")

    assert plan.language == "mixed"
    assert plan.confidence == 0.8
    assert plan.subquestions[0].intent == "call_flow"
    assert plan.subquestions[0].retrieval_mode == "bm25"
    assert plan.subquestions[1].intent == "implementation"
    assert plan.subquestions[1].retrieval_mode == "bm25"


def test_router_cannot_disable_mandatory_semantic_with_a_hybrid_choice() -> None:
    payload = _payload()
    payload["subquestions"][0]["retrieval_mode"] = "hybrid"

    try:
        parse_query_plan("question", json.dumps(payload))
    except ModelResponseError as error:
        assert "retrieval_mode 不受支持" in str(error)
    else:
        raise AssertionError("Router 只能选择稀疏检索增强")


def test_router_returns_bm25_linear_fallback_on_invalid_output() -> None:
    result = route_question(
        "Where is checkout?",
        complete=_invalid_completion,
    )

    assert result.used_fallback is True
    assert result.plan.execution_route == "linear"
    assert result.plan.subquestions[0].retrieval_mode == "bm25"
    assert result.fallback_reason == "router_invalid_or_low_confidence"


def test_router_falls_back_when_model_call_fails() -> None:
    def fail(_system, _user):
        raise ModelCallError("network unavailable")

    result = route_question("Where is checkout?", complete=fail)

    assert result.used_fallback is True
    assert result.fallback_reason == "router_call_failed"
    assert result.input_tokens == 0


def test_router_falls_back_below_confidence_threshold() -> None:
    payload = _payload()
    payload["confidence"] = 0.69

    def complete_payload(_system: str, _user: str) -> ModelCompletion:
        return _completion(payload)

    result = route_question(
        "checkout 先校验什么？",
        complete=complete_payload,
    )

    assert result.used_fallback is True
    assert result.plan.confidence == 0.0


def test_router_rejects_evidence_fields() -> None:
    payload = _payload()
    payload["evidence"] = []
    try:
        parse_query_plan("question", json.dumps(payload))
    except ModelResponseError as error:
        assert "JSON 结构不受支持" in str(error)
    else:
        raise AssertionError("Router 必须拒绝 evidence 字段")
