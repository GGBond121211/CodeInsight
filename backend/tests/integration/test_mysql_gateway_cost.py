from decimal import Decimal

from sqlalchemy.orm import Session, sessionmaker

from codeinsight.infrastructure.db.stores import MySqlGatewayCostStore
from codeinsight.infrastructure.model_gateway import CostRecord


def test_gateway_cost_roundtrip_keeps_original_price_version(
    session_factory: sessionmaker[Session],
) -> None:
    store = MySqlGatewayCostStore(session_factory)
    original = CostRecord(
        request_id="req-cost",
        attempt_id="attempt-cost",
        tenant_id="default",
        user_id="local",
        scene="change-plan",
        prompt_version="planner-v1",
        provider="frontier-openai-compatible",
        model_tier="best",
        model="deepseek-v4-flash",
        input_tokens=100,
        output_tokens=20,
        cached_tokens=10,
        price_version="frontier-stars-2026-09-04",
        total_stars=Decimal("0.001801996"),
        latency_milliseconds=12.5,
        ttft_milliseconds=None,
        fallback_reason=None,
    )

    store.record(original)
    loaded = store.list_for_request("req-cost")

    assert len(loaded) == 1
    assert loaded[0].price_version == original.price_version
    assert loaded[0].total_stars == Decimal("0.00180200")
