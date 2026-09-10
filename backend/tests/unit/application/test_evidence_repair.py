"""Evidence Repair 控制器的预算与动作契约测试（Q-009）。"""

import pytest

from codeinsight.application.evidence_assessor import (
    CONTRADICTORY,
    CONTRADICTORY_EVIDENCE,
    INSUFFICIENT,
    INVALID,
    LOW_COVERAGE,
    MISSING_REQUIRED_EVIDENCE,
    NO_RESULTS,
    SUFFICIENT,
    EvidenceAssessment,
)
from codeinsight.application.evidence_repair import (
    ACTION_ADJACENT_CONTEXT,
    ACTION_QUERY_BROADEN,
    ACTION_RELATION_TOOLS,
    ACTION_WIDEN_SCOPE,
    EvidenceRepairController,
    RepairBudget,
    RepairRequest,
)


def _assessment(status: str, *reasons: str, missing: tuple[str, ...] = ()) -> EvidenceAssessment:
    return EvidenceAssessment(
        status=status, reasons=tuple(reasons), missing_anchors=missing
    )


def test_sufficient_evidence_is_never_repaired() -> None:
    controller = EvidenceRepairController()
    decision = controller.decide(
        assessment=_assessment(SUFFICIENT), repair_round=1
    )
    assert decision.allowed is False
    assert "证据已足够" in decision.reason


def test_invalid_evidence_does_not_enter_repair() -> None:
    controller = EvidenceRepairController()
    decision = controller.decide(
        assessment=_assessment(INVALID, "INVALID_OR_STALE_LOCATION"), repair_round=1
    )
    assert decision.allowed is False
    assert "安全边界" in decision.reason


def test_no_results_first_action_broadens_the_query() -> None:
    controller = EvidenceRepairController()
    decision = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=1
    )
    assert decision.allowed is True
    assert decision.request.action == ACTION_QUERY_BROADEN
    assert "保留原始问题" in decision.request.guidance


def test_missing_anchor_guidance_lists_the_anchor() -> None:
    controller = EvidenceRepairController()
    decision = controller.decide(
        assessment=_assessment(
            INSUFFICIENT, MISSING_REQUIRED_EVIDENCE, missing=("localHelper",)
        ),
        repair_round=1,
    )
    assert decision.request.action == ACTION_QUERY_BROADEN
    assert "localHelper" in decision.request.guidance


def test_low_coverage_uses_the_widening_action() -> None:
    controller = EvidenceRepairController()
    decision = controller.decide(
        assessment=_assessment(INSUFFICIENT, LOW_COVERAGE), repair_round=1
    )
    assert decision.request.action == ACTION_WIDEN_SCOPE


def test_contradictory_evidence_asks_for_a_fresh_read() -> None:
    controller = EvidenceRepairController()
    decision = controller.decide(
        assessment=_assessment(CONTRADICTORY, CONTRADICTORY_EVIDENCE), repair_round=1
    )
    assert decision.request.action == ACTION_ADJACENT_CONTEXT
    assert "read_file" in decision.request.guidance


def test_same_action_is_never_repeated() -> None:
    controller = EvidenceRepairController(budget=RepairBudget(max_repair_rounds=3))
    first = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=1
    )
    controller.commit(first.request)
    second = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=2
    )
    assert second.allowed is True
    assert second.request.action != ACTION_QUERY_BROADEN
    assert second.request.action == ACTION_RELATION_TOOLS


def test_budget_exhaustion_stops_repair() -> None:
    controller = EvidenceRepairController(budget=RepairBudget(max_repair_rounds=1))
    first = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=1
    )
    controller.commit(first.request)
    second = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=2
    )
    assert second.allowed is False
    assert "轮数已用尽" in second.reason


def test_zero_budget_means_repair_is_disabled() -> None:
    controller = EvidenceRepairController(budget=RepairBudget(max_repair_rounds=0))
    decision = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=1
    )
    assert decision.allowed is False


def test_all_actions_used_stops_repeating_searches() -> None:
    controller = EvidenceRepairController(budget=RepairBudget(max_repair_rounds=5))
    for round_index in range(1, 4):
        decision = controller.decide(
            assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=round_index
        )
        assert decision.allowed is True
        controller.commit(decision.request)
    exhausted = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=4
    )
    assert exhausted.allowed is False
    assert "没有未使用过的 Repair 动作" in exhausted.reason
    assert controller.used_actions == (
        ACTION_QUERY_BROADEN,
        ACTION_RELATION_TOOLS,
        ACTION_WIDEN_SCOPE,
    )


def test_round_number_must_follow_the_committed_rounds() -> None:
    controller = EvidenceRepairController()
    decision = controller.decide(
        assessment=_assessment(INSUFFICIENT, NO_RESULTS), repair_round=3
    )
    assert decision.allowed is False
    assert "序号" in decision.reason


def test_committing_the_same_action_twice_is_rejected() -> None:
    controller = EvidenceRepairController()
    request = RepairRequest(
        action=ACTION_QUERY_BROADEN, round=1, reason=NO_RESULTS, guidance="g"
    )
    controller.commit(request)
    with pytest.raises(ValueError):
        controller.commit(request)


def test_unknown_action_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        RepairRequest(action="Z_something", round=1, reason="x", guidance="y")


@pytest.mark.parametrize("max_rounds", [-1])
def test_negative_repair_rounds_is_rejected(max_rounds: int) -> None:
    with pytest.raises(ValueError):
        RepairBudget(max_repair_rounds=max_rounds)


def test_read_file_guidance_respects_the_line_budget() -> None:
    controller = EvidenceRepairController(
        budget=RepairBudget(max_repair_rounds=1, max_read_file_lines=123)
    )
    decision = controller.decide(
        assessment=_assessment(CONTRADICTORY, CONTRADICTORY_EVIDENCE), repair_round=1
    )
    assert "123" in decision.request.guidance
