from pathlib import Path

from codeinsight.domain.answer import AnswerCitation, RepositoryAnswer
from codeinsight.evaluation.answer_metrics import (
    citations_jointly_cover,
    evaluate_answer_results,
)


def _case(
    *,
    case_id: str = "case-1",
    outcome: str = "answered",
    evidence: list[dict] | None = None,
    required_terms: list[str] | None = None,
) -> dict:
    expected = {"outcome": outcome}
    if evidence is not None:
        expected["evidence"] = evidence
    return {
        "id": case_id,
        "expected": expected,
        "required_terms": required_terms or [],
    }


def _answer(
    *,
    outcome: str = "answered",
    text: str = "run handles the request",
    citations: tuple[AnswerCitation, ...] = (),
) -> RepositoryAnswer:
    return RepositoryAnswer(
        outcome=outcome,
        answer=text,
        citations=citations,
        retrieval_mode="hybrid",
        model="fake-model",
        prompt_version="code-answer-v1",
        input_tokens=10,
        output_tokens=5,
    )


def _fixture(tmp_path: Path) -> Path:
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (source / "other.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    return tmp_path


def test_complete_answer_produces_perfect_metrics(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    case = _case(
        evidence=[{"path": "src/app.py", "start_line": 2, "end_line": 3}],
        required_terms=["run"],
    )
    result = _answer(citations=(AnswerCitation("E1", "src/app.py", 1, 3),))

    metrics = evaluate_answer_results([case], {"case-1": result}, root)

    assert metrics.case_count == 1
    assert metrics.outcome_accuracy == 1.0
    assert metrics.valid_citation_rate == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.evidence_recall == 1.0
    assert metrics.required_term_case_rate == 1.0


def test_wrong_outcome_lowers_accuracy(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    case = _case(outcome="insufficient_evidence")

    metrics = evaluate_answer_results([case], {"case-1": _answer()}, root)

    assert metrics.outcome_accuracy == 0.0


def test_unexpected_citation_lowers_precision_but_remains_valid(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    case = _case(evidence=[{"path": "src/app.py", "start_line": 2, "end_line": 3}])
    result = _answer(citations=(AnswerCitation("E1", "src/other.py", 1, 3),))

    metrics = evaluate_answer_results([case], {"case-1": result}, root)

    assert metrics.valid_citation_rate == 1.0
    assert metrics.citation_precision == 0.0


def test_missing_multi_file_evidence_lowers_recall(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    case = _case(
        evidence=[
            {"path": "src/app.py", "start_line": 1, "end_line": 2},
            {"path": "src/other.py", "start_line": 1, "end_line": 2},
        ]
    )
    result = _answer(citations=(AnswerCitation("E1", "src/app.py", 1, 3),))

    metrics = evaluate_answer_results([case], {"case-1": result}, root)

    assert metrics.evidence_recall == 0.5


def test_invalid_path_lowers_valid_citation_rate(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    case = _case(evidence=[{"path": "src/app.py", "start_line": 1, "end_line": 2}])
    result = _answer(citations=(AnswerCitation("E1", "src/missing.py", 1, 2),))

    metrics = evaluate_answer_results([case], {"case-1": result}, root)

    assert metrics.valid_citation_rate == 0.0


def test_missing_required_term_fails_case_proxy(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    case = _case(required_terms=["validate_request"])

    metrics = evaluate_answer_results([case], {"case-1": _answer()}, root)

    assert metrics.required_term_case_rate == 0.0


def test_insufficient_case_passes_only_with_outcome_and_no_citations(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    case = _case(outcome="insufficient_evidence")
    passing = _answer(
        outcome="insufficient_evidence",
        text="The repository does not contain enough evidence.",
    )
    failing = _answer(
        outcome="insufficient_evidence",
        citations=(AnswerCitation("E1", "src/app.py", 1, 3),),
    )

    passed = evaluate_answer_results([case], {"case-1": passing}, root)
    failed = evaluate_answer_results([case], {"case-1": failing}, root)

    assert passed.required_term_case_rate == 1.0
    assert failed.required_term_case_rate == 0.0


def test_multiple_citations_can_jointly_cover_one_long_requirement() -> None:
    evidence = {"path": "src/app.py", "start_line": 1, "end_line": 120}
    citations = (
        AnswerCitation("E1", "src/app.py", 1, 80),
        AnswerCitation("E2", "src/app.py", 81, 140),
    )

    assert citations_jointly_cover(citations, evidence)


def test_joint_coverage_rejects_a_gap_between_citations() -> None:
    evidence = {"path": "src/app.py", "start_line": 1, "end_line": 120}
    citations = (
        AnswerCitation("E1", "src/app.py", 1, 60),
        AnswerCitation("E2", "src/app.py", 62, 140),
    )

    assert not citations_jointly_cover(citations, evidence)
