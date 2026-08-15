"""基于证据回答仓库问题的确定性质量指标。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from codeinsight.domain.answer import INSUFFICIENT_EVIDENCE, AnswerCitation, RepositoryAnswer


@dataclass(frozen=True)
class AnswerMetrics:
    case_count: int
    outcome_accuracy: float
    valid_citation_rate: float
    citation_precision: float
    evidence_recall: float
    required_term_case_rate: float


def citation_covers(citation: AnswerCitation, evidence: dict) -> bool:
    """判断引用是否覆盖一个预期证据范围。"""
    return (
        citation.relative_path == evidence["path"]
        and citation.start_line <= evidence["start_line"]
        and citation.end_line >= evidence["end_line"]
    )


def citation_overlaps(citation: AnswerCitation, evidence: dict) -> bool:
    """判断引用是否与预期证据区域相交。"""
    return (
        citation.relative_path == evidence["path"]
        and citation.start_line <= evidence["end_line"]
        and citation.end_line >= evidence["start_line"]
    )


def citations_jointly_cover(citations: Sequence[AnswerCitation], evidence: dict) -> bool:
    """判断同一文件中的多个引用是否共同覆盖完整要求。"""
    intervals: list[tuple[int, int]] = []
    for citation in citations:
        if citation_overlaps(citation, evidence):
            intervals.append(
                (
                    max(citation.start_line, evidence["start_line"]),
                    min(citation.end_line, evidence["end_line"]),
                )
            )
    intervals.sort()
    if not intervals:
        return False
    covered_until = evidence["start_line"] - 1
    for start, end in intervals:
        if start > covered_until + 1:
            return False
        covered_until = max(covered_until, end)
        if covered_until >= evidence["end_line"]:
            return True
    return False


def expected_evidence_requirements(expected: dict) -> tuple[dict, ...]:
    """返回原始要求，同时兼容历史的扁平 evidence 结构。"""
    requirements = expected.get("evidence_requirements")
    return tuple(requirements if requirements is not None else expected.get("evidence", ()))


def citation_is_valid(citation: AnswerCitation, fixture_root: Path) -> bool:
    """检查引用是否指向 fixture 中真实的、从 1 开始的行号范围。"""
    resolved_root = fixture_root.resolve()
    resolved = (fixture_root / citation.relative_path).resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        return False
    if citation.start_line < 1 or citation.end_line < citation.start_line:
        return False
    line_count = len(resolved.read_text(encoding="utf-8").splitlines())
    return citation.end_line <= line_count


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def answer_failure_types(
    case: dict,
    result: RepositoryAnswer | None,
    fixture_root: Path,
) -> list[str]:
    """为一个评测用例分类确定性回答失败。"""
    if result is None:
        return ["model_error"]

    failures = []
    expected = case["expected"]
    requirements = expected_evidence_requirements(expected)
    if result.outcome != expected["outcome"]:
        failures.append("outcome")

    has_citation_failure = False
    for citation in result.citations:
        citation_overlaps_requirement = False
        for item in requirements:
            if citation_overlaps(citation, item):
                citation_overlaps_requirement = True
                break
        citation_is_usable = citation_is_valid(citation, fixture_root)
        if not citation_is_usable or not citation_overlaps_requirement:
            has_citation_failure = True
            break
    if not has_citation_failure:
        for item in requirements:
            if not citations_jointly_cover(result.citations, item):
                has_citation_failure = True
                break
    if has_citation_failure:
        failures.append("citation")

    required_terms = case.get("required_terms", ())
    if expected["outcome"] == INSUFFICIENT_EVIDENCE:
        terms_pass = result.outcome == INSUFFICIENT_EVIDENCE and not result.citations
    else:
        folded_answer = result.answer.casefold()
        terms_pass = True
        for term in required_terms:
            if term.casefold() not in folded_answer:
                terms_pass = False
                break
    if not terms_pass:
        failures.append("required_terms")
    return failures


def evaluate_answer_results(
    cases: Sequence[dict],
    results: Mapping[str, RepositoryAnswer | None],
    fixture_root: Path,
) -> AnswerMetrics:
    """评估结果、基于证据的引用、证据覆盖和术语代理指标。"""
    outcome_hits = 0
    valid_citations = 0
    citation_hits = 0
    citation_total = 0
    evidence_hits = 0
    evidence_total = 0
    required_term_case_hits = 0

    for case in cases:
        result = results.get(case["id"])
        expected = case["expected"]
        expected_outcome = expected["outcome"]
        requirements = expected_evidence_requirements(expected)
        evidence_total += len(requirements)

        if result is None:
            continue
        if result.outcome == expected_outcome:
            outcome_hits += 1

        citation_total += len(result.citations)
        for citation in result.citations:
            if citation_is_valid(citation, fixture_root):
                valid_citations += 1

            citation_hits_requirement = False
            for item in requirements:
                if citation_overlaps(citation, item):
                    citation_hits_requirement = True
                    break
            if citation_hits_requirement:
                citation_hits += 1

        for item in requirements:
            if citations_jointly_cover(result.citations, item):
                evidence_hits += 1

        required_terms = case.get("required_terms", ())
        if expected_outcome == INSUFFICIENT_EVIDENCE:
            term_case_passes = result.outcome == INSUFFICIENT_EVIDENCE and not result.citations
        else:
            folded_answer = result.answer.casefold()
            term_case_passes = True
            for term in required_terms:
                if term.casefold() not in folded_answer:
                    term_case_passes = False
                    break
        required_term_case_hits += int(term_case_passes)

    case_count = len(cases)
    return AnswerMetrics(
        case_count=case_count,
        outcome_accuracy=_rate(outcome_hits, case_count),
        valid_citation_rate=_rate(valid_citations, citation_total),
        citation_precision=_rate(citation_hits, citation_total),
        evidence_recall=_rate(evidence_hits, evidence_total),
        required_term_case_rate=_rate(required_term_case_hits, case_count),
    )
