"""基于评测用例和扫描结果块计算检索质量指标。"""

from collections.abc import Sequence
from dataclasses import dataclass

from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.bm25 import search_chunks_bm25
from codeinsight.retrieval.lexical import search_chunks as search_chunks_lexical


@dataclass(frozen=True)
class RetrievalMetrics:
    """汇总适用评测用例上的检索质量。"""

    case_count: int
    applicable_case_count: int
    top1: float
    mean_reciprocal_rank: float
    recall_at_5: float
    valid_evidence_rate: float


def _evidence_items(case: dict) -> tuple[dict, ...]:
    return tuple(case.get("expected", {}).get("evidence", ()))


def _covers(chunk: SourceChunk, evidence: dict) -> bool:
    return (
        chunk.relative_path == evidence["path"]
        and chunk.start_line <= evidence["start_line"]
        and chunk.end_line >= evidence["end_line"]
    )


def _coverage_rank(results: Sequence, evidence: tuple[dict, ...]) -> int | None:
    covered = [False] * len(evidence)
    for result in results:
        for index, item in enumerate(evidence):
            covered[index] = covered[index] or _covers(result.chunk, item)
        if all(covered):
            return result.rank
    return None


def _is_valid(chunk: SourceChunk) -> bool:
    return (
        bool(chunk.relative_path) and chunk.start_line >= 1 and chunk.end_line >= chunk.start_line
    )


def _zero_metrics(case_count: int) -> RetrievalMetrics:
    return RetrievalMetrics(
        case_count=case_count,
        applicable_case_count=0,
        top1=0.0,
        mean_reciprocal_rank=0.0,
        recall_at_5=0.0,
        valid_evidence_rate=0.0,
    )


def _search_chunks(
    question: str,
    chunks: Sequence[SourceChunk],
    *,
    limit: int,
    retrieval_mode: str,
):
    if retrieval_mode == "lexical":
        return search_chunks_lexical(question, chunks, limit=limit)
    if retrieval_mode == "bm25":
        return search_chunks_bm25(question, chunks, limit=limit)
        raise ValueError(f"不支持的检索模式：{retrieval_mode}")


def evaluate_cases(
    cases: Sequence[dict],
    chunks: Sequence[SourceChunk],
    *,
    retrieval_mode: str = "lexical",
) -> RetrievalMetrics:
    """使用 *chunks* 评估 *cases* 上的检索质量。

    每个用例都会计入 ``case_count``。只有带有非空 ``expected.evidence`` 列表的
    answered 用例才适用；每个适用用例都使用指定检索模式搜索，当返回块匹配证据项的
    路径和行号范围时，认为该证据项被覆盖。
    ``insufficient_evidence`` 用例会计数但不会搜索，``reason_code`` 不会贡献任何分数。
    """
    case_count = len(cases)
    applicable = [case for case in cases if _evidence_items(case)]
    if not applicable:
        return _zero_metrics(case_count)

    top1_hits = 0
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    returned_total = 0
    returned_valid = 0
    for case in applicable:
        evidence = _evidence_items(case)
        results = _search_chunks(
            case["input"]["question"],
            chunks,
            limit=5,
            retrieval_mode=retrieval_mode,
        )
        returned_total += len(results)
        returned_valid += sum(1 for item in results if _is_valid(item.chunk))
        if results and all(_covers(results[0].chunk, item) for item in evidence):
            top1_hits += 1
        covering_rank = _coverage_rank(results, evidence)
        reciprocal_ranks.append(1.0 / covering_rank if covering_rank else 0.0)
        if all(
            any(_covers(result.chunk, evidence_item) for result in results)
            for evidence_item in evidence
        ):
            recall_hits += 1

    applicable_count = len(applicable)
    valid_rate = returned_valid / returned_total if returned_total else 0.0
    return RetrievalMetrics(
        case_count=case_count,
        applicable_case_count=applicable_count,
        top1=top1_hits / applicable_count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / applicable_count,
        recall_at_5=recall_hits / applicable_count,
        valid_evidence_rate=valid_rate,
    )
