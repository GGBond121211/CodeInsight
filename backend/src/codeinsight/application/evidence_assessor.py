"""Evidence 状态的确定性评估器（Q-009）。

它回答的是「当前证据够不够支撑回答」，不是「答案写得好不好」。
因此只做可判定检查：有没有结果、位置是否合法、指纹是否自相矛盾、
用户明确提到的锚点是否被覆盖。不用文本相似度分数冒充「证据正确」。

输出四种状态：sufficient / insufficient / contradictory / invalid。
``invalid`` 表示证据管线给出了越界或畸形的位置。这属于安全边界，
既不放行也不进入 Repair，直接终止。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from codeinsight.application.evidence_ledger import EvidenceLedger, EvidenceRecord

SUFFICIENT = "sufficient"
INSUFFICIENT = "insufficient"
CONTRADICTORY = "contradictory"
INVALID = "invalid"

# 触发原因；与计划里的触发条件一一对应。
NO_RESULTS = "NO_RESULTS"
LOW_COVERAGE = "LOW_COVERAGE"
MISSING_REQUIRED_EVIDENCE = "MISSING_REQUIRED_EVIDENCE"
CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
INVALID_OR_STALE_LOCATION = "INVALID_OR_STALE_LOCATION"

ASSESSMENT_STATUSES: frozenset[str] = frozenset(
    {SUFFICIENT, INSUFFICIENT, CONTRADICTORY, INVALID}
)

_PATH_EXTENSIONS: frozenset[str] = frozenset(
    {
        "py", "pyi", "ts", "tsx", "md", "txt",
        "toml", "yaml", "yml", "json",
    }
)
_PATHISH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z][A-Za-z0-9]{0,4}")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_BOUNDARY = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")

# 只排除明显的英文疑问词和泛词，避免把普通词误判成「必须覆盖的锚点」。
# 判定刻意偏保守：宁可漏报缺失，也不要因为一个泛词就拦住正确答案。
_ANCHOR_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "are", "but", "not", "you", "what", "when", "where",
        "which", "who", "why", "how", "does", "do", "did", "can", "could", "should",
        "would", "this", "that", "these", "those", "with", "from", "into", "about",
        "there", "their", "them", "then", "than", "have", "has", "had", "was", "were",
        "will", "code", "file", "files", "function", "functions", "method", "methods",
        "class", "classes", "module", "modules", "work", "works", "working", "used",
        "using", "use", "call", "calls", "called", "return", "returns", "value", "values",
        "please", "explain", "describe", "show", "tell", "help", "need", "want",
        "implement", "implemented", "implementation", "behavior", "behaviour", "line", "lines",
    }
)


@dataclass(frozen=True)
class EvidenceAssessment:
    """一次证据评估的确定性结果。"""

    status: str
    reasons: tuple[str, ...] = ()
    missing_anchors: tuple[str, ...] = ()
    covered_anchors: tuple[str, ...] = ()
    evidence_count: int = 0
    distinct_paths: int = 0
    retrieval_round: int = 0
    navigation_only: bool = False
    guidance: str = ""

    @property
    def sufficient(self) -> bool:
        return self.status == SUFFICIENT

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "missing_anchors": list(self.missing_anchors),
            "covered_anchors": list(self.covered_anchors),
            "evidence_count": self.evidence_count,
            "distinct_paths": self.distinct_paths,
            "retrieval_round": self.retrieval_round,
            "navigation_only": self.navigation_only,
        }


class EvidenceAssessor:
    """按固定规则评估证据；不调用模型，不接受模型自报的事实。"""

    def __init__(self, *, min_evidence: int = 2, min_distinct_paths: int = 1) -> None:
        if min_evidence < 1:
            raise ValueError("min_evidence 必须是正整数")
        if min_distinct_paths < 1:
            raise ValueError("min_distinct_paths 必须是正整数")
        self.min_evidence = min_evidence
        self.min_distinct_paths = min_distinct_paths

    def assess(
        self,
        *,
        question: str,
        ledger: EvidenceLedger,
        retrieval_round: int = 0,
        navigation_seen: bool = False,
        invalid_signals: tuple[str, ...] = (),
    ) -> EvidenceAssessment:
        records = ledger.records
        anchors = extract_hard_anchors(question)
        covered, missing = _split_anchors(anchors, records)
        count = len(records)
        paths = len({record.path for record in records})

        if invalid_signals:
            return EvidenceAssessment(
                status=INVALID,
                reasons=(INVALID_OR_STALE_LOCATION,),
                missing_anchors=missing,
                covered_anchors=covered,
                evidence_count=count,
                distinct_paths=paths,
                retrieval_round=retrieval_round,
                navigation_only=navigation_seen and count == 0,
                guidance=(
                    "证据管线返回了越界或畸形的位置，属于安全边界问题；"
                    "不要继续检索，也不要据此作答。"
                ),
            )

        if count == 0:
            return EvidenceAssessment(
                status=INSUFFICIENT,
                reasons=(NO_RESULTS,),
                missing_anchors=missing,
                covered_anchors=covered,
                evidence_count=0,
                distinct_paths=0,
                retrieval_round=retrieval_round,
                navigation_only=navigation_seen,
                guidance=(
                    "还没有任何可引用证据。get_repository_map、lsp_definition 和 "
                    "scip_references 只提供导航线索，不能作为回答依据；"
                    "请用 search_repository 或 read_file 形成证据。"
                ),
            )

        conflicts = _contradictory_keys(records)
        if conflicts:
            sample = "、".join(
                f"{path}:{start}-{end}" for path, start, end in conflicts[:3]
            )
            return EvidenceAssessment(
                status=CONTRADICTORY,
                reasons=(CONTRADICTORY_EVIDENCE,),
                missing_anchors=missing,
                covered_anchors=covered,
                evidence_count=count,
                distinct_paths=paths,
                retrieval_round=retrieval_round,
                navigation_only=False,
                guidance=(
                    f"同一位置出现了内容不一致的证据（{sample}），"
                    "说明索引与工作区可能不同步。请用 read_file 读取该位置的当前内容，"
                    "确认后再作答，不要同时引用两个互相冲突的版本。"
                ),
            )

        reasons: list[str] = []
        if missing:
            reasons.append(MISSING_REQUIRED_EVIDENCE)
        if count < self.min_evidence or paths < self.min_distinct_paths:
            reasons.append(LOW_COVERAGE)
        if reasons:
            return EvidenceAssessment(
                status=INSUFFICIENT,
                reasons=tuple(reasons),
                missing_anchors=missing,
                covered_anchors=covered,
                evidence_count=count,
                distinct_paths=paths,
                retrieval_round=retrieval_round,
                navigation_only=False,
                guidance=_insufficient_guidance(missing, count, self.min_evidence),
            )

        return EvidenceAssessment(
            status=SUFFICIENT,
            reasons=(),
            missing_anchors=(),
            covered_anchors=covered,
            evidence_count=count,
            distinct_paths=paths,
            retrieval_round=retrieval_round,
            navigation_only=False,
            guidance="证据已足够，请直接给出最终回答。",
        )


def extract_hard_anchors(question: str) -> tuple[str, ...]:
    """抽出用户明确提到、必须被证据覆盖的代码形锚点。

    只认「像代码」的词：带下划线、驼峰，或以已知源码扩展名结尾的路径。
    普通英文词不算锚点，避免把 function 这类泛词当成硬性要求。
    """
    found: list[str] = []
    for match in _PATHISH.finditer(question):
        token = match.group(0).replace("\\", "/").lstrip("./")
        if token and token.rsplit(".", 1)[-1].casefold() in _PATH_EXTENSIONS:
            found.append(token)
    for match in _TOKEN.finditer(question):
        token = match.group(0)
        if len(token) < 3 or token.casefold() in _ANCHOR_STOPWORDS:
            continue
        if "_" in token or _CAMEL_BOUNDARY.search(token):
            found.append(token)
    unique: list[str] = []
    for token in found:
        if token not in unique:
            unique.append(token)
    return tuple(unique)


def _split_anchors(
    anchors: tuple[str, ...], records: tuple[EvidenceRecord, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    covered: list[str] = []
    missing: list[str] = []
    for anchor in anchors:
        if _anchor_covered(anchor, records):
            covered.append(anchor)
        else:
            missing.append(anchor)
    return tuple(covered), tuple(missing)


def _anchor_covered(anchor: str, records: tuple[EvidenceRecord, ...]) -> bool:
    lowered = anchor.casefold()
    for record in records:
        path = record.path.casefold()
        if path == lowered or path.endswith("/" + lowered):
            return True
        if anchor in record.excerpt:
            return True
    return False


def _contradictory_keys(
    records: tuple[EvidenceRecord, ...],
) -> tuple[tuple[str, int, int], ...]:
    """同一 path+行区间出现多个指纹即为冲突。"""
    grouped: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for record in records:
        grouped[(record.path, record.start_line, record.end_line)].add(
            record.source_fingerprint
        )
    return tuple(key for key, fingerprints in grouped.items() if len(fingerprints) > 1)


def _insufficient_guidance(
    missing: tuple[str, ...], count: int, min_evidence: int
) -> str:
    parts = ["当前证据不足以支撑回答。"]
    if missing:
        parts.append(f"问题明确提到的 {list(missing)} 还没有被任何证据覆盖。")
    if count < min_evidence:
        parts.append(f"可引用证据只有 {count} 条，低于 {min_evidence} 条下限。")
    parts.append("请补充检索后再作答；确实找不到时如实说明不足，不要猜测。")
    return "".join(parts)
