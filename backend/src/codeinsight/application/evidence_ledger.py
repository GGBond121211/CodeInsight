"""代码理解运行期的 Evidence Ledger（Q-008 / Q-009）。

职责：
    把 ToolResult 里可以成为证据的内容规范化成带稳定编号的记录，
    并保证「模型只能引用应用分配的编号」这条硬门槛在 Tool Loop 里同样成立。

三条边界：
    1. 只有 ``search_repository``、``get_evidence_context``、``read_file`` 的
       结果可以成为 Evidence。``get_repository_map``、``lsp_definition``、
       ``scip_references`` 只提供导航线索，不构成可引用证据。
    2. 同一 (path, start_line, end_line, source_fingerprint) 只占一个编号。
       不同工具或不同轮次命中同一块时记录来源标签，但不重复占用额度。
    3. 记录里保存的 excerpt 只用于确定性的覆盖检查，不作为引用文本。
       正式引用始终以 path/line 为准，由调用方回到仓库重新读取。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from codeinsight.agent.tool_loop import ToolCall, ToolResult

# 可以把结果变成 Evidence 的工具。
EVIDENCE_TOOLS: frozenset[str] = frozenset(
    {"search_repository", "get_evidence_context", "read_file"}
)

# 只提供导航线索、不能直接成为 Evidence 的工具。
NAVIGATION_TOOLS: frozenset[str] = frozenset(
    {"get_repository_map", "lsp_definition", "scip_references"}
)

DEFAULT_MAX_EVIDENCE = 40
EXCERPT_CHARS = 400

# 片段锚定用的问题词表：只用于把窗口对准相关行，不参与任何判定。
_FOCUS_STOPWORDS = frozenset(
    {
        "the", "and", "for", "from", "with", "what", "where", "which", "when",
        "how", "why", "does", "did", "are", "was", "were", "this", "that",
        "there", "here", "into", "over", "under", "about", "code", "file",
        "files", "class", "method", "function", "defined", "definition", "whereis",
    }
)
_FOCUS_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_DEFINITION_LINE = re.compile(r"^\s*(?:async\s+def|def|class)\s+")


def _focus_tokens(text: str) -> tuple[str, ...]:
    """从问题里取标识符候选；顺序即优先级，最多 8 个。"""

    tokens: list[str] = []
    for token in _FOCUS_TOKEN.findall(text):
        lowered = token.lower()
        if lowered in _FOCUS_STOPWORDS or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) >= 8:
            break
    return tuple(tokens)


def _line_contains_identifier(line: str, token: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", line) is not None


def _anchor_offset(text: str, tokens: Sequence[str]) -> int:
    """返回片段窗口的起始字符偏移。

    为什么需要锚定：证据块常常有几十行，而问题要的那一行不一定在开头。之前一律取
    ``text[:excerpt_chars]``，于是「class Client 定义在哪」这类问题拿到的片段里根本
    没有 ``class Client``——模型只能凭编号列表猜该引用哪一条，实测就猜错过。
    规则按优先级取第一个命中的：定义行（``class``/``def``）→ 含标识符的行 → 块开头。
    """

    if not tokens:
        return 0
    offset = 0
    first_identifier_line: int | None = None
    for line in text.splitlines(keepends=True):
        matched = [token for token in tokens if _line_contains_identifier(line, token)]
        if matched:
            if _DEFINITION_LINE.match(line):
                return offset
            if first_identifier_line is None:
                first_identifier_line = offset
        offset += len(line)
    return first_identifier_line if first_identifier_line is not None else 0



@dataclass(frozen=True)
class EvidenceRecord:
    """一条可引用证据；位置和指纹由应用决定，模型无法自报。"""

    evidence_id: str
    source_tool: str
    query_or_symbol: str
    path: str
    start_line: int
    end_line: int
    source_fingerprint: str
    retrieval_round: int
    candidate_rank: int
    excerpt: str
    # 片段在块内的起始行；等于 start_line 时表示窗口仍在块开头。
    excerpt_start_line: int | None = None
    routes: tuple[str, ...] = ()
    untrusted: bool = True

    @property
    def dedupe_key(self) -> tuple[str, int, int, str]:
        return (self.path, self.start_line, self.end_line, self.source_fingerprint)

    def as_dict(self) -> dict[str, object]:
        """公开形状；不含仓库正文。"""
        return {
            "evidence_id": self.evidence_id,
            "source_tool": self.source_tool,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "retrieval_round": self.retrieval_round,
            "candidate_rank": self.candidate_rank,
            "routes": list(self.routes),
            "untrusted": self.untrusted,
        }


@dataclass(frozen=True)
class LedgerIngestReport:
    """一次工具结果的入库结果，用于事件和 Repair 归因。"""

    added: tuple[EvidenceRecord, ...] = ()
    duplicates: tuple[str, ...] = ()
    skipped_reasons: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def new_count(self) -> int:
        return len(self.added)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)


class EvidenceLedger:
    """一次 Run 内的证据台账；不跨 Run 复用，不持久化仓库正文。"""

    def __init__(
        self,
        *,
        max_evidence: int = DEFAULT_MAX_EVIDENCE,
        excerpt_chars: int = EXCERPT_CHARS,
    ) -> None:
        if max_evidence < 1:
            raise ValueError("max_evidence 必须是正整数")
        if excerpt_chars < 1:
            raise ValueError("excerpt_chars 必须是正整数")
        self.max_evidence = max_evidence
        self.excerpt_chars = excerpt_chars
        self._records: dict[str, EvidenceRecord] = {}
        self._by_key: dict[tuple[str, int, int, str], str] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(record.path for record in self._records.values()))

    def by_id(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def unknown_ids(self, evidence_ids: Iterable[str]) -> tuple[str, ...]:
        """返回不属于本台账的编号；模型自报的外部编号会在这里被拦下。"""
        seen: list[str] = []
        for evidence_id in evidence_ids:
            if evidence_id not in self._records and evidence_id not in seen:
                seen.append(evidence_id)
        return tuple(seen)

    def ingest(
        self,
        *,
        call: ToolCall,
        result: ToolResult,
        retrieval_round: int = 0,
        focus_text: str = "",
    ) -> LedgerIngestReport:
        """把一次成功的工具结果并入台账。失败结果和导航工具一律不入库。

        ``skipped`` 只记录形状问题的原因；越界和行号非法由调用方转成
        ``invalid_signals``，使评估器按安全边界处理，而不是被静默忽略。
        """
        if not result.ok:
            return LedgerIngestReport()
        if call.name not in EVIDENCE_TOOLS:
            return LedgerIngestReport()
        candidates, skip_reasons = _extract_candidates(
            call,
            result,
            excerpt_chars=self.excerpt_chars,
            focus_text=focus_text,
        )
        if not candidates:
            return LedgerIngestReport(skipped_reasons=tuple(skip_reasons))
        query_or_symbol = _query_label(call)
        added: list[EvidenceRecord] = []
        duplicates: list[str] = []
        skipped: list[str] = list(skip_reasons)
        truncated = False
        for candidate in candidates:
            key = (
                candidate.path,
                candidate.start_line,
                candidate.end_line,
                candidate.source_fingerprint,
            )
            existing_id = self._by_key.get(key)
            if existing_id is not None:
                duplicates.append(existing_id)
                self._add_route(existing_id, call.name)
                continue
            if len(self._records) >= self.max_evidence:
                truncated = True
                break
            evidence_id = f"E{len(self._records) + 1}"
            record = EvidenceRecord(
                evidence_id=evidence_id,
                source_tool=call.name,
                query_or_symbol=query_or_symbol,
                path=candidate.path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                source_fingerprint=candidate.source_fingerprint,
                retrieval_round=retrieval_round,
                candidate_rank=candidate.candidate_rank,
                excerpt=candidate.excerpt,
                excerpt_start_line=candidate.excerpt_start_line,
                routes=(call.name,),
            )
            self._records[evidence_id] = record
            self._by_key[key] = evidence_id
            added.append(record)
        return LedgerIngestReport(
            added=tuple(added),
            duplicates=tuple(duplicates),
            skipped_reasons=tuple(skipped),
            truncated=truncated,
        )

    def _add_route(self, evidence_id: str, tool: str) -> None:
        record = self._records[evidence_id]
        if tool in record.routes:
            return
        self._records[evidence_id] = replace(
            record, routes=tuple(sorted((*record.routes, tool)))
        )


@dataclass(frozen=True)
class _Candidate:
    """尚未分配编号的内部候选。"""

    path: str
    start_line: int
    end_line: int
    source_fingerprint: str
    candidate_rank: int
    excerpt: str
    excerpt_start_line: int | None = None


def _extract_candidates(
    call: ToolCall,
    result: ToolResult,
    *,
    excerpt_chars: int,
    focus_text: str = "",
) -> tuple[list[_Candidate], list[str]]:
    """按工具类型拆出候选；形状不对的条目只记录原因，不抛异常。"""
    candidates: list[_Candidate] = []
    reasons: list[str] = []
    if call.name == "read_file":
        raw_items = [result.data]
    else:
        raw_items = list(_sequence_field(result.data, ("results", "evidence")))
    for rank, raw in enumerate(raw_items, start=1):
        candidate, reason = _normalize(
            raw, rank, excerpt_chars=excerpt_chars, focus_text=focus_text
        )
        if candidate is None:
            reasons.append(reason)
            continue
        candidates.append(candidate)
    return candidates, reasons


def _sequence_field(data: Mapping[str, object], names: tuple[str, ...]) -> tuple[object, ...]:
    for name in names:
        value = data.get(name)
        if isinstance(value, (list, tuple)):
            return tuple(value)
    return ()


def _normalize(
    raw: object, rank: int, *, excerpt_chars: int, focus_text: str = ""
) -> tuple[_Candidate | None, str]:
    if not isinstance(raw, Mapping):
        return None, "不是对象"
    path = raw.get("path") if "path" in raw else raw.get("relative_path")
    start_line = raw.get("start_line")
    end_line = raw.get("end_line")
    text = raw.get("text")
    if not isinstance(path, str) or not path.strip():
        return None, "path 缺失"
    if not _safe_relative(path):
        return None, "path 越界或格式非法"
    if not _positive_int(start_line) or not _positive_int(end_line) or end_line < start_line:
        return None, "行号非法"
    if not isinstance(text, str):
        return None, "text 缺失"
    tokens = _focus_tokens(focus_text)
    offset = _anchor_offset(text, tokens)
    return (
        _Candidate(
            path=path.strip().replace("\\", "/"),
            start_line=start_line,
            end_line=end_line,
            source_fingerprint=_fingerprint(path, start_line, end_line, text),
            candidate_rank=rank,
            excerpt=text[offset : offset + excerpt_chars],
            excerpt_start_line=(start_line + text[:offset].count("\n")) if offset else None,
        ),
        "",
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _safe_relative(path: str) -> bool:
    """只做纯字符串检查，不访问文件系统。"""
    candidate = path.replace("\\", "/").strip()
    if not candidate or candidate.startswith("/"):
        return False
    if "\x00" in candidate:
        return False
    parts = candidate.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _fingerprint(path: str, start_line: int, end_line: int, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{start_line}:{end_line}".encode())
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:20]


def _query_label(call: ToolCall) -> str:
    """给证据打一个可公开的来源标签，不含仓库正文。"""
    for name in ("question", "symbol_id", "path"):
        value = call.arguments.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return call.name
