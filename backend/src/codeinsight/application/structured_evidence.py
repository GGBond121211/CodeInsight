"""结构化结果区（Q-012）：把「在哪里、第几行」从散文里拿出来。

为什么由代码映射而不是让模型输出：模型写路径与行号时，读的人无法核验它
写的是真读到的还是编的。这里的每一行都来自某次真实工具调用的返回，模型
在整条链路上只负责措辞——它给不出、也改不动这张表。

三条边界：
    1. 路径必须是仓库内的相对路径；绝对路径与任何含 `..` 的路径直接丢弃。
    2. 行号缺失就写 None，不补 1 之类的默认值：把「没有行号」伪装成「第 1 行」
       比缺一行更糟。
    3. LSP/SCIP/RepositoryMap 的行只说明「去哪儿看」，不等于 Evidence；
       只有 Evidence Ledger 的条目带 evidence_id。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from codeinsight.agent.tool_loop import ToolResult

DEFINITION = "definition"
REFERENCE = "reference"
MAP_FILE = "map_file"
EVIDENCE = "evidence"

KIND_LABELS: dict[str, str] = {
    DEFINITION: "定义位置",
    REFERENCE: "引用位置",
    MAP_FILE: "仓库文件",
    EVIDENCE: "可引用证据",
}

DEFAULT_MAX_ROWS = 40


@dataclass(frozen=True)
class StructuredEvidenceRow:
    """一行结构化事实；字段全部来自工具返回，模型不参与。"""

    kind: str
    source_tool: str
    path: str
    symbol: str = ""
    start_line: int | None = None
    end_line: int | None = None
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KIND_LABELS:
            raise ValueError(f"不支持的 structured evidence kind：{self.kind}")
        if not self.path.strip():
            raise ValueError("structured evidence 必须有路径")
        if self.start_line is not None and self.start_line < 1:
            raise ValueError("structured evidence 的行号从 1 开始")
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line 不能早于 start_line")

    @property
    def dedupe_key(self) -> tuple[object, ...]:
        return (self.kind, self.path, self.start_line, self.end_line, self.symbol)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "kind_label": KIND_LABELS[self.kind],
            "source_tool": self.source_tool,
            "path": self.path,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "evidence_id": self.evidence_id,
        }


def is_safe_relative_path(path: str) -> bool:
    """工具返回的路径按不可信数据处理：越界路径一律不展示。"""

    if not path.strip():
        return False
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("~"):
        return False
    if ":" in normalized.split("/")[0]:
        return False
    return ".." not in normalized.split("/")


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed >= 1 else None
    return None


def _location_rows(
    result: ToolResult, kind: str, tool_name: str
) -> list[StructuredEvidenceRow]:
    rows: list[StructuredEvidenceRow] = []
    locations = result.data.get("locations")
    if not isinstance(locations, Sequence) or isinstance(locations, (str, bytes)):
        return rows
    for item in locations:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path", ""))
        if not is_safe_relative_path(path):
            continue
        start = _int_or_none(item.get("start_line"))
        end = _int_or_none(item.get("end_line"))
        symbol = str(item.get("symbol_id") or "")
        rows.append(
            StructuredEvidenceRow(
                kind=kind,
                source_tool=tool_name,
                path=path.replace("\\", "/"),
                symbol=symbol,
                start_line=start,
                end_line=end if end is not None else start,
            )
        )
    return rows


def _map_rows(result: ToolResult, tool_name: str) -> list[StructuredEvidenceRow]:
    rows: list[StructuredEvidenceRow] = []
    items = result.data.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return rows
    for item in items:
        if not isinstance(item, Mapping) or item.get("kind") != "file":
            continue
        path = str(item.get("path", ""))
        if not is_safe_relative_path(path):
            continue
        rows.append(
            StructuredEvidenceRow(
                kind=MAP_FILE,
                source_tool=tool_name,
                path=path.replace("\\", "/"),
                symbol=str(item.get("module") or ""),
            )
        )
    return rows


def build_structured_evidence(
    *,
    tool_results: Iterable[ToolResult],
    evidence: Iterable[EvidenceRecordLike],
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[tuple[StructuredEvidenceRow, ...], bool]:
    """把工具结果与证据台账映射成表格行，返回 (行, 是否被截断)。"""

    if max_rows < 1:
        raise ValueError("max_rows 必须是正整数")
    collected: list[StructuredEvidenceRow] = []
    for result in tool_results:
        if not result.ok:
            continue
        if result.tool_name == "lsp_definition":
            collected.extend(_location_rows(result, DEFINITION, result.tool_name))
        elif result.tool_name == "scip_references":
            collected.extend(_location_rows(result, REFERENCE, result.tool_name))
        elif result.tool_name == "get_repository_map":
            collected.extend(_map_rows(result, result.tool_name))
    for record in evidence:
        if not is_safe_relative_path(record.path):
            continue
        collected.append(
            StructuredEvidenceRow(
                kind=EVIDENCE,
                source_tool=record.source_tool,
                path=record.path.replace("\\", "/"),
                symbol=record.query_or_symbol,
                start_line=record.start_line,
                end_line=record.end_line,
                evidence_id=record.evidence_id,
            )
        )
    seen: set[tuple[object, ...]] = set()
    unique: list[StructuredEvidenceRow] = []
    for row in collected:
        if row.dedupe_key in seen:
            continue
        seen.add(row.dedupe_key)
        unique.append(row)
    truncated = len(unique) > max_rows
    return tuple(unique[:max_rows]), truncated


class EvidenceRecordLike(Protocol):
    """只需要这几个字段；避免 application 层反向依赖 ledger 的完整形状。"""

    evidence_id: str
    source_tool: str
    query_or_symbol: str
    path: str
    start_line: int
    end_line: int
