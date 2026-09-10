"""LSP/SCIP 导航结果的共同边界。

这些结果只回答“下一步去哪里看”，不等于 Evidence。调用方必须继续通过
``read_file`` 读取仓库文本并校验行号；外部索引和语言服务器输出都按不可信
数据处理。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

AVAILABLE = "AVAILABLE"
NOT_CONFIGURED = "NOT_CONFIGURED"
INDEX_NOT_FOUND = "INDEX_NOT_FOUND"
INDEX_STALE = "INDEX_STALE"
UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
TIMEOUT = "TIMEOUT"
INVALID_RESPONSE = "INVALID_RESPONSE"

CODE_INTELLIGENCE_STATUSES = frozenset(
    {
        AVAILABLE,
        NOT_CONFIGURED,
        INDEX_NOT_FOUND,
        INDEX_STALE,
        UNSUPPORTED_LANGUAGE,
        TIMEOUT,
        INVALID_RESPONSE,
    }
)


@dataclass(frozen=True)
class CodeLocation:
    """一个可继续导航的仓库位置。行号对外保持 1-based。"""

    path: str
    start_line: int
    start_column: int
    end_line: int | None = None
    end_column: int | None = None
    symbol_id: str | None = None
    source: str | None = None
    repo_id: str | None = None
    repo_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("CodeLocation path 不能为空")
        if self.start_line < 1 or self.start_column < 0:
            raise ValueError("CodeLocation 的行列号无效")
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("CodeLocation end_line 不能早于 start_line")
        if self.source not in {None, "lsp", "scip"}:
            raise ValueError("CodeLocation source 必须是 lsp 或 scip")

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path,
            "start_line": self.start_line,
            "start_column": self.start_column,
        }
        if self.end_line is not None:
            result["end_line"] = self.end_line
        if self.end_column is not None:
            result["end_column"] = self.end_column
        if self.symbol_id:
            result["symbol_id"] = self.symbol_id
        if self.source:
            result["source"] = self.source
        if self.repo_id:
            result["repo_id"] = self.repo_id
        if self.repo_fingerprint:
            result["repo_fingerprint"] = self.repo_fingerprint
        return result


@dataclass(frozen=True)
class CodeIntelligenceResult:
    """LSP/SCIP 工具对外的受控返回值。"""

    status: str
    locations: tuple[CodeLocation, ...] = ()
    message: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in CODE_INTELLIGENCE_STATUSES:
            raise ValueError(f"不支持的 code intelligence status：{self.status}")

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "locations": [item.as_dict() for item in self.locations],
            "message": self.message,
            **dict(self.metadata),
            "untrusted": True,
            "purpose": "navigation_only",
        }
