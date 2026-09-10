"""固定位置、只读、带仓库指纹校验的 SCIP 引用索引读取器。

当前不引入任意 protobuf/外部命令执行。项目约定的索引导出为
``.codeinsight/scip/index.json``，由受控的 SCIP 导出步骤生成，工具只读取它。
没有该导出时返回 INDEX_NOT_FOUND；指纹不匹配时返回 INDEX_STALE。
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from pathlib import Path

from codeinsight.domain.code_intelligence import (
    AVAILABLE,
    INDEX_NOT_FOUND,
    INDEX_STALE,
    INVALID_RESPONSE,
    CodeIntelligenceResult,
    CodeLocation,
)
from codeinsight.ingestion.path_policy import safe_path
from codeinsight.retrieval.repository_map import repository_fingerprint
from codeinsight.retrieval.scip_index_manifest import ScipIndexManifest

SCIP_INDEX_RELATIVE_PATH = ".codeinsight/scip/index.json"


class ScipReferenceReader:
    """读取受控 JSON 导出的 SCIP 引用，不执行仓库内容。"""

    def __init__(self, repository_root: str | Path) -> None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root 必须是目录")
        self.root = root

    def references(
        self,
        *,
        symbol_id: str | None = None,
        path: str | None = None,
        line: int | None = None,
        column: int | None = None,
        index_version: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> CodeIntelligenceResult:
        index_path = safe_path(self.root, SCIP_INDEX_RELATIVE_PATH)
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return CodeIntelligenceResult(
                INDEX_NOT_FOUND,
                message=f"SCIP 索引不存在：{SCIP_INDEX_RELATIVE_PATH}",
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            return CodeIntelligenceResult(INVALID_RESPONSE, message="SCIP 索引无法读取")
        if not isinstance(payload, dict):
            return CodeIntelligenceResult(INVALID_RESPONSE, message="SCIP 索引根节点必须是 object")
        current_fingerprint = repository_fingerprint(self.root)
        try:
            manifest = ScipIndexManifest.from_payload(payload)
        except ValueError as error:
            return CodeIntelligenceResult(INVALID_RESPONSE, message=str(error))
        if manifest.repo_id != _repository_id(self.root):
            return CodeIntelligenceResult(INDEX_STALE, message="SCIP 索引 repo_id 不匹配")
        if manifest.repo_fingerprint != current_fingerprint:
            return CodeIntelligenceResult(
                INDEX_STALE,
                message="SCIP 索引与当前仓库指纹不匹配",
                metadata={"index_path": SCIP_INDEX_RELATIVE_PATH},
            )
        if index_version is not None and index_version != manifest.index_version:
            return CodeIntelligenceResult(INDEX_STALE, message="请求的 SCIP index_version 不匹配")
        if limit <= 0 or limit > 200:
            return CodeIntelligenceResult(
                INVALID_RESPONSE, message="SCIP references limit 超出范围"
            )
        raw_references = payload.get("references")
        if not isinstance(raw_references, list):
            return CodeIntelligenceResult(INVALID_RESPONSE, message="SCIP 索引缺少 references 数组")
        try:
            selected = _select_references(
                self.root,
                raw_references,
                symbol_id=symbol_id,
                path=path,
                line=line,
                column=column,
            )
            start = _cursor_offset(
                cursor, manifest.index_version, symbol_id, path, line, column
            )
        except (ValueError, PermissionError) as error:
            return CodeIntelligenceResult(INVALID_RESPONSE, message=str(error))
        if start > len(selected):
            return CodeIntelligenceResult(INVALID_RESPONSE, message="SCIP cursor 超出结果范围")
        page = tuple(selected[start : start + limit])
        end = start + len(page)
        next_cursor = None
        if end < len(selected):
            next_cursor = _encode_cursor(
                manifest.index_version, symbol_id, path, line, column, end
            )
        return CodeIntelligenceResult(
            AVAILABLE,
            locations=page,
            metadata={
                "index_path": SCIP_INDEX_RELATIVE_PATH,
                "repo_id": manifest.repo_id,
                "repo_fingerprint": manifest.repo_fingerprint,
                "index_version": manifest.index_version,
                "generator_version": manifest.generator_version,
                "next_cursor": next_cursor,
            },
        )


def _select_references(
    root: Path,
    raw_references: list[object],
    *,
    symbol_id: str | None,
    path: str | None,
    line: int | None,
    column: int | None,
) -> list[CodeLocation]:
    selected: list[CodeLocation] = []
    normalized_path = None
    if path is not None:
        normalized_path = safe_path(root, path).relative_to(root).as_posix()
    for raw in raw_references:
        if not isinstance(raw, Mapping):
            raise ValueError("SCIP reference 必须是 object")
        raw_symbol = raw.get("symbol_id")
        raw_path = raw.get("path")
        raw_line = raw.get("line")
        raw_column = raw.get("column")
        if not all(
            (
                isinstance(raw_symbol, str) and raw_symbol.strip(),
                isinstance(raw_path, str) and raw_path.strip(),
                isinstance(raw_line, int) and not isinstance(raw_line, bool) and raw_line >= 1,
                isinstance(raw_column, int)
                and not isinstance(raw_column, bool)
                and raw_column >= 0,
            )
        ):
            raise ValueError("SCIP reference 字段无效")
        relative = safe_path(root, raw_path).relative_to(root).as_posix()
        if symbol_id is not None and raw_symbol != symbol_id:
            continue
        if normalized_path is not None and relative != normalized_path:
            continue
        if line is not None and raw_line != line:
            continue
        if column is not None and raw_column != column:
            continue
        end_line = raw.get("end_line")
        end_column = raw.get("end_column")
        if end_line is not None and (not isinstance(end_line, int) or end_line < raw_line):
            raise ValueError("SCIP reference end_line 无效")
        if end_column is not None and (not isinstance(end_column, int) or end_column < 0):
            raise ValueError("SCIP reference end_column 无效")
        selected.append(
            CodeLocation(
                relative,
                raw_line,
                raw_column,
                end_line=end_line,
                end_column=end_column,
                symbol_id=raw_symbol,
                source="scip",
                repo_id=_repository_id(root),
                repo_fingerprint=repository_fingerprint(root),
            )
        )
    selected.sort(
        key=lambda item: (
            item.path,
            item.start_line,
            item.start_column,
            item.symbol_id or "",
        )
    )
    return selected


def _cursor_payload(
    index_version: str,
    symbol_id: str | None,
    path: str | None,
    line: int | None,
    column: int | None,
    offset: int,
) -> dict[str, object]:
    return {
        "index_version": index_version,
        "symbol_id": symbol_id,
        "path": path,
        "line": line,
        "column": column,
        "offset": offset,
    }


def _encode_cursor(
    index_version: str,
    symbol_id: str | None,
    path: str | None,
    line: int | None,
    column: int | None,
    offset: int,
) -> str:
    encoded = json.dumps(
        _cursor_payload(index_version, symbol_id, path, line, column, offset),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _cursor_offset(
    cursor: str | None,
    index_version: str,
    symbol_id: str | None,
    path: str | None,
    line: int | None,
    column: int | None,
) -> int:
    if cursor is None:
        return 0
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError, binascii.Error) as error:
        raise ValueError("SCIP cursor 格式无效") from error
    if not isinstance(payload, dict):
        raise ValueError("SCIP cursor 必须是 object")
    expected = _cursor_payload(index_version, symbol_id, path, line, column, payload.get("offset"))
    if any(
        payload.get(key) != value
        for key, value in expected.items()
        if key != "offset"
    ):
        raise ValueError("SCIP cursor 与当前筛选条件不匹配")
    offset = payload.get("offset")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("SCIP cursor offset 无效")
    return offset


def _repository_id(root: Path) -> str:
    import hashlib

    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
