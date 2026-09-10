"""构建只用于导航和预算选择的轻量仓库地图。"""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from codeinsight.domain.change import RepositoryMap
from codeinsight.ingestion.path_policy import safe_path, safe_relative_path
from codeinsight.ingestion.scanner import scan_repository

MAP_VERSION = "repository-map-v2"
MAX_MAP_LIMIT = 1000
MAX_TOKEN_BUDGET = 32_000
_INCLUDES = frozenset({"files", "symbols", "imports"})


@dataclass(frozen=True)
class RepositoryMapPage:
    """一个有界的导航页；其中所有仓库内容都是不可信数据。"""

    repo_id: str
    repo_fingerprint: str
    map_version: str
    index_version: str
    items: tuple[dict[str, object], ...]
    next_cursor: str | None
    truncated: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "repo_fingerprint": self.repo_fingerprint,
            "map_version": self.map_version,
            "index_version": self.index_version,
            "items": [dict(item) for item in self.items],
            "next_cursor": self.next_cursor,
            "truncated": self.truncated,
            "untrusted": True,
            "purpose": "navigation_only",
        }


def build_repository_map(
    root: str | Path,
    *,
    repo_id: str | None = None,
    index_version: str = MAP_VERSION,
    token_budget: int = 0,
) -> RepositoryMap:
    """读取仓库文件并提取稳定的文件、符号和 import 信息。

    优先使用 ``git ls-files``，避免把依赖、构建目录和未纳入仓库的临时文件
    混入地图；非 Git 目录回退到项目现有 scanner。目标仓库内容只作为数据
    解析，绝不 import 或执行。
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"仓库根路径不是目录：{root_path}")
    paths = _tracked_paths(root_path)
    if paths is None:
        paths = tuple(item.relative_path for item in scan_repository(root_path).files)

    symbols: list[str] = []
    imports: list[tuple[str, str]] = []
    summaries: list[tuple[str, str]] = []
    fingerprint = hashlib.sha256()
    for relative_path in sorted(paths):
        try:
            path = safe_path(root_path, relative_path)
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, PermissionError, ValueError):
            continue
        fingerprint.update(relative_path.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(text.encode("utf-8"))
        file_symbols, file_imports = _parse_python(relative_path, text)
        symbols.extend(file_symbols)
        imports.extend(file_imports)
        summaries.append((relative_path, _file_summary(relative_path, text, file_symbols)))

    resolved_repo_id = repo_id or hashlib.sha256(str(root_path).encode("utf-8")).hexdigest()[:20]
    return RepositoryMap(
        index_version=index_version,
        repo_id=resolved_repo_id,
        symbols=tuple(symbols),
        imports=tuple(imports),
        file_summaries=tuple(summaries),
        token_budget=token_budget,
        repo_fingerprint=fingerprint.hexdigest(),
    )


def repository_fingerprint(root: str | Path) -> str:
    """计算地图缓存使用的源码指纹，不执行被分析仓库。"""
    return build_repository_map(root).repo_fingerprint


def query_repository_map(
    root: str | Path,
    *,
    path_prefix: str | None = None,
    symbol_query: str | None = None,
    include: Sequence[str] = ("files", "symbols", "imports"),
    max_files: int = 200,
    max_symbols: int = 400,
    max_imports: int = 400,
    cursor: str | None = None,
    token_budget: int = 8000,
    repo_map: RepositoryMap | None = None,
) -> RepositoryMapPage:
    """对 RepositoryMap 做过滤、分页和输出预算限制。"""
    selected = tuple(include)
    if not selected or any(item not in _INCLUDES for item in selected):
        raise ValueError("include 只能包含 files、symbols、imports")
    for name, value in (
        ("max_files", max_files),
        ("max_symbols", max_symbols),
        ("max_imports", max_imports),
        ("token_budget", token_budget),
    ):
        if value < 0 or (name != "token_budget" and value == 0):
            raise ValueError(f"{name} 超出允许范围")
        if name == "token_budget" and value > MAX_TOKEN_BUDGET:
            raise ValueError(f"{name} 超出允许范围")
        if name != "token_budget" and value > MAX_MAP_LIMIT:
            raise ValueError(f"{name} 超出允许范围")
    normalized_prefix = _normalize_prefix(path_prefix)
    normalized_symbol_query = (symbol_query or "").strip().casefold()
    current = repo_map or build_repository_map(root)
    items = _items(
        current,
        include=selected,
        path_prefix=normalized_prefix,
        symbol_query=normalized_symbol_query,
        max_files=max_files,
        max_symbols=max_symbols,
        max_imports=max_imports,
    )
    filters = {
        "path_prefix": normalized_prefix,
        "symbol_query": normalized_symbol_query,
        "include": list(selected),
        "max_files": max_files,
        "max_symbols": max_symbols,
        "max_imports": max_imports,
        "token_budget": token_budget,
    }
    start = _cursor_offset(cursor, current, filters)
    if start > len(items):
        raise ValueError("cursor 已超出当前地图范围")
    page_items: list[dict[str, object]] = []
    used = _json_size({"items": page_items, "repo_id": current.repo_id})
    for item in items[start:]:
        item_size = _json_size(item)
        if token_budget and page_items and used + item_size > token_budget * 4:
            break
        page_items.append(item)
        used += item_size
    if not page_items and start < len(items):
        page_items.append(items[start])
    end = start + len(page_items)
    truncated = end < len(items)
    next_cursor = _encode_cursor(current, filters, end) if truncated else None
    return RepositoryMapPage(
        repo_id=current.repo_id,
        repo_fingerprint=current.repo_fingerprint,
        map_version=MAP_VERSION,
        index_version=current.index_version,
        items=tuple(page_items),
        next_cursor=next_cursor,
        truncated=truncated,
    )


def _items(
    repo_map: RepositoryMap,
    *,
    include: Sequence[str],
    path_prefix: str | None,
    symbol_query: str,
    max_files: int,
    max_symbols: int,
    max_imports: int,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    if "files" in include:
        for path, summary in repo_map.file_summaries:
            if _path_matches(path, path_prefix):
                items.append(
                    {
                        "path": path,
                        "kind": "file",
                        "line_hint": 1,
                        "module": _module(path),
                        "summary": summary,
                    }
                )
                if sum(item["kind"] == "file" for item in items) >= max_files:
                    break
    if "symbols" in include:
        count = 0
        for value in repo_map.symbols:
            path, symbol = value.split("::", 1)
            if not _path_matches(path, path_prefix) or (
                symbol_query and symbol_query not in symbol.casefold()
            ):
                continue
            parent = symbol.rsplit(".", 1)[0] if "." in symbol else None
            items.append(
                {
                    "path": path,
                    "kind": "symbol",
                    "symbol": symbol,
                    "parent": parent,
                    "line_hint": None,
                    "module": _module(path),
                }
            )
            count += 1
            if count >= max_symbols:
                break
    if "imports" in include:
        count = 0
        for path, imported in repo_map.imports:
            if not _path_matches(path, path_prefix) or (
                symbol_query and symbol_query not in imported.casefold()
            ):
                continue
            items.append(
                {
                    "path": path,
                    "kind": "import",
                    "module": imported,
                    "line_hint": None,
                }
            )
            count += 1
            if count >= max_imports:
                break
    items.sort(key=lambda item: (str(item.get("path", "")), str(item.get("kind", "")), str(item)))
    return items


def _normalize_prefix(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return safe_relative_path(value.strip().replace("\\", "/")).rstrip("/")


def _path_matches(path: str, prefix: str | None) -> bool:
    return prefix is None or path == prefix or path.startswith(f"{prefix}/")


def _module(path: str) -> str:
    return ".".join(Path(path).with_suffix("").parts)


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _encode_cursor(repo_map: RepositoryMap, filters: Mapping[str, object], offset: int) -> str:
    payload = {
        "version": 1,
        "repo_id": repo_map.repo_id,
        "repo_fingerprint": repo_map.repo_fingerprint,
        "map_version": MAP_VERSION,
        "filters": dict(filters),
        "offset": offset,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _cursor_offset(
    cursor: str | None,
    repo_map: RepositoryMap,
    filters: Mapping[str, object],
) -> int:
    if cursor is None:
        return 0
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError, binascii.Error) as error:
        raise ValueError("cursor 格式无效") from error
    if not isinstance(payload, dict) or any(
        payload.get(key) != value
        for key, value in (
            ("repo_id", repo_map.repo_id),
            ("repo_fingerprint", repo_map.repo_fingerprint),
            ("map_version", MAP_VERSION),
            ("filters", dict(filters)),
        )
    ):
        raise ValueError("cursor 与当前仓库、版本或筛选条件不匹配")
    offset = payload.get("offset")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("cursor offset 无效")
    return offset


def _tracked_paths(root: Path) -> tuple[str, ...] | None:
    """返回 Git 已跟踪的相对路径；不是 Git 仓库时返回 None。"""
    try:
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top_level).resolve() != root:
            return None
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    values = completed.stdout.decode("utf-8", errors="strict").split("\0")
    return tuple(value for value in values if value)


def _parse_python(relative_path: str, text: str) -> tuple[list[str], list[tuple[str, str]]]:
    if not relative_path.lower().endswith(".py"):
        return [], []
    try:
        tree = ast.parse(text, filename=relative_path)
    except (SyntaxError, ValueError):
        return [], []
    symbols: list[str] = []
    for node, parents in _named_nodes(tree):
        names = [*parents, node.name]
        symbols.append(f"{relative_path}::" + ".".join(names))
    imports: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((relative_path, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                imports.append((relative_path, f"{module}:{alias.name}"))
    return symbols, imports


def _named_nodes(tree: ast.Module) -> Iterable[tuple[ast.AST, list[str]]]:
    def visit(nodes: Iterable[ast.AST], parents: list[str]):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node, parents
                body = getattr(node, "body", ())
                yield from visit(body, [*parents, node.name])

    yield from visit(tree.body, [])


def _file_summary(relative_path: str, text: str, symbols: list[str]) -> str:
    line_count = len(text.replace("\r\n", "\n").replace("\r", "\n").splitlines())
    names = ", ".join(item.rsplit("::", 1)[-1] for item in symbols)
    suffix = f" symbols={names}" if names else ""
    return f"path={relative_path} lines={line_count}{suffix}"
