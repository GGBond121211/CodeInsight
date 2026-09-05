"""构建只用于导航和预算选择的轻量仓库地图。"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

from codeinsight.domain.change import RepositoryMap
from codeinsight.ingestion.scanner import scan_repository


def build_repository_map(
    root: str | Path,
    *,
    repo_id: str | None = None,
    index_version: str = "repository-map-v1",
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
    for relative_path in sorted(paths):
        path = root_path / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
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
    )


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
