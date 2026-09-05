"""按 Python 顶层代码结构切块，并为超长结构提供固定行回退。"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from codeinsight.domain.source import SourceChunk, SourceFile
from codeinsight.ingestion.chunker import chunk_source_file


@dataclass(frozen=True)
class _Span:
    start_line: int
    end_line: int
    symbol_path: str | None


def chunk_structural_source_file(
    source: SourceFile,
    *,
    fallback_max_lines: int = 80,
    max_structure_lines: int = 120,
) -> tuple[SourceChunk, ...]:
    """按顶层 class/function/module 区域切分一个文件。

    结构块保持 1-based、闭区间行号。非 Python 文件、语法错误文件和没有
    可定位顶层结构的文件使用固定行切分，保证摄取链路不会因为解析器失败
    丢掉源码。
    """
    if max_structure_lines <= 0:
        raise ValueError("max_structure_lines 必须是正整数")
    if not source.relative_path.lower().endswith(".py"):
        return chunk_source_file(source, max_lines=fallback_max_lines)
    try:
        tree = ast.parse(source.text, filename=source.relative_path)
    except (SyntaxError, ValueError):
        return chunk_source_file(source, max_lines=fallback_max_lines)

    lines = _logical_lines(source.text)
    if not lines or not tree.body:
        return chunk_source_file(source, max_lines=fallback_max_lines)
    spans = _top_level_spans(tree, len(lines))
    if not spans:
        return chunk_source_file(source, max_lines=fallback_max_lines)

    chunks: list[SourceChunk] = []
    cursor = 1
    for span in spans:
        if cursor < span.start_line:
            chunks.extend(
                _fixed_range_chunks(
                    source,
                    lines,
                    cursor,
                    span.start_line - 1,
                    fallback_max_lines,
                    None,
                )
            )
        size = span.end_line - span.start_line + 1
        if size <= max_structure_lines:
            chunks.append(_make_chunk(source, lines, span))
        else:
            chunks.extend(
                _fixed_range_chunks(
                    source,
                    lines,
                    span.start_line,
                    span.end_line,
                    fallback_max_lines,
                    span.symbol_path,
                )
            )
        cursor = span.end_line + 1
    if cursor <= len(lines):
        chunks.extend(
            _fixed_range_chunks(
                source,
                lines,
                cursor,
                len(lines),
                fallback_max_lines,
                None,
            )
        )
    return tuple(chunks)


def chunk_structural_scan_result(
    scan_result,
    *,
    fallback_max_lines: int = 80,
    max_structure_lines: int = 120,
) -> tuple[SourceChunk, ...]:
    """按文件顺序切分扫描结果。"""
    if fallback_max_lines <= 0:
        raise ValueError("fallback_max_lines 必须是正整数")
    chunks: list[SourceChunk] = []
    for source in scan_result.files:
        chunks.extend(
            chunk_structural_source_file(
                source,
                fallback_max_lines=fallback_max_lines,
                max_structure_lines=max_structure_lines,
            )
        )
    return tuple(chunks)


def _top_level_spans(tree: ast.Module, total_lines: int) -> tuple[_Span, ...]:
    spans: list[_Span] = []
    cursor = 1
    for node in tree.body:
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators = getattr(node, "decorator_list", ())
            for decorator in decorators:
                decorator_line = getattr(decorator, "lineno", None)
                if isinstance(decorator_line, int):
                    start_line = min(start_line, decorator_line - 1)
            symbol_path = getattr(node, "name", None)
            if cursor < start_line:
                spans.append(_Span(cursor, start_line - 1, "<module>"))
            spans.append(_Span(max(1, start_line), end_line, symbol_path))
            cursor = end_line + 1
        else:
            continue
    if cursor <= total_lines:
        spans.append(_Span(cursor, total_lines, "<module>"))
    return tuple(spans)


def _fixed_range_chunks(
    source: SourceFile,
    lines: list[str],
    start_line: int,
    end_line: int,
    max_lines: int,
    symbol_path: str | None,
) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    for start in range(start_line, end_line + 1, max_lines):
        end = min(start + max_lines - 1, end_line)
        chunks.append(
            SourceChunk(
                source.relative_path,
                start,
                end,
                "\n".join(lines[start - 1 : end]),
                symbol_path,
            )
        )
    return chunks


def _make_chunk(source: SourceFile, lines: list[str], span: _Span) -> SourceChunk:
    return SourceChunk(
        source.relative_path,
        span.start_line,
        span.end_line,
        "\n".join(lines[span.start_line - 1 : span.end_line]),
        span.symbol_path,
    )


def _logical_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    lines = normalized.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines
