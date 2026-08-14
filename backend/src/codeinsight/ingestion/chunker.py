"""把源码文件切分为固定大小且互不重叠的行块。"""

from codeinsight.domain.source import ScanResult, SourceChunk, SourceFile


def chunk_source_file(source: SourceFile, *, max_lines: int = 80) -> tuple[SourceChunk, ...]:
    """把 *source* 切分为最多包含 *max_lines* 个逻辑行的块。

    切分前会把 CRLF 和单独的 CR 换行统一为 LF。
    行号以原文件的逻辑行为准，从 1 开始并且两端都包含。块文本使用 LF 连接，
    不会人为添加末尾换行。
    """
    if max_lines <= 0:
        raise ValueError("max_lines 必须是正整数")
    lines = _logical_lines(source.text)
    if not lines:
        return ()
    chunks: list[SourceChunk] = []
    for start in range(0, len(lines), max_lines):
        end = min(start + max_lines, len(lines))
        chunks.append(
            SourceChunk(
                relative_path=source.relative_path,
                start_line=start + 1,
                end_line=end,
                text="\n".join(lines[start:end]),
            )
        )
    return tuple(chunks)


def chunk_scan_result(scan_result: ScanResult, *, max_lines: int = 80) -> tuple[SourceChunk, ...]:
    """切分 *scan_result* 中的每个文件，并保留文件顺序。"""
    if max_lines <= 0:
        raise ValueError("max_lines 必须是正整数")
    chunks: list[SourceChunk] = []
    for source in scan_result.files:
        chunks.extend(chunk_source_file(source, max_lines=max_lines))
    return tuple(chunks)


def _logical_lines(text: str) -> list[str]:
    """把 CRLF 和 CR 统一为 LF 后返回文件的逻辑行。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    lines = normalized.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines
