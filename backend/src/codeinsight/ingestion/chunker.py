"""把源码文件切分为固定大小的行块，并可选择保留相邻块重叠。"""

import math

from codeinsight.domain.source import ScanResult, SourceChunk, SourceFile


def chunk_source_file(
    source: SourceFile,
    *,
    max_lines: int = 80,
    overlap_ratio: float = 0.0,
) -> tuple[SourceChunk, ...]:
    """把 *source* 切分为最多包含 *max_lines* 个逻辑行的块。

    切分前会把 CRLF 和单独的 CR 换行统一为 LF。
    行号以原文件的逻辑行为准，从 1 开始并且两端都包含。块文本使用 LF 连接，
    不会人为添加末尾换行。``overlap_ratio`` 按 ``max_lines`` 向下取整计算
    重叠行数；默认值 0 保持原来的互不重叠行为。
    """
    if max_lines <= 0:
        raise ValueError("max_lines 必须是正整数")
    overlap_lines = _overlap_lines(max_lines, overlap_ratio)
    lines = _logical_lines(source.text)
    if not lines:
        return ()
    chunks: list[SourceChunk] = []
    stride = max_lines - overlap_lines
    for start in range(0, len(lines), stride):
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


def chunk_scan_result(
    scan_result: ScanResult,
    *,
    max_lines: int = 80,
    overlap_ratio: float = 0.0,
) -> tuple[SourceChunk, ...]:
    """切分 *scan_result* 中的每个文件，并保留文件顺序。"""
    if max_lines <= 0:
        raise ValueError("max_lines 必须是正整数")
    _overlap_lines(max_lines, overlap_ratio)
    chunks: list[SourceChunk] = []
    for source in scan_result.files:
        chunks.extend(
            chunk_source_file(
                source,
                max_lines=max_lines,
                overlap_ratio=overlap_ratio,
            )
        )
    return tuple(chunks)


def _overlap_lines(max_lines: int, overlap_ratio: float) -> int:
    """把比例转换成确定性的重叠行数，并拒绝会导致无进展的配置。"""
    if not math.isfinite(overlap_ratio) or not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio 必须是 [0, 1) 范围内的有限数")
    overlap_lines = math.floor(max_lines * overlap_ratio)
    if overlap_lines >= max_lines:
        raise ValueError("overlap_ratio 不能产生等于 max_lines 的重叠行数")
    return overlap_lines


def _logical_lines(text: str) -> list[str]:
    """把 CRLF 和 CR 统一为 LF 后返回文件的逻辑行。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    lines = normalized.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines
