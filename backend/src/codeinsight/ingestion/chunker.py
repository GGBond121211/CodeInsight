"""把源码文件切分为固定大小的行块，并可选择保留相邻块重叠。"""

import math

from codeinsight.domain.source import ScanResult, SourceChunk, SourceFile
from codeinsight.domain.tokens import estimate_tokens


def chunk_source_file(
    source: SourceFile,
    *,
    max_lines: int = 80,
    overlap_ratio: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[SourceChunk, ...]:
    """把 *source* 切分为最多包含 *max_lines* 个逻辑行的块。

    切分前会把 CRLF 和单独的 CR 换行统一为 LF。
    行号以原文件的逻辑行为准，从 1 开始并且两端都包含。块文本使用 LF 连接，
    不会人为添加末尾换行。``overlap_ratio`` 按 ``max_lines`` 向下取整计算
    重叠行数；默认值 0 保持原来的互不重叠行为。

    ``max_tokens`` 是额外的块级 token 上限（Q-006）。行块按行号切好之后，
    仍然超过上限的块会再按行细分为多个子块，因此上限对调用方是硬约束，
    代价是子块之间不再保留 ``overlap_ratio`` 重叠。单行本身超限时该行
    保持整行，不做字符级切分：代码行的完整性优先于 token 上限。
    默认 None 表示只按行切分，行为与 2.1.0 之前完全一致。
    """
    if max_lines <= 0:
        raise ValueError("max_lines 必须是正整数")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens 必须是正整数")
    overlap_lines = _overlap_lines(max_lines, overlap_ratio)
    lines = _logical_lines(source.text)
    if not lines:
        return ()
    chunks: list[SourceChunk] = []
    stride = max_lines - overlap_lines
    for start in range(0, len(lines), stride):
        end = min(start + max_lines, len(lines))
        for piece_start, piece_end in _token_bounded_spans(lines[start:end], max_tokens):
            chunks.append(
                SourceChunk(
                    relative_path=source.relative_path,
                    start_line=start + piece_start + 1,
                    end_line=start + piece_end,
                    text="\n".join(lines[start + piece_start : start + piece_end]),
                )
            )
    return tuple(chunks)


def chunk_scan_result(
    scan_result: ScanResult,
    *,
    max_lines: int = 80,
    overlap_ratio: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[SourceChunk, ...]:
    """切分 *scan_result* 中的每个文件，并保留文件顺序。"""
    if max_lines <= 0:
        raise ValueError("max_lines 必须是正整数")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens 必须是正整数")
    _overlap_lines(max_lines, overlap_ratio)
    chunks: list[SourceChunk] = []
    for source in scan_result.files:
        chunks.extend(
            chunk_source_file(
                source,
                max_lines=max_lines,
                overlap_ratio=overlap_ratio,
                max_tokens=max_tokens,
            )
        )
    return tuple(chunks)


def _token_bounded_spans(
    block: list[str], max_tokens: int | None
) -> list[tuple[int, int]]:
    """返回 *block* 内不超过 token 上限的行区间 ``[start, end)``。

    ``max_tokens`` 为 None 时整块原样返回。单行本身超限时保留整行，
    否则会切出无法定位的半个语句。
    """
    if max_tokens is None:
        return [(0, len(block))]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(block):
        end = start + 1
        while end < len(block):
            candidate = "\n".join(block[start : end + 1])
            if estimate_tokens(candidate) > max_tokens:
                break
            end += 1
        spans.append((start, end))
        start = end
    return spans


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
