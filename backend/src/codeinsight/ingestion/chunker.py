"""Split source files into fixed-size, non-overlapping line chunks."""

from codeinsight.domain.source import ScanResult, SourceChunk, SourceFile


def chunk_source_file(source: SourceFile, *, max_lines: int = 80) -> tuple[SourceChunk, ...]:
    """Split *source* into chunks of at most *max_lines* logical lines.

    CRLF and lone CR line endings are normalized to LF before splitting.
    Line numbers are 1-based and inclusive over the original file's logical
    lines. Chunk text is the chunk's lines joined with LF and never gets an
    artificial trailing newline.
    """
    if max_lines <= 0:
        raise ValueError("max_lines must be a positive integer")
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
    """Chunk every file in *scan_result*, preserving its file order."""
    if max_lines <= 0:
        raise ValueError("max_lines must be a positive integer")
    chunks: list[SourceChunk] = []
    for source in scan_result.files:
        chunks.extend(chunk_source_file(source, max_lines=max_lines))
    return tuple(chunks)


def _logical_lines(text: str) -> list[str]:
    """Return the file's logical lines after normalizing CRLF and CR to LF."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    lines = normalized.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines
