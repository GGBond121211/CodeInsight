"""Immutable value types returned by repository scanning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFile:
    """A text file read from the scanned repository."""

    relative_path: str
    text: str


@dataclass(frozen=True)
class SkippedFile:
    """A path that was deliberately not read, together with the reason."""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    """The outcome of a repository scan."""

    files: tuple[SourceFile, ...]
    skipped: tuple[SkippedFile, ...]


@dataclass(frozen=True)
class SourceChunk:
    """A source slice with stable lines and optional language-aware context."""

    relative_path: str
    start_line: int
    end_line: int
    text: str
    symbol_path: str | None = None
