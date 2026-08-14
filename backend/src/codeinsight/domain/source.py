"""仓库扫描返回的不可变值对象。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFile:
    """从已扫描仓库中读取的文本文件。"""

    relative_path: str
    text: str


@dataclass(frozen=True)
class SkippedFile:
    """有意跳过读取的路径及其原因。"""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    """仓库扫描结果。"""

    files: tuple[SourceFile, ...]
    skipped: tuple[SkippedFile, ...]


@dataclass(frozen=True)
class SourceChunk:
    """带有稳定行号和可选语言上下文的源码切片。"""

    relative_path: str
    start_line: int
    end_line: int
    text: str
    symbol_path: str | None = None
