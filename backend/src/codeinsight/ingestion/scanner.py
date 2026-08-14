"""读取仓库中的受支持文本文件，用于理解代码。"""

import os
from pathlib import Path

from codeinsight.domain.errors import RepositoryScanError
from codeinsight.domain.source import ScanResult, SkippedFile, SourceFile
from codeinsight.ingestion.filters import is_ignored_directory, skip_reason

REASON_READ_ERROR = "read_error"


def scan_repository(root: str | Path) -> ScanResult:
    """按稳定的相对路径顺序读取 *root* 下的受支持文本文件。"""
    root_path = Path(root)
    if not root_path.exists():
        raise RepositoryScanError(f"仓库根目录不存在：{root_path}")
    if not root_path.is_dir():
        raise RepositoryScanError(f"仓库根路径不是目录：{root_path}")

    files: list[SourceFile] = []
    skipped: list[SkippedFile] = []
    for directory, directory_names, file_names in os.walk(root_path):
        directory_names[:] = sorted(
            name for name in directory_names if not is_ignored_directory(name)
        )
        for name in sorted(file_names):
            path = Path(directory) / name
            relative = path.relative_to(root_path).as_posix()
            reason = skip_reason(relative)
            if reason is not None:
                skipped.append(SkippedFile(relative, reason))
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                skipped.append(SkippedFile(relative, REASON_READ_ERROR))
                continue
            files.append(SourceFile(relative, text))
    files.sort(key=lambda item: item.relative_path)
    skipped.sort(key=lambda item: item.relative_path)
    return ScanResult(files=tuple(files), skipped=tuple(skipped))
