"""固定的 LSP server 注册表。

注册表是权限边界：调用方只能选择支持矩阵中的语言，不能从工具参数传入
任意可执行文件、参数或工作目录。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from codeinsight.domain.code_intelligence import (
    AVAILABLE,
    NOT_CONFIGURED,
    UNSUPPORTED_LANGUAGE,
)


@dataclass(frozen=True)
class LspServerResolution:
    language: str
    status: str
    server_name: str
    command: tuple[str, ...] = ()
    message: str = ""


class FixedLspRegistry:
    """只登记项目明确支持的固定 server 命令。"""

    _SERVERS: dict[str, tuple[str, tuple[str, ...]]] = {
        "python": ("pyright", ("--langserver", "--stdio")),
        "typescript": ("typescript-language-server", ("--stdio",)),
        "typescriptreact": ("typescript-language-server", ("--stdio",)),
    }

    def resolve(self, language: str) -> LspServerResolution:
        normalized = language.strip().lower()
        selected = self._SERVERS.get(normalized)
        if selected is None:
            return LspServerResolution(
                language=normalized,
                status=UNSUPPORTED_LANGUAGE,
                server_name="",
                message=f"未登记 LSP 语言：{normalized or '(empty)'}",
            )
        server_name, arguments = selected
        executable = shutil.which(server_name)
        if executable is None:
            return LspServerResolution(
                language=normalized,
                status=NOT_CONFIGURED,
                server_name=server_name,
                message=f"未配置固定 LSP server：{server_name}",
            )
        return LspServerResolution(
            language=normalized,
            status=AVAILABLE,
            server_name=server_name,
            command=(executable, *arguments),
        )


def language_for_path(path: str) -> str | None:
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    return {
        "py": "python",
        "ts": "typescript",
        "tsx": "typescriptreact",
    }.get(suffix)
