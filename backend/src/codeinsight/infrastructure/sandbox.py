"""固定 validation profile 的 Docker Sandbox 执行器。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from codeinsight.application.quality_guard import get_validation_profile
from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.infrastructure.redaction import redact_sensitive


@dataclass(frozen=True)
class SandboxResult:
    profile: str
    commands: tuple[str, ...]
    passed: bool
    error_class: str | None = None
    message_excerpt: str = ""


class SandboxRunner(Protocol):
    def run(self, profile: str, workspace_path: str | Path) -> SandboxResult: ...


class DockerSandbox:
    """不接受模型命令，只接受登记过的 profile 名称。"""

    def __init__(self, *, image: str = "codeinsight-sandbox:py312-pytest-v1") -> None:
        self.image = image

    def run(self, profile: str, workspace_path: str | Path) -> SandboxResult:
        with get_telemetry().span(
            "sandbox", "validation", attributes={"validation_profile": profile}
        ):
            return self._run(profile, workspace_path)

    def _run(self, profile: str, workspace_path: str | Path) -> SandboxResult:
        selected = get_validation_profile(profile)
        workspace = Path(workspace_path).resolve()
        if not workspace.is_dir():
            raise ValueError("workspace_path 必须是目录")
        last_error: SandboxResult | None = None
        for command in selected.commands:
            argv = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "1000:1000",
                "--cpus",
                "1.0",
                "--memory",
                "512m",
                "--pids-limit",
                "128",
                "--mount",
                f"type=bind,src={workspace},dst=/workspace",
                "--workdir",
                "/workspace",
                self.image,
                *command,
            ]
            try:
                completed = subprocess.run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=selected.timeout_seconds,
                )
            except FileNotFoundError:
                return SandboxResult(
                    profile,
                    selected.display_commands,
                    False,
                    "SANDBOX_UNAVAILABLE",
                    "Docker 不可用",
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    profile,
                    selected.display_commands,
                    False,
                    "TIMEOUT",
                    "检查超过时间限制",
                )
            if completed.returncode != 0:
                output = (completed.stdout + "\n" + completed.stderr).strip()
                last_error = SandboxResult(
                    profile,
                    selected.display_commands,
                    False,
                    "CHECK_FAILED",
                    _excerpt(output),
                )
                break
        return last_error or SandboxResult(profile, selected.display_commands, True)


def _excerpt(value: str) -> str:
    return redact_sensitive(value)[-2000:]
