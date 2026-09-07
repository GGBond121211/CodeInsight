"""固定 validation profile 的 Docker Sandbox 执行器。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

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


@dataclass(frozen=True)
class SandboxPreflightResult:
    """固定校验开始前的 Docker daemon 与镜像可用性结果。"""

    profile: str
    available: bool
    error_class: str | None = None
    message_excerpt: str = ""


class SandboxRunner(Protocol):
    def run(self, profile: str, workspace_path: str | Path) -> SandboxResult: ...

    def preflight(self, profile: str) -> SandboxPreflightResult: ...


class SandboxCleanupError(RuntimeError):
    """Container lifetime is unknown; callers must not race it with rollback."""


class DockerSandbox:
    """不接受模型命令，只接受登记过的 profile 名称。"""

    def __init__(self, *, image: str = "codeinsight-sandbox:py312-pytest-v1") -> None:
        self.image = image

    def preflight(self, profile: str) -> SandboxPreflightResult:
        """在生成/审批补丁前确认 Docker daemon 和固定镜像可访问。"""
        get_validation_profile(profile)
        try:
            daemon = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except FileNotFoundError:
            return SandboxPreflightResult(
                profile, False, "SANDBOX_UNAVAILABLE", "Docker 可执行文件不存在"
            )
        except subprocess.TimeoutExpired:
            return SandboxPreflightResult(
                profile, False, "SANDBOX_UNAVAILABLE", "Docker daemon 连接超时"
            )
        if daemon.returncode != 0:
            error_class, message = _classify_docker_failure(
                daemon.stdout + "\n" + daemon.stderr,
                default="SANDBOX_UNAVAILABLE",
            )
            return SandboxPreflightResult(profile, False, error_class, message)

        try:
            image = subprocess.run(
                ["docker", "image", "inspect", self.image],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return SandboxPreflightResult(
                profile, False, "SANDBOX_UNAVAILABLE", "Sandbox 镜像检查超时"
            )
        if image.returncode != 0:
            error_class, message = _classify_docker_failure(
                image.stdout + "\n" + image.stderr,
                default="SANDBOX_IMAGE_MISSING",
            )
            return SandboxPreflightResult(profile, False, error_class, message)
        return SandboxPreflightResult(profile, True)

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
            container_name = f"codeinsight-check-{uuid4().hex}"
            argv = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
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
                self._stop_container(container_name)
                return SandboxResult(
                    profile,
                    selected.display_commands,
                    False,
                    "TIMEOUT",
                    "检查超过时间限制",
                )
            if completed.returncode != 0:
                output = (completed.stdout + "\n" + completed.stderr).strip()
                error_class, message = _classify_docker_failure(
                    output, default="CHECK_FAILED"
                )
                last_error = SandboxResult(
                    profile,
                    selected.display_commands,
                    False,
                    error_class,
                    message,
                )
                break
        return last_error or SandboxResult(profile, selected.display_commands, True)

    @staticmethod
    def _stop_container(name: str) -> None:
        try:
            removed = subprocess.run(
                ["docker", "rm", "--force", name],
                check=False, capture_output=True, text=True, timeout=10,
            )
            if removed.returncode == 0:
                return
            # --rm may already have removed it; a daemon failure is not proof of absence.
            inspected = subprocess.run(
                ["docker", "container", "ls", "--all", "--quiet", "--filter", f"name=^{name}$"],
                check=False, capture_output=True, text=True, timeout=10,
            )
            if inspected.returncode == 0 and not inspected.stdout.strip():
                return
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SandboxCleanupError(f"Sandbox cleanup unconfirmed: {name}") from error
        raise SandboxCleanupError(f"Sandbox cleanup unconfirmed: {name}")


def _excerpt(value: str) -> str:
    return redact_sensitive(value)[-2000:]


def _classify_docker_failure(value: str, *, default: str) -> tuple[str, str]:
    """把 Docker 基础设施错误与被校验代码错误区分开。"""
    safe = redact_sensitive(value).strip()
    lowered = safe.lower()
    if "access is denied" in lowered or "permission denied" in lowered:
        return "SANDBOX_PERMISSION_DENIED", _excerpt(safe)
    return default, _excerpt(safe) or "Docker Sandbox 执行失败"
