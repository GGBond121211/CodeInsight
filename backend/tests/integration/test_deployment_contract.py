"""Static and local-tool checks for the Step 9 delivery contract."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPOSE = PROJECT_ROOT / "ops" / "docker-compose.yml"


def test_step9_assets_are_present_and_pinned() -> None:
    required = (
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / ".dockerignore",
        COMPOSE,
        PROJECT_ROOT / "ops" / "gateway-proxy" / "nginx.conf",
        PROJECT_ROOT / "ops" / "otel" / "otel-collector-config.yml",
        PROJECT_ROOT / "ops" / "prometheus" / "prometheus.yml",
        PROJECT_ROOT / "k8s" / "base" / "kustomization.yaml",
        PROJECT_ROOT / "k8s" / "base" / "api-gateway-worker.yaml",
        PROJECT_ROOT / "k8s" / "base" / "dependencies.yaml",
    )
    assert all(path.is_file() for path in required)
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "USER 10001:10001" in dockerfile
    assert "mysql:" in compose and "redis:" in compose
    assert "gateway-a" in compose and "gateway-b" in compose
    assert "@sha256:" in compose


def test_compose_config_is_valid_when_docker_is_available() -> None:
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "config"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        pytest.skip("Docker CLI unavailable")
    assert result.returncode == 0, result.stderr


def test_kubernetes_manifest_has_local_dry_run_entrypoint() -> None:
    kustomization = (
        PROJECT_ROOT / "k8s" / "base" / "kustomization.yaml"
    ).read_text(encoding="utf-8")
    assert "api-gateway-worker.yaml" in kustomization
    assert "dependencies.yaml" in kustomization
