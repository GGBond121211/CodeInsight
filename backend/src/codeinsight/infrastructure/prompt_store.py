"""Prompt 模板、版本、Release 和运行记录的进程内实现。"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass

from codeinsight.domain.prompt_models import (
    PROMPT_ACTIVE,
    PromptRelease,
    PromptRun,
    PromptTemplate,
    PromptVersion,
    hash_prompt_variables,
)


@dataclass(frozen=True)
class PromptResolution:
    """一次解析得到的版本、Release 和渲染文本。"""

    version: PromptVersion
    release: PromptRelease
    content: str


def variables_hash(variables: Mapping[str, object]) -> str:
    """只对变量做稳定哈希，不保存变量原文。"""
    return hash_prompt_variables(variables)


class InMemoryPromptStore:
    """Prompt 注册表的最小实现，适合本地运行和确定性测试。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._templates: dict[str, PromptTemplate] = {}
        self._versions: dict[str, PromptVersion] = {}
        self._releases: dict[str, PromptRelease] = {}
        self._runs: dict[str, PromptRun] = {}

    def save_template(self, template: PromptTemplate) -> None:
        with self._lock:
            self._templates[template.template_id] = template

    def save_version(self, version: PromptVersion) -> None:
        with self._lock:
            if version.template_id not in self._templates:
                raise ValueError(f"Prompt 模板不存在：{version.template_id}")
            self._versions[version.version_id] = version

    def save_release(self, release: PromptRelease) -> None:
        with self._lock:
            template = self._templates.get(release.template_id)
            if template is None:
                raise ValueError(f"Prompt 模板不存在：{release.template_id}")
            version = self._versions.get(release.version_id)
            if version is None:
                raise ValueError(f"Prompt 版本不存在：{release.version_id}")
            if version.template_id != template.template_id:
                raise ValueError("Prompt Release 的模板与版本不匹配")
            self._releases[release.release_id] = release

    def resolve(
        self,
        prompt_name: str,
        *,
        environment: str,
        tenant_id: str,
    ) -> tuple[PromptVersion, PromptRelease]:
        with self._lock:
            template = None
            for candidate in self._templates.values():
                if candidate.name == prompt_name:
                    template = candidate
                    break
            if template is None:
                raise KeyError(f"Prompt 模板不存在：{prompt_name}")

            candidates: list[PromptRelease] = []
            for release in self._releases.values():
                if release.template_id != template.template_id:
                    continue
                if release.environment != environment:
                    continue
                if release.status != PROMPT_ACTIVE:
                    continue
                if release.tenant_id not in {tenant_id, "*"}:
                    continue
                if self._in_traffic(release, tenant_id):
                    candidates.append(release)
            candidates.sort(key=lambda item: (item.tenant_id == "*", item.release_id))
            if not candidates:
                raise KeyError(
                    f"Prompt 没有匹配的 Release：name={prompt_name}, environment={environment}"
                )
            release = candidates[0]
            version = self._versions[release.version_id]
            if version.status != PROMPT_ACTIVE:
                raise ValueError(f"Prompt 版本不是 active：{version.version_id}")
            return version, release

    def render(
        self,
        prompt_name: str,
        *,
        environment: str,
        tenant_id: str,
        variables: Mapping[str, object],
    ) -> PromptResolution:
        version, release = self.resolve(
            prompt_name,
            environment=environment,
            tenant_id=tenant_id,
        )
        return PromptResolution(
            version=version,
            release=release,
            content=version.render(dict(variables)),
        )

    def record_run(self, run: PromptRun) -> None:
        with self._lock:
            if run.run_id in self._runs:
                raise ValueError(f"PromptRun 已存在：{run.run_id}")
            self._runs[run.run_id] = run

    def get_run(self, run_id: str) -> PromptRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def _in_traffic(self, release: PromptRelease, tenant_id: str) -> bool:
        if release.traffic_ratio >= 1:
            return True
        seed = f"{tenant_id}:{release.release_id}".encode()
        bucket = int.from_bytes(hashlib.sha256(seed).digest()[:4], "big") / 2**32
        return bucket < release.traffic_ratio
