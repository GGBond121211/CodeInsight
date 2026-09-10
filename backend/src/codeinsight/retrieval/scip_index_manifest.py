"""SCIP 索引导出的小型身份 manifest，不保存源码或向量。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ScipIndexManifest:
    repo_id: str
    repo_fingerprint: str
    index_version: str
    generator_version: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ScipIndexManifest:
        values = {
            name: payload.get(name)
            for name in ("repo_id", "repo_fingerprint", "index_version", "generator_version")
        }
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("SCIP 索引身份字段不完整")
        return cls(**values)  # type: ignore[arg-type]
