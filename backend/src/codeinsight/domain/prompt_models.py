"""可回放 Prompt 的领域模型。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

PROMPT_ACTIVE = "active"
PROMPT_DRAFT = "draft"
PROMPT_RETIRED = "retired"
SUPPORTED_PROMPT_STATUSES = frozenset({PROMPT_ACTIVE, PROMPT_DRAFT, PROMPT_RETIRED})

PROMPT_SYSTEM = "system"
PROMPT_USER = "user"
PROMPT_REVIEW = "review"
SUPPORTED_PROMPT_TYPES = frozenset({PROMPT_SYSTEM, PROMPT_USER, PROMPT_REVIEW})
SUPPORTED_VARIABLE_TYPES = frozenset({"string", "integer", "number", "boolean", "json"})
VARIABLE_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _require_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} 不能为空")


def hash_prompt_variables(variables: Mapping[str, object]) -> str:
    """只保存稳定哈希，不把变量原文写进运行记录。"""
    encoded = json.dumps(dict(variables), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptTemplate:
    """Prompt 的稳定身份，不直接保存某一次内容。"""

    template_id: str
    name: str
    scene_code: str
    prompt_type: str
    status: str = PROMPT_ACTIVE

    def __post_init__(self) -> None:
        _require_text(self.template_id, "template_id")
        _require_text(self.name, "name")
        _require_text(self.scene_code, "scene_code")
        if self.prompt_type not in SUPPORTED_PROMPT_TYPES:
            raise ValueError(f"不支持的 Prompt 类型：{self.prompt_type}")
        if self.status not in SUPPORTED_PROMPT_STATUSES:
            raise ValueError(f"不支持的 Prompt 状态：{self.status}")


@dataclass(frozen=True)
class PromptVariable:
    """一个可在模型调用前校验的 Prompt 变量。"""

    name: str
    value_type: str = "string"
    max_length: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "变量名")
        if self.value_type not in SUPPORTED_VARIABLE_TYPES:
            raise ValueError(f"不支持的 Prompt 变量类型：{self.value_type}")
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("max_length 必须为正")

    def validate(self, value: object) -> str:
        type_matches = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "json": isinstance(value, (dict, list, str, int, float, bool)) or value is None,
        }
        if not type_matches[self.value_type]:
            raise ValueError(f"Prompt 变量 {self.name} 应为 {self.value_type}")
        if self.value_type == "json":
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        if self.max_length is not None and len(rendered) > self.max_length:
            raise ValueError(
                f"Prompt 变量 {self.name} 长度 {len(rendered)} 超过上限 {self.max_length}"
            )
        return rendered


@dataclass(frozen=True)
class PromptVersion:
    """一份不可变 Prompt 内容。"""

    version_id: str
    template_id: str
    version: str
    content: str
    variables_schema: tuple[PromptVariable | tuple[str, str] | tuple[str, str, int], ...] = ()
    model_config: tuple[tuple[str, str], ...] = ()
    created_by: str = "local"
    change_reason: str = ""
    status: str = PROMPT_ACTIVE

    def __post_init__(self) -> None:
        _require_text(self.version_id, "version_id")
        _require_text(self.template_id, "template_id")
        _require_text(self.version, "version")
        _require_text(self.content, "content")
        _require_text(self.created_by, "created_by")
        if self.status not in SUPPORTED_PROMPT_STATUSES:
            raise ValueError(f"不支持的 Prompt 状态：{self.status}")
        variable_names: set[str] = set()
        for variable in self._normalized_variables():
            if variable.name in variable_names:
                raise ValueError(f"Prompt 变量重复：{variable.name}")
            variable_names.add(variable.name)

    def _normalized_variables(self) -> tuple[PromptVariable, ...]:
        """兼容早期的 ``(name, type)`` schema 写法。"""
        normalized: list[PromptVariable] = []
        for raw in self.variables_schema:
            if isinstance(raw, PromptVariable):
                normalized.append(raw)
                continue
            if len(raw) == 2:
                normalized.append(PromptVariable(name=raw[0], value_type=raw[1]))
            else:
                normalized.append(
                    PromptVariable(name=raw[0], value_type=raw[1], max_length=raw[2])
                )
        return tuple(normalized)

    def render(self, variables: dict[str, object]) -> str:
        """校验变量后渲染内容，不把变量原文写进 PromptRun。"""
        variables_schema = self._normalized_variables()
        expected_names = {variable.name for variable in variables_schema}
        missing = sorted(expected_names - variables.keys())
        if missing:
            raise ValueError(f"Prompt 缺少变量：{missing}")
        unexpected = sorted(variables.keys() - expected_names)
        if unexpected:
            raise ValueError(f"Prompt 收到未登记变量：{unexpected}")

        rendered_values: dict[str, str] = {}
        for variable in variables_schema:
            rendered_values[variable.name] = variable.validate(variables[variable.name])
        placeholders = set(VARIABLE_PATTERN.findall(self.content))
        unknown_placeholders = sorted(placeholders - expected_names)
        if unknown_placeholders:
            raise ValueError(f"Prompt 使用了未登记变量：{unknown_placeholders}")
        return VARIABLE_PATTERN.sub(
            lambda match: rendered_values[match.group(1)],
            self.content,
        )


@dataclass(frozen=True)
class PromptRelease:
    """把某个版本发布到环境和租户范围。"""

    release_id: str
    template_id: str
    version_id: str
    environment: str
    tenant_id: str = "*"
    traffic_ratio: float = 1.0
    status: str = PROMPT_ACTIVE

    def __post_init__(self) -> None:
        _require_text(self.release_id, "release_id")
        _require_text(self.template_id, "template_id")
        _require_text(self.version_id, "version_id")
        _require_text(self.environment, "environment")
        _require_text(self.tenant_id, "tenant_id")
        if not 0 < self.traffic_ratio <= 1:
            raise ValueError("traffic_ratio 必须大于 0 且不超过 1")
        if self.status not in SUPPORTED_PROMPT_STATUSES:
            raise ValueError(f"不支持的 Release 状态：{self.status}")


@dataclass(frozen=True)
class PromptRun:
    """一次模型调用使用的 Prompt 版本记录，只保存变量摘要。"""

    run_id: str
    request_id: str
    template_id: str
    version_id: str
    release_id: str
    variables_hash: str
    model_version: str = "unknown"
    strategy_version: str = "unknown"
    input_tokens: int | None = None
    output_tokens: int | None = None
    output_ref: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "run_id"),
            (self.request_id, "request_id"),
            (self.template_id, "template_id"),
            (self.version_id, "version_id"),
            (self.release_id, "release_id"),
            (self.variables_hash, "variables_hash"),
            (self.model_version, "model_version"),
            (self.strategy_version, "strategy_version"),
        ):
            _require_text(value, label)
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens 不能为负")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output_tokens 不能为负")
