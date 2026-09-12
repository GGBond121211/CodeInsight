"""Q-012：工具目录导出物与示例问题字段的契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeinsight.agent.code_understanding_tool_loop import CODE_UNDERSTANDING_TOOLS
from codeinsight.infrastructure.tool_registry import (
    CATALOG_VERSION,
    ToolSpec,
    build_default_registry,
)

CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "codeinsight" / "tools" / "catalog.yaml"
)


def test_catalog_matches_the_code_directory() -> None:
    """导出物漂移必须让测试失败，否则目录和代码会出现两套事实。"""

    assert CATALOG_PATH.exists(), f"缺少导出物：{CATALOG_PATH}"
    assert CATALOG_PATH.read_text(encoding="utf-8") == (
        build_default_registry().export_catalog_text()
    )


def test_catalog_covers_every_registered_tool() -> None:
    text = build_default_registry().export_catalog_text()
    assert f'catalog_version: "{CATALOG_VERSION}"' in text
    for name in build_default_registry()._specs:
        assert f'name: "{name}"' in text


def test_read_only_code_understanding_tools_carry_example_questions() -> None:
    """示例问题是问题原型库的原料；缺了它，快路径就没有可比的文本锚点。"""

    registry = build_default_registry()
    for name in CODE_UNDERSTANDING_TOOLS:
        spec = registry.get(name)
        assert spec is not None, f"{name} 未登记"
        assert len(spec.example_questions) >= 3, f"{name} 的示例问题少于 3 条"
        assert all(question.strip() for question in spec.example_questions)


def test_example_questions_stay_out_of_the_model_call_payload() -> None:
    """目录元数据不该跟着每次模型调用一起发出去。"""

    payload = build_default_registry().get("search_repository")
    assert payload is not None
    assert "exampleQuestions" not in payload.as_mcp()
    assert "example_questions" not in payload.as_mcp()


def test_blank_or_duplicated_example_questions_are_rejected() -> None:
    schema: dict[str, object] = {"type": "object", "properties": {}}
    with pytest.raises(ValueError):
        ToolSpec(
            "demo",
            "说明",
            schema,
            True,
            False,
            True,
            False,
            "low",
            1.0,
            "repository",
            example_questions=("  ",),
        )
    with pytest.raises(ValueError):
        ToolSpec(
            "demo",
            "说明",
            schema,
            True,
            False,
            True,
            False,
            "low",
            1.0,
            "repository",
            example_questions=("重复", "重复"),
        )
