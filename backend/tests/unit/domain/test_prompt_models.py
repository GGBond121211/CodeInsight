from __future__ import annotations

import pytest

from codeinsight.domain.prompt_models import PromptVariable, PromptVersion


def _version(**overrides: object) -> PromptVersion:
    values: dict[str, object] = {
        "version_id": "v1",
        "template_id": "t1",
        "version": "1.0.0",
        "content": "回答 {question}",
        "variables_schema": (PromptVariable("question", max_length=20),),
    }
    values.update(overrides)
    return PromptVersion(**values)  # type: ignore[arg-type]


def test_prompt_variables_are_typed_and_length_limited() -> None:
    assert _version().render({"question": "解释入口"}) == "回答 解释入口"
    with pytest.raises(ValueError, match="应为 string"):
        _version().render({"question": 1})
    with pytest.raises(ValueError, match="长度"):
        _version().render({"question": "x" * 21})


def test_prompt_render_does_not_treat_json_braces_as_variables() -> None:
    version = _version(content='输出 {question}，JSON: {"ok": true}')
    assert version.render({"question": "任务"}) == '输出 任务，JSON: {"ok": true}'


def test_prompt_rejects_missing_unexpected_and_unknown_variables() -> None:
    with pytest.raises(ValueError, match="缺少变量"):
        _version().render({})
    with pytest.raises(ValueError, match="未登记变量"):
        _version().render({"question": "任务", "extra": "x"})
    with pytest.raises(ValueError, match="未登记变量"):
        _version(content="{question} {unknown}").render({"question": "任务"})
