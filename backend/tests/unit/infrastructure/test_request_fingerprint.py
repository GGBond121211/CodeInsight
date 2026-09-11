from __future__ import annotations

from codeinsight.infrastructure.model_gateway import GatewayRequest
from codeinsight.infrastructure.request_fingerprint import request_fingerprints


def _request(**overrides: object) -> GatewayRequest:
    values: dict[str, object] = {
        "request_id": "req-1",
        "tenant_id": "default",
        "user_id": "local",
        "scene": "explain",
        "prompt_version": "code-understanding-v1",
        "messages": (
            {"role": "system", "content": "安全规则"},
            {"role": "user", "content": "解释入口"},
        ),
        "estimated_input_tokens": 10,
        "reserved_output_tokens": 50,
    }
    values.update(overrides)
    return GatewayRequest(**values)  # type: ignore[arg-type]


def test_stable_prefix_ignores_changes_in_the_dynamic_suffix() -> None:
    base = request_fingerprints(_request(), model_id="deepseek-v4-flash")
    longer = request_fingerprints(
        _request(
            messages=(
                {"role": "system", "content": "安全规则"},
                {"role": "user", "content": "解释入口"},
                {"role": "tool", "content": "第一轮结果"},
                {"role": "user", "content": "接着看第二个函数"},
            )
        ),
        model_id="deepseek-v4-flash",
    )

    assert base.stable_prefix == longer.stable_prefix
    assert base.dynamic_suffix != longer.dynamic_suffix
    assert base.request != longer.request


def test_compaction_only_replaces_the_dynamic_suffix() -> None:
    before = request_fingerprints(
        _request(
            messages=(
                {"role": "system", "content": "安全规则"},
                {"role": "user", "content": "第 1 轮"},
                {"role": "user", "content": "第 2 轮"},
            ),
            compaction_boundary_id=None,
        ),
        model_id="deepseek-v4-flash",
    )
    after = request_fingerprints(
        _request(
            messages=(
                {"role": "system", "content": "安全规则"},
                {"role": "user", "content": "摘要：第 1、2 轮"},
            ),
            compaction_boundary_id="cmp-session-2-1",
        ),
        model_id="deepseek-v4-flash",
    )

    # 压缩替换的是后缀，前缀原样保留，所以前缀仍然可以被复用。
    assert before.stable_prefix == after.stable_prefix
    assert before.dynamic_suffix != after.dynamic_suffix
    assert before.surface != after.surface


def test_stable_prefix_changes_when_an_invariant_changes() -> None:
    base = request_fingerprints(_request(), model_id="deepseek-v4-flash")

    assert base.stable_prefix != request_fingerprints(
        _request(), model_id="gpt-5.4"
    ).stable_prefix
    assert base.stable_prefix != request_fingerprints(
        _request(prompt_version="code-understanding-v2"), model_id="deepseek-v4-flash"
    ).stable_prefix
    assert base.stable_prefix != request_fingerprints(
        _request(tools=({"type": "function"},)), model_id="deepseek-v4-flash"
    ).stable_prefix
    assert base.stable_prefix != request_fingerprints(
        _request(response_format={"type": "json_object"}), model_id="deepseek-v4-flash"
    ).stable_prefix


def test_surface_describes_shape_not_content() -> None:
    first = request_fingerprints(
        _request(
            messages=(
                {"role": "system", "content": "安全规则"},
                {"role": "user", "content": "内容甲"},
            )
        ),
        model_id="deepseek-v4-flash",
    )
    second = request_fingerprints(
        _request(
            messages=(
                {"role": "system", "content": "安全规则"},
                {"role": "user", "content": "内容乙"},
            )
        ),
        model_id="deepseek-v4-flash",
    )

    assert first.surface == second.surface
    assert first.request != second.request


def test_surface_carries_the_compaction_boundary() -> None:
    without = request_fingerprints(_request(), model_id="deepseek-v4-flash")
    with_boundary = request_fingerprints(
        _request(compaction_boundary_id="cmp-session-9-3"),
        model_id="deepseek-v4-flash",
    )

    assert without.surface != with_boundary.surface
    # 边界本身不进入缓存键用的完整请求指纹之外，也不改变前缀。
    assert without.stable_prefix == with_boundary.stable_prefix


def test_fingerprints_are_deterministic() -> None:
    first = request_fingerprints(_request(), model_id="deepseek-v4-flash")
    second = request_fingerprints(_request(), model_id="deepseek-v4-flash")

    assert first == second
    assert len(first.request) == 64
    assert len(first.stable_prefix) == 64
    assert len(first.dynamic_suffix) == 64
    assert len(first.surface) == 64

