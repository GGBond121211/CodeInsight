from pathlib import Path

from codeinsight.agent.tool_loop import ToolCall
from codeinsight.infrastructure.tool_executor import ToolExecutor


def test_executor_keeps_read_and_write_inside_repository(tmp_path: Path):
    source = tmp_path / "src.py"
    source.write_text("value = 1\n", encoding="utf-8")
    executor = ToolExecutor(tmp_path)

    escaped_read = executor.execute(ToolCall("r1", "read_file", {"path": "../secret.txt"}))
    escaped_patch = executor.execute(
        ToolCall("p1", "generate_patch", {"path": "../secret.txt", "new_content": "secret"})
    )
    proposed = executor.execute(
        ToolCall("p2", "generate_patch", {"path": "src.py", "new_content": "value = 2\n"})
    )
    blocked_apply = executor.execute(
        ToolCall(
            "a1",
            "apply_patch_isolated",
            {
                "patch_id": proposed.data.get("patch_id", ""),
                "approval_token": "a",
                "checkpoint_id": "c",
            },
        )
    )

    assert not escaped_read.ok and escaped_read.error_code == "PERMISSION"
    assert not escaped_patch.ok and escaped_patch.error_code == "PERMISSION"
    assert proposed.ok
    assert not blocked_apply.ok
    assert blocked_apply.error_code == "VALIDATION"
    assert source.read_text(encoding="utf-8") == "value = 1\n"


def test_executor_has_no_arbitrary_shell_tool():
    executor = ToolExecutor(Path.cwd())
    result = executor.execute(ToolCall("x1", "run_shell", {"command": "type secret.txt"}))
    assert not result.ok
    assert result.error_code == "VALIDATION"


def test_executor_rejects_sensitive_files_and_schema_ranges(tmp_path: Path):
    (tmp_path / ".env.production").write_text("SECRET=value", encoding="utf-8")
    executor = ToolExecutor(tmp_path)
    sensitive = executor.execute(ToolCall("s1", "read_file", {"path": ".env.production"}))
    invalid_limit = executor.execute(
        ToolCall("s2", "search_repository", {"question": "x", "limit": 0})
    )
    assert not sensitive.ok and sensitive.error_code == "PERMISSION"
    assert not invalid_limit.ok and invalid_limit.error_code == "VALIDATION"


def test_executor_normalizes_a_bound_repository_prefix_without_widening_scope(
    tmp_path: Path,
):
    root = tmp_path / "sample_repo"
    root.mkdir()
    source = root / "src.py"
    source.write_text("value = 1\n", encoding="utf-8")
    executor = ToolExecutor(root)

    prefixed = executor.execute(
        ToolCall("prefixed", "read_file", {"path": "tests/fixtures/sample_repo/src.py"})
    )
    absolute = executor.execute(
        ToolCall("absolute", "read_file", {"path": str(source.resolve())})
    )

    assert prefixed.ok and prefixed.data["path"] == "src.py"
    assert absolute.ok and absolute.data["path"] == "src.py"


def test_executor_reports_relative_path_guidance_for_missing_model_path(tmp_path: Path):
    root = tmp_path / "sample_repo"
    root.mkdir()
    executor = ToolExecutor(root)

    result = executor.execute(
        ToolCall("missing", "read_file", {"path": "sample_repo/missing.py"})
    )

    assert not result.ok
    assert result.error_code == "NOT_FOUND"
    assert "相对于已绑定的仓库根目录" in (result.error_message or "")
