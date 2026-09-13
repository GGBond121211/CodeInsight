from pathlib import Path
from types import SimpleNamespace

from codeinsight.agent.tool_loop import ToolCall
from codeinsight.infrastructure import tool_executor as executor_module
from codeinsight.infrastructure.tool_executor import ToolExecutor


def test_executor_reads_paths_that_start_with_the_repository_name(tmp_path: Path):
    """仓库目录名与顶层包同名时（如 httpx），检索回来的路径必须原样可读。"""

    root = tmp_path / "httpx"
    package = root / "httpx"
    package.mkdir(parents=True)
    (package / "_client.py").write_text("def send():\n    return 1\n", encoding="utf-8")
    executor = ToolExecutor(root)

    result = executor.execute(
        ToolCall("same-name", "read_file", {"path": "httpx/_client.py", "start_line": 1})
    )

    assert result.ok, result.error_message
    assert result.data["path"] == "httpx/_client.py"


def test_executor_reuses_retrieval_state_across_searches(tmp_path: Path, monkeypatch):
    """同一进程内第二次检索复用同一份索引，不再重建整个仓库。"""

    root = tmp_path / "repo"
    root.mkdir()
    (root / "src.py").write_text("value = 1\n", encoding="utf-8")
    tools = SimpleNamespace(embed=lambda texts: None, rerank=lambda *a, **k: None)
    executor = ToolExecutor(root, embedding_factory=lambda: tools, reranker_factory=lambda: tools)
    prepared: list[str] = []
    calls: list[dict[str, object]] = []

    def fake_prepare(root_arg, *, semantic_embed, **kwargs):
        prepared.append("build")
        return ("index", "store")

    def fake_search(root_arg, question, **kwargs):
        calls.append(kwargs)
        return ()

    monkeypatch.setattr(executor_module, "prepare_search_state", fake_prepare)
    monkeypatch.setattr(executor_module, "search_repository", fake_search)

    first = executor.execute(ToolCall("c1", "search_repository", {"question": "a"}))
    second = executor.execute(ToolCall("c2", "search_repository", {"question": "b"}))

    assert first.ok and second.ok
    assert prepared == ["build"]
    assert [call["semantic_index"] for call in calls] == ["index", "index"]
    assert [call["semantic_store"] for call in calls] == ["store", "store"]


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
