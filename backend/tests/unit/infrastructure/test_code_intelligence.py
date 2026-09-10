"""LSP/SCIP 只读工具的能力缺失和索引身份测试。"""

import hashlib
import json

from codeinsight.agent.tool_loop import ToolCall
from codeinsight.domain.code_intelligence import (
    INDEX_NOT_FOUND,
    INDEX_STALE,
    NOT_CONFIGURED,
    UNSUPPORTED_LANGUAGE,
)
from codeinsight.infrastructure.lsp_registry import FixedLspRegistry
from codeinsight.infrastructure.tool_executor import ToolExecutor
from codeinsight.retrieval.repository_map import repository_fingerprint


def test_fixed_lsp_registry_does_not_accept_unknown_languages(monkeypatch) -> None:
    monkeypatch.setattr("codeinsight.infrastructure.lsp_registry.shutil.which", lambda _: None)
    registry = FixedLspRegistry()

    assert registry.resolve("ruby").status == UNSUPPORTED_LANGUAGE
    assert registry.resolve("python").status == NOT_CONFIGURED


def test_repository_map_cache_invalidates_after_source_change(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("def first():\n    return 1\n", encoding="utf-8")
    executor = ToolExecutor(tmp_path)

    first = executor.execute(
        ToolCall("map-1", "get_repository_map", {"include": ["symbols"]})
    )
    source.write_text(
        "def first():\n    return 1\n\ndef second():\n    return 2\n",
        encoding="utf-8",
    )
    second = executor.execute(
        ToolCall("map-2", "get_repository_map", {"include": ["symbols"]})
    )

    assert first.ok and second.ok
    assert first.data["repo_fingerprint"] != second.data["repo_fingerprint"]
    assert any(item["symbol"] == "second" for item in second.data["items"])


def test_executor_reports_unconfigured_lsp_without_running_a_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("codeinsight.infrastructure.lsp_registry.shutil.which", lambda _: None)
    source = tmp_path / "app.py"
    source.write_text("value = 1\n", encoding="utf-8")
    executor = ToolExecutor(tmp_path)

    result = executor.execute(
        ToolCall("lsp", "lsp_definition", {"path": "app.py", "line": 1, "column": 0})
    )

    assert not result.ok
    assert result.error_code in {NOT_CONFIGURED, UNSUPPORTED_LANGUAGE}
    assert result.data["purpose"] == "navigation_only"


def test_scip_reader_reports_missing_then_stale_index(tmp_path) -> None:
    executor = ToolExecutor(tmp_path)
    missing = executor.execute(
        ToolCall("missing", "scip_references", {"symbol_id": "scip.test"})
    )
    assert not missing.ok and missing.error_code == INDEX_NOT_FOUND

    index_path = tmp_path / ".codeinsight" / "scip" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "index_version": "scip-json-v1",
                "repo_id": hashlib.sha256(str(tmp_path.resolve()).encode()).hexdigest()[:20],
                "repo_fingerprint": "not-current",
                "generator_version": "scip-export-v1",
                "references": [],
            }
        ),
        encoding="utf-8",
    )
    stale = executor.execute(
        ToolCall("stale", "scip_references", {"symbol_id": "scip.test"})
    )
    assert not stale.ok and stale.error_code == INDEX_STALE


def test_scip_reader_returns_location_from_fingerprint_matched_export(tmp_path) -> None:
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    index_path = tmp_path / ".codeinsight" / "scip" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "index_version": "scip-json-v1",
                "repo_id": hashlib.sha256(str(tmp_path.resolve()).encode()).hexdigest()[:20],
                "repo_fingerprint": repository_fingerprint(tmp_path),
                "generator_version": "scip-export-v1",
                "references": [
                    {
                        "symbol_id": "scip.test/run",
                        "path": "app.py",
                        "line": 1,
                        "column": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    executor = ToolExecutor(tmp_path)

    result = executor.execute(
        ToolCall("refs", "scip_references", {"symbol_id": "scip.test/run"})
    )

    assert result.ok
    assert result.data["locations"] == [
        {
            "path": "app.py",
            "start_line": 1,
            "start_column": 4,
            "symbol_id": "scip.test/run",
            "source": "scip",
            "repo_id": result.data["locations"][0]["repo_id"],
            "repo_fingerprint": result.data["locations"][0]["repo_fingerprint"],
        }
    ]
