"""RepositoryMap 的只读解析测试。"""

import subprocess

from codeinsight.retrieval.repository_map import build_repository_map


def test_map_uses_git_tracked_files_and_extracts_symbols_and_imports(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(
        "import os\n"
        "from pkg import helper\n"
        "\n"
        "class Service:\n"
        "    def run(self):\n"
        "        return helper()\n",
        encoding="utf-8",
    )
    (repo / "ignored.py").write_text("def should_not_appear():\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "app.py"], check=True, capture_output=True)

    result = build_repository_map(repo, repo_id="demo", index_version="v1", token_budget=200)

    assert result.repo_id == "demo"
    assert result.index_version == "v1"
    assert result.token_budget == 200
    assert result.symbols == (
        "app.py::Service",
        "app.py::Service.run",
    )
    assert result.imports == (
        ("app.py", "os"),
        ("app.py", "pkg:helper"),
    )
    assert result.file_summaries[0][0] == "app.py"
    assert "ignored.py" not in result.file_summaries[0][0]


def test_non_git_directory_falls_back_to_supported_scanner(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (repo / "notes.rst").write_text("not indexed", encoding="utf-8")

    result = build_repository_map(repo)

    assert result.file_summaries == (("app.py", "path=app.py lines=2 symbols=run"),)
