"""正文与引用一致性检查的单元测试（Q-012 阶段 C 复测发现的问题）。"""

from __future__ import annotations

from codeinsight.application.answer_citations import uncited_mentioned_paths
from codeinsight.domain.answer import AnswerCitation


def _citation(evidence_id: str, path: str) -> AnswerCitation:
    return AnswerCitation(
        evidence_id=evidence_id, relative_path=path, start_line=1, end_line=2
    )


def test_flags_a_path_the_answer_names_but_does_not_cite() -> None:
    """复测里的原样复现：正文写 httpx/_client.py，引用挂到别的文件。"""

    answer = "Client 类定义在 httpx/_client.py。"
    citations = (
        _citation("E3", "httpx/_main.py"),
        _citation("E5", "httpx/_api.py"),
    )

    flagged = uncited_mentioned_paths(
        answer,
        citations,
        ["httpx/_client.py", "httpx/_main.py", "httpx/_api.py"],
    )

    assert flagged == ("httpx/_client.py",)


def test_accepts_an_answer_that_cites_what_it_names() -> None:
    answer = "Client 类定义在 httpx/_client.py:594。"
    citations = (_citation("E2", "httpx/_client.py"),)

    assert uncited_mentioned_paths(
        answer, citations, ["httpx/_client.py", "httpx/_main.py"]
    ) == ()


def test_matches_a_unique_basename() -> None:
    """模型常写简称；只要这个名字在台账里唯一，就按它判断。"""

    answer = "结论来自 _config.py。"

    assert uncited_mentioned_paths(
        answer, (), ["httpx/_config.py", "httpx/_client.py"]
    ) == ("httpx/_config.py",)


def test_ignores_ambiguous_basenames() -> None:
    """同名文件（如 __init__.py）判不准就不判，避免误伤正常答案。"""

    answer = "看 __init__.py。"

    assert uncited_mentioned_paths(
        answer, (), ["pkg/a/__init__.py", "pkg/b/__init__.py"]
    ) == ()


def test_ignores_paths_the_answer_never_mentions() -> None:
    # 不提任何文件时不该凭空指认；提了却没引用才是问题（上一条用例）。
    assert uncited_mentioned_paths("send 定义在应用层。", (), ["src/app.py"]) == ()
