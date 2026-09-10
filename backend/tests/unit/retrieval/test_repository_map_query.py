"""2.1.0 RepositoryMap 的筛选、分页和身份约束测试。"""

import pytest

from codeinsight.retrieval.repository_map import (
    build_repository_map,
    query_repository_map,
)


def test_map_page_preserves_identity_and_cursor_filters(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")

    repo_map = build_repository_map(tmp_path)
    first = query_repository_map(
        tmp_path,
        include=("files",),
        token_budget=1,
        repo_map=repo_map,
    )
    second = query_repository_map(
        tmp_path,
        include=("files",),
        token_budget=1,
        cursor=first.next_cursor,
        repo_map=repo_map,
    )

    assert first.repo_fingerprint == second.repo_fingerprint == repo_map.repo_fingerprint
    assert first.items[0]["path"] == "a.py"
    assert second.items[0]["path"] == "b.py"
    assert first.next_cursor is not None


def test_map_page_filters_symbols_and_rejects_mismatched_cursor(tmp_path) -> None:
    (tmp_path / "service.py").write_text(
        "class CheckoutService:\n"
        "    def checkout(self):\n"
        "        return True\n",
        encoding="utf-8",
    )
    repo_map = build_repository_map(tmp_path)
    page = query_repository_map(
        tmp_path,
        symbol_query="checkout",
        include=("symbols",),
        max_symbols=10,
        repo_map=repo_map,
    )
    paged = query_repository_map(
        tmp_path,
        include=("symbols",),
        max_symbols=10,
        token_budget=1,
        repo_map=repo_map,
    )

    assert [item["symbol"] for item in page.items] == [
        "CheckoutService",
        "CheckoutService.checkout",
    ]
    with pytest.raises(ValueError, match="筛选条件不匹配"):
        query_repository_map(
            tmp_path,
            symbol_query="other",
            include=("symbols",),
            cursor=paged.next_cursor or "invalid",
            repo_map=repo_map,
        )
