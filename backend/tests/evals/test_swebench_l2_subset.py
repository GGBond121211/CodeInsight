"""SWE-bench L2 定位子集的契约测试。

保护的是 DEC-0034 与增补 R-12 的三条承诺：
    1. 按仓库配额选取，不让任何单一仓库主导评测结果；
    2. 不依赖 Docker，只做定位评测；
    3. gold 行号取 base_commit（修复前）版本，与 008 索引的状态对齐。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUBSET_PATH = Path(__file__).with_name("swebench_l2_subset.json")

EXPECTED_CASE_COUNT = 194

EXPECTED_REPOS = frozenset(
    {
        "sphinx-doc/sphinx",
        "matplotlib/matplotlib",
        "scikit-learn/scikit-learn",
        "astropy/astropy",
        "pydata/xarray",
        "pytest-dev/pytest",
        "pylint-dev/pylint",
        "psf/requests",
        "mwaskom/seaborn",
        "pallets/flask",
    }
)

EXCLUDED_REPOS = frozenset({"django/django", "sympy/sympy"})

# 单仓占比上限。django 在全量里占 46.2%，正是要避免的情况。
MAX_SINGLE_REPO_SHARE = 0.25


def load_subset() -> dict[str, Any]:
    return json.loads(SUBSET_PATH.read_text(encoding="utf-8"))


def test_subset_exists_and_parses() -> None:
    subset = load_subset()
    assert subset["schema_version"] == 1
    assert subset["layer"] == "L2"
    assert subset["task_type"] == "code_modification"


def test_case_count_is_frozen() -> None:
    subset = load_subset()
    assert subset["case_count"] == EXPECTED_CASE_COUNT
    assert len(subset["cases"]) == EXPECTED_CASE_COUNT


def test_license_is_recorded() -> None:
    """008 是 MIT 仓库，外部数据集的许可必须登记（DEC-0034）。"""
    subset = load_subset()
    assert subset["source"]["license"] == "MIT"
    assert subset["source"]["dataset"] == "princeton-nlp/SWE-bench_Verified"


def test_only_selected_repos_present() -> None:
    subset = load_subset()
    seen: set[str] = set()
    for case in subset["cases"]:
        seen.add(case["repo"])
    assert seen <= EXPECTED_REPOS
    assert not (seen & EXCLUDED_REPOS)


def test_no_repo_dominates() -> None:
    """核心约束：不允许单一仓库主导评测结果。"""
    subset = load_subset()
    total = subset["case_count"]
    for repo, count in subset["selected_repo_distribution"].items():
        share = count / total
        assert share <= MAX_SINGLE_REPO_SHARE, f"{repo} 占比 {share:.1%} 超过上限"


def test_excluded_repos_have_documented_reason() -> None:
    subset = load_subset()
    for repo in EXCLUDED_REPOS:
        assert repo in subset["excluded_repos"]
        assert subset["excluded_repos"][repo].strip()


def test_instance_ids_unique() -> None:
    subset = load_subset()
    ids: list[str] = []
    for case in subset["cases"]:
        ids.append(case["instance_id"])
    assert len(set(ids)) == len(ids)


def test_every_case_pins_base_commit() -> None:
    """没有 commit 就没有可复现性——行号会随版本漂移。"""
    subset = load_subset()
    for case in subset["cases"]:
        assert len(case["base_commit"]) == 40, case["instance_id"]


def test_every_case_has_gold_locations() -> None:
    subset = load_subset()
    for case in subset["cases"]:
        locations = case["gold_locations"]
        assert locations, f"{case['instance_id']} 没有解析出 gold 位置"
        for item in locations:
            assert item["path"], case["instance_id"]
            # 新建文件在 base_commit 里不存在，没有旧行号可定位，
            # 只能做文件级评分——这是合法状态，不是解析失败。
            if item.get("is_new_file"):
                assert item["line_ranges"] == [], case["instance_id"]
            else:
                assert item["line_ranges"], case["instance_id"]


def test_new_file_locations_are_flagged_not_dropped() -> None:
    """新建文件必须显式标记，而不是被静默丢弃。

    如果按普通区间打分，Agent 永远不可能「找到」一个在 base_commit 里
    根本不存在的文件的第 0 行。评分器必须能区别对待。
    """
    subset = load_subset()
    flagged = 0
    for case in subset["cases"]:
        for item in case["gold_locations"]:
            if item.get("is_new_file"):
                flagged += 1
                assert item["line_ranges"] == []
    # 当前子集中已知存在一处（astropy__astropy-13398）
    assert flagged >= 1


def test_gold_line_ranges_are_well_formed() -> None:
    """行号必须是 1-based 闭区间且 start <= end。"""
    subset = load_subset()
    checked = 0
    for case in subset["cases"]:
        for item in case["gold_locations"]:
            for start, end in item["line_ranges"]:
                assert start >= 1, case["instance_id"]
                assert start <= end, case["instance_id"]
                checked += 1
    assert checked >= EXPECTED_CASE_COUNT


def test_counts_match_cases() -> None:
    subset = load_subset()
    total_files = 0
    multi_file = 0
    for case in subset["cases"]:
        assert case["gold_file_count"] == len(case["gold_locations"])
        total_files += case["gold_file_count"]
        if case["gold_file_count"] > 1:
            multi_file += 1
    assert subset["total_gold_files"] == total_files
    assert subset["multi_file_case_count"] == multi_file


def test_docker_is_not_required_at_this_layer() -> None:
    """L2 的全部价值就在于不需要 Docker。"""
    subset = load_subset()
    scope = subset["evaluation_scope"]
    assert scope["docker_required"] is False
    assert scope["tests_executed"] is False
    assert "resolve rate" not in " ".join(scope["primary_metrics"])


def test_reporting_discipline_is_recorded() -> None:
    """防止把子集结果说成「在 SWE-bench 上达到 X%」。"""
    subset = load_subset()
    joined = " ".join(subset["reporting_discipline"])
    assert "SWE-bench 上达到" in joined
    assert "子集规模" in joined


def test_no_truncated_fields() -> None:
    """字段被服务端截断会让 gold patch 解析出错。"""
    subset = load_subset()
    assert subset["truncation_warnings"] == []
