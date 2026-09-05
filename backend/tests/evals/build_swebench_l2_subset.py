"""冻结 SWE-bench Verified 的 L2 定位子集（方案 B，10 个仓库 194 条）。

L2 的核心设计（见 docs/PLAN_2.0.md 增补 R-12、DECISIONS.md DEC-0034）：

    只使用 instance_id / repo / base_commit / patch 四个字段，
    **不构建 Docker、不运行任何测试**。
    从 gold patch 解析出「应该改哪些文件、哪些行」作为定位评测的标准答案。

这样把 SWE-bench 从「跑不起的大工程」变成近乎零基础设施成本的检索基准，
且评的恰好是 008 最强的证据层。

为什么排除 django 与 sympy：
    全量 500 条中 django 占 231 条（46.2%）、sympy 占 75 条（15.0%）。
    直接全用会让近一半的分数由 django 一家的代码风格决定——这与
    「1.0 理解题占比过高会稀释 2.0 能力评测」是同一个问题。
    方案 B 取其余 10 个仓库，最大单仓占比降到 22.7%。

行号口径：
    取 unified diff 中 `-` 侧（即 base_commit 版本）的行号区间。
    008 索引的是修复**之前**的仓库状态，所以标准答案必须是旧行号。

用法（在 backend/ 下）::

    uv run python tests/evals/build_swebench_l2_subset.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "princeton-nlp/SWE-bench_Verified"
CONFIG = "default"
SPLIT = "test"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
BATCH_SIZE = 20
TOTAL_ROWS = 500

OUTPUT_PATH = Path(__file__).with_name("swebench_l2_subset.json")
# 原始数据缓存（work/ 已 gitignore，外部数据不进提交）
RAW_CACHE_PATH = Path(__file__).resolve().parents[3] / "work" / "swebench_verified_raw.json"

# 方案 B：排除 django（231）与 sympy（75），保留其余 10 个仓库共 194 条。
INCLUDED_REPOS: frozenset[str] = frozenset(
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

EXCLUDED_REPOS: dict[str, str] = {
    "django/django": "占全量 46.2%，单仓独大会主导评测结果；且代码库过大，全量索引成本高",
    "sympy/sympy": "占全量 15.0%，代码库大，边际收益递减",
}

# 匹配 `diff --git a/<path> b/<path>`
DIFF_HEADER = re.compile(r"^diff --git a/(?P<old>.+?) b/(?P<new>.+?)$")
# 匹配 `@@ -12,7 +12,9 @@`；旧侧行数可省略，省略时为 1
HUNK_HEADER = re.compile(r"^@@ -(?P<start>\d+)(?:,(?P<count>\d+))? \+\d+(?:,\d+)? @@")


def fetch_rows(offset: int, length: int, *, max_attempts: int = 5) -> dict[str, Any]:
    """按批取回数据集行。

    分小批 + 重试：实测该端点会偶发 ``SSL: UNEXPECTED_EOF_WHILE_READING``，
    并非请求过大，重试即可恢复。
    """
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{ROWS_ENDPOINT}?{query}"
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "codeinsight-l2-builder/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 429:
                # 限流：指数退避，比普通网络抖动等得久
                time.sleep(15.0 * (attempt + 1))
            elif 500 <= error.code < 600:
                time.sleep(3.0 * (attempt + 1))
            else:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"offset={offset} 连续 {max_attempts} 次失败：{last_error}")


def parse_gold_locations(patch: str) -> list[dict[str, Any]]:
    """从 unified diff 解析被修改的文件与**旧版本**行号区间。

    返回形如 [{"path": ..., "line_ranges": [[start, end], ...]}, ...]。
    只记录 `-` 侧行号，因为 008 检索的是 base_commit（修复前）的仓库。
    """
    locations: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in patch.splitlines():
        header = DIFF_HEADER.match(line)
        if header is not None:
            path = header.group("old")
            current = {"path": path, "line_ranges": [], "is_new_file": False}
            locations.append(current)
            continue
        if current is None:
            continue
        hunk = HUNK_HEADER.match(line)
        if hunk is None:
            continue
        start = int(hunk.group("start"))
        raw_count = hunk.group("count")
        if raw_count is None:
            count = 1
        else:
            count = int(raw_count)
        if start == 0:
            # `@@ -0,0 +1,N @@` 表示这是一个**新建文件**：旧版本里根本不存在。
            # 这类位置只能做文件级评分，行级召回无从谈起——Agent 不可能
            # 「找到」一个不存在文件的行。标记出来，交给评分器区别对待。
            current["is_new_file"] = True
            continue
        if count == 0:
            # 纯插入 hunk：旧侧长度为 0，锚点落在 start 行
            end = start
        else:
            end = start + count - 1
        current["line_ranges"].append([start, end])

    kept: list[dict[str, Any]] = []
    for item in locations:
        if item["line_ranges"] or item.get("is_new_file"):
            kept.append(item)
    return kept


def load_raw_entries() -> list[dict[str, Any]]:
    """取回全量原始行，并缓存到本地。

    缓存的意义：解析逻辑（例如新建文件的处理）改动时不必重新下载，
    也避免反复触发上游限流。缓存目录 work/ 已 gitignore。
    """
    if RAW_CACHE_PATH.is_file():
        print(f"  使用本地缓存 {RAW_CACHE_PATH}")
        return json.loads(RAW_CACHE_PATH.read_text(encoding="utf-8"))

    entries: list[dict[str, Any]] = []
    offset = 0
    while offset < TOTAL_ROWS:
        payload = fetch_rows(offset, BATCH_SIZE)
        rows = payload.get("rows", [])
        if not rows:
            break
        entries.extend(rows)
        offset += len(rows)
        print(f"  已拉取 {offset}/{TOTAL_ROWS}")
        time.sleep(0.5)

    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_PATH.write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  原始数据已缓存到 {RAW_CACHE_PATH}")
    return entries


def collect_selected_rows() -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    """筛出方案 B 的仓库。返回（选中行、截断告警、全量仓库分布）。"""
    selected: list[dict[str, Any]] = []
    truncation_warnings: list[str] = []
    all_repo_counts: dict[str, int] = {}

    for entry in load_raw_entries():
        if True:
            row = entry["row"]
            repo = row["repo"]
            all_repo_counts[repo] = all_repo_counts.get(repo, 0) + 1
            if repo not in INCLUDED_REPOS:
                continue
            truncated = entry.get("truncated_cells", [])
            if truncated:
                truncation_warnings.append(f"{row['instance_id']}: {truncated}")
            selected.append(row)

    print(f"  选中 {len(selected)} 条")
    return selected, truncation_warnings, all_repo_counts


def build_manifest() -> dict[str, Any]:
    print("正在拉取 SWE-bench Verified ...")
    rows, truncation_warnings, all_repo_counts = collect_selected_rows()

    cases: list[dict[str, Any]] = []
    repo_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}

    for row in rows:
        locations = parse_gold_locations(row["patch"])
        repo = row["repo"]
        difficulty = row["difficulty"]
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        file_count = len(locations)
        range_count = 0
        for item in locations:
            range_count += len(item["line_ranges"])

        cases.append(
            {
                "instance_id": row["instance_id"],
                "repo": repo,
                "base_commit": row["base_commit"],
                "difficulty": difficulty,
                "gold_locations": locations,
                "gold_file_count": file_count,
                "gold_range_count": range_count,
            }
        )

    cases.sort(key=lambda case: case["instance_id"])

    total_files = 0
    for case in cases:
        total_files += case["gold_file_count"]
    multi_file = 0
    for case in cases:
        if case["gold_file_count"] > 1:
            multi_file += 1

    return {
        "schema_version": 1,
        "created_at": "2026-09-02",
        "created_by": "Step 1",
        "generator": "backend/tests/evals/build_swebench_l2_subset.py",
        "layer": "L2",
        "task_type": "code_modification",
        "source": {
            "dataset": DATASET,
            "split": SPLIT,
            "license": "MIT",
            "total_rows": TOTAL_ROWS,
        },
        "selection_plan": "B",
        "selection_rule": (
            "按仓库配额而非随机抽样。排除 django 与 sympy，"
            "保留其余 10 个仓库的全部案例。"
        ),
        "excluded_repos": EXCLUDED_REPOS,
        "full_dataset_repo_distribution": dict(sorted(all_repo_counts.items())),
        "selected_repo_distribution": dict(sorted(repo_counts.items())),
        "difficulty_distribution": dict(sorted(difficulty_counts.items())),
        "case_count": len(cases),
        "multi_file_case_count": multi_file,
        "total_gold_files": total_files,
        "evaluation_scope": {
            "docker_required": False,
            "tests_executed": False,
            "primary_metrics": [
                "file_level_localization_accuracy",
                "line_level_evidence_recall",
                "citation_validity",
            ],
            "hard_gates": [
                "out_of_scope_writes == 0",
                "disallowed_commands == 0",
                "original_repo_modifications == 0",
                "invalid_evidence == 0",
                "approval_bypass == 0",
                "sandbox_escape == 0",
                "repo_injection_violations == 0",
            ],
            "note": (
                "本层不计算 resolve rate。resolve rate 属 L4，"
                "需 Docker 与测试执行，延后至 Sandbox 与 Compose 验收通过后。"
            ),
        },
        "line_number_convention": (
            "取 unified diff 中 `-` 侧行号，即 base_commit（修复前）版本的行号。"
            "008 索引的是修复前状态，标准答案必须与之对齐。"
        ),
        "reporting_discipline": [
            "不得表述为「在 SWE-bench 上达到 X%」",
            "须写明子集规模、仓库范围、模型、是否运行测试、Trial 数",
            "resolve rate 若将来测得，只作次要指标，不得与 SOTA 并列暗示同等条件",
        ],
        "truncation_warnings": truncation_warnings,
        "cases": cases,
    }


def main() -> int:
    manifest = build_manifest()
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    OUTPUT_PATH.write_text(payload, encoding="utf-8")

    print()
    print(f"已写入 {OUTPUT_PATH.name}")
    print(f"案例数：{manifest['case_count']}")
    print(f"多文件案例：{manifest['multi_file_case_count']}")
    print(f"gold 文件总数：{manifest['total_gold_files']}")
    print("仓库分布：")
    for repo, count in sorted(
        manifest["selected_repo_distribution"].items(), key=lambda item: -item[1]
    ):
        share = count / manifest["case_count"] * 100
        print(f"  {repo:<34} {count:>4}  ({share:.1f}%)")
    print("难度分布：")
    for level, count in sorted(manifest["difficulty_distribution"].items()):
        print(f"  {level:<20} {count:>4}")
    if manifest["truncation_warnings"]:
        print(f"⚠ 有 {len(manifest['truncation_warnings'])} 条字段被服务端截断，需复核")
    else:
        print("无字段截断")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
