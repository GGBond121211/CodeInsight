"""Q-006/Q-007：把「512 token 切块 / 100 token 重叠」的建议换算到本项目的行口径。

为什么需要这个脚本：
    公开资料常给「chunk size 512、overlap 100」这类建议，但本项目的切块器
    （ingestion/chunker.py）按逻辑行切分，登记表中的单位是 lines / ratio。
    单位不同就不能直接照抄，也不能凭感觉换算。

    本脚本用项目自己的 token 估算器（domain/tokens.py：utf8_bytes_div_4，
    登记表 S3-P03）测量真实仓库的每行 token 密度，再把 512 / 100 换算成行数
    和重叠比例，并实测候选参数与 token 上限的实际效果。

    它只做确定性测量，不调用模型、不写索引、不修改仓库。

用法::

    python experiments/chunk_parameter_probe.py --root backend/src
    python experiments/chunk_parameter_probe.py --root backend/src --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from codeinsight.domain.tokens import estimate_tokens  # noqa: E402
from codeinsight.ingestion.chunker import chunk_source_file  # noqa: E402
from codeinsight.ingestion.scanner import scan_repository  # noqa: E402

# 用户登记的推荐值（token 口径）。
RECOMMENDED_CHUNK_TOKENS = 512
RECOMMENDED_OVERLAP_TOKENS = 100

# 登记表里 chunk_max_lines / chunk_overlap_ratio 的候选值。
LINE_CANDIDATES = (40, 80, 120)
RATIO_CANDIDATES = (0.0, 0.10, 0.20)


@dataclass(frozen=True)
class ChunkProfile:
    """一组切分参数在一个真实仓库上的确定性测量结果。"""

    max_lines: int
    overlap_ratio: float
    max_tokens: int | None
    chunks: int
    mean_tokens: float
    median_tokens: float
    p90_tokens: float
    largest_chunk_tokens: int
    chunks_over_512: int


def measure_line_density(root: Path) -> dict[str, float | int]:
    """用项目 token 估算器统计每个逻辑行的 token 密度。"""
    scan = scan_repository(root)
    line_tokens: list[int] = []
    for source in scan.files:
        normalized = source.text.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized.split("\n"):
            line_tokens.append(estimate_tokens(line))
    # 空行不参与密度统计：它们不携带证据，只会拉低均值。
    informative = [value for value in line_tokens if value > 0]
    if not informative:
        raise ValueError(f"仓库没有可统计的文本行：{root}")
    total_tokens = sum(informative)
    return {
        "files": len(scan.files),
        "lines_total": len(line_tokens),
        "lines_informative": len(informative),
        "tokens_total": total_tokens,
        "tokens_per_line_mean": total_tokens / len(informative),
        "tokens_per_line_median": statistics.median(informative),
        "tokens_per_line_p90": _percentile(informative, 0.90),
        "tokens_per_line_max": max(informative),
    }


def profile(
    root: Path,
    max_lines: int,
    overlap_ratio: float,
    max_tokens: int | None = None,
) -> ChunkProfile:
    """用给定参数真实切分一遍，测量块数量与块 token 分布。"""
    scan = scan_repository(root)
    sizes: list[int] = []
    for source in scan.files:
        for chunk in chunk_source_file(
            source,
            max_lines=max_lines,
            overlap_ratio=overlap_ratio,
            max_tokens=max_tokens,
        ):
            sizes.append(estimate_tokens(chunk.text))
    if not sizes:
        raise ValueError(f"仓库没有产生任何块：{root}")
    return ChunkProfile(
        max_lines=max_lines,
        overlap_ratio=overlap_ratio,
        max_tokens=max_tokens,
        chunks=len(sizes),
        mean_tokens=sum(sizes) / len(sizes),
        median_tokens=statistics.median(sizes),
        p90_tokens=_percentile(sizes, 0.90),
        largest_chunk_tokens=max(sizes),
        chunks_over_512=sum(1 for value in sizes if value > RECOMMENDED_CHUNK_TOKENS),
    )


def translate(tokens_per_line: float) -> dict[str, object]:
    """把 token 口径的推荐值换算成本项目的行口径。"""
    chunk_lines = RECOMMENDED_CHUNK_TOKENS / tokens_per_line
    overlap_lines = RECOMMENDED_OVERLAP_TOKENS / tokens_per_line
    recommended_ratio = RECOMMENDED_OVERLAP_TOKENS / RECOMMENDED_CHUNK_TOKENS
    nearest_line = min(LINE_CANDIDATES, key=lambda value: abs(value - chunk_lines))
    return {
        "tokens_per_line": tokens_per_line,
        "recommended_chunk_lines": chunk_lines,
        "recommended_overlap_lines": overlap_lines,
        "recommended_overlap_ratio": recommended_ratio,
        "nearest_line_candidate": nearest_line,
        "nearest_line_candidate_gap_lines": abs(nearest_line - chunk_lines),
        "nearest_ratio_candidate": min(
            RATIO_CANDIDATES, key=lambda value: abs(value - recommended_ratio)
        ),
        "line_candidates_contain_recommendation": any(
            abs(value - chunk_lines) <= 0.5 for value in LINE_CANDIDATES
        ),
    }


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def build_report(root: Path) -> dict[str, object]:
    density = measure_line_density(root)
    profiles = [
        profile(root, max_lines, ratio)
        for max_lines in LINE_CANDIDATES
        for ratio in RATIO_CANDIDATES
    ]
    # token 上限候选：直接以 512 为硬上限，观察实际块分布。
    capped = [
        profile(root, 80, 0.0, RECOMMENDED_CHUNK_TOKENS),
        profile(root, 80, 0.20, RECOMMENDED_CHUNK_TOKENS),
    ]
    return {
        "root": str(root),
        "estimator": "utf8_bytes_div_4 (domain/tokens.estimate_tokens)",
        "recommendation": {
            "chunk_tokens": RECOMMENDED_CHUNK_TOKENS,
            "overlap_tokens": RECOMMENDED_OVERLAP_TOKENS,
        },
        "density": density,
        "translation": translate(float(density["tokens_per_line_mean"])),
        "profiles": [item.__dict__ for item in profiles],
        "token_capped_profiles": [item.__dict__ for item in capped],
    }


def render(report: dict[str, object]) -> str:
    density = report["density"]
    translation = report["translation"]
    lines = [
        f"仓库：{report['root']}",
        f"估算器：{report['estimator']}",
        "",
        "— 行密度 —",
        f"  文件 {density['files']}，行 {density['lines_total']}"
        f"（非空 {density['lines_informative']}）",
        f"  token 总量 {density['tokens_total']}",
        f"  每行 token：均值 {density['tokens_per_line_mean']:.2f}、"
        f"中位数 {density['tokens_per_line_median']:.1f}、"
        f"P90 {density['tokens_per_line_p90']:.1f}、"
        f"最大 {density['tokens_per_line_max']}",
        "",
        "— 512/100 换算 —",
        f"  512 token ≈ {translation['recommended_chunk_lines']:.1f} 行",
        f"  100 token ≈ {translation['recommended_overlap_lines']:.1f} 行",
        f"  重叠比例 = {translation['recommended_overlap_ratio']:.3f}",
        f"  最接近的行候选 {translation['nearest_line_candidate']}"
        f"（差 {translation['nearest_line_candidate_gap_lines']:.1f} 行），"
        f"比例候选 {translation['nearest_ratio_candidate']}",
        "",
        "— 行候选实测（无 token 上限）—",
        _table_header(),
    ]
    for item in report["profiles"]:
        lines.append(_table_row(item))
    lines.append("")
    lines.append("— 512 token 硬上限实测 —")
    lines.append(_table_header())
    for item in report["token_capped_profiles"]:
        lines.append(_table_row(item))
    return "\n".join(lines)


def _table_header() -> str:
    return (
        f"  {'行':>4} {'重叠':>6} {'上限':>6} {'块数':>6} {'均值':>8} "
        f"{'中位':>8} {'P90':>8} {'最大':>8} {'>512':>6}"
    )


def _table_row(item: dict[str, object]) -> str:
    cap = item["max_tokens"]
    cap_text = "-" if cap is None else str(cap)
    return (
        f"  {item['max_lines']:>4} {item['overlap_ratio']:>6.2f} {cap_text:>6} "
        f"{item['chunks']:>6} {item['mean_tokens']:>8.1f} "
        f"{item['median_tokens']:>8.1f} {item['p90_tokens']:>8.1f} "
        f"{item['largest_chunk_tokens']:>8} {item['chunks_over_512']:>6}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiments/chunk_parameter_probe.py",
        description="把 token 口径的切分建议换算成本项目的行口径，并实测候选参数。",
    )
    parser.add_argument("--root", default="backend/src", help="要测量的仓库根目录")
    parser.add_argument("--json", dest="json_path", default=None, help="把结果写入 JSON 文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = Path(arguments.root).resolve()
    if not root.is_dir():
        print(f"不是目录：{root}", file=sys.stderr)
        return 2
    report = build_report(root)
    print(render(report))
    if arguments.json_path:
        Path(arguments.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n已写入 {arguments.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
