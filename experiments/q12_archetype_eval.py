"""Q-012 C2.2：原型路由的命中率、错命中率、回退率，以及阈值扫描。

这是快路径的风险入口：命中意味着**跳过开放 Tool Loop**，直接按固定配方取证。
所以三个数要一起看：

  - 命中率（holdout 里判对原型的比例）：低说明快路径救不了多少问题；
  - 错命中率（负例被判进某个原型，或正例判成别的原型）：高说明会答非所问；
  - 回退率（route 返回 None）：这是安全兜底，高是对的，高到把正例也吞掉才是问题。

花费：只调用 Embedding（无对话模型）。同一段文本在同一进程里只算一次，
所以扫描多个阈值不会成倍花钱——示例问题向量与问题向量都命中缓存。

用法：
    backend/.venv/Scripts/python.exe experiments/q12_archetype_eval.py
    backend/.venv/Scripts/python.exe experiments/q12_archetype_eval.py --thresholds 0.5,0.62,0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from codeinsight.application.archetype_router import (  # noqa: E402
    ArchetypeRouter,
    ArchetypeRouterConfig,
)

GOLDEN = ROOT / "experiments" / "configs" / "q12_archetype_golden.yaml"
DEFAULT_THRESHOLDS = (0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75)
DEFAULT_MARGINS = (0.0, 0.02, 0.05)


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip()


class CachingEmbedder:
    """按文本缓存向量：阈值扫描要反复路由同一批问题，不缓存就是重复付费。"""

    def __init__(self, model) -> None:
        self._model = model
        self._cache: dict[str, tuple[float, ...]] = {}
        self.api_calls = 0
        self.api_texts = 0

    def __call__(self, texts):
        missing = [text for text in texts if text not in self._cache]
        if missing:
            batch = self._model.embed(missing)
            self.api_calls += 1
            self.api_texts += len(missing)
            for text, vector in zip(missing, batch.vectors):
                self._cache[text] = tuple(float(value) for value in vector)
        return SimpleNamespace(vectors=tuple(self._cache[text] for text in texts))


def _load_golden(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    holdout = [
        {"question": str(item["question"]), "expected": str(item["expected"])}
        for item in document["holdout"]
    ]
    negative = [str(item) for item in document["negative"]]
    return holdout, negative


def _evaluate(
    router: ArchetypeRouter,
    holdout: list[dict[str, str]],
    negative: list[str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for item in holdout:
        match = router.route(item["question"])
        predicted = match.archetype.name if match else None
        rows.append(
            {
                "question": item["question"],
                "expected": item["expected"],
                "predicted": predicted,
                "score": round(match.score, 4) if match else None,
                "margin": round(match.margin, 4) if match else None,
                "hit": predicted is not None,
                "correct": predicted == item["expected"],
            }
        )
    false_hits = 0
    for question in negative:
        match = router.route(question)
        rows.append(
            {
                "question": question,
                "expected": None,
                "predicted": match.archetype.name if match else None,
                "score": round(match.score, 4) if match else None,
                "margin": round(match.margin, 4) if match else None,
                "hit": match is not None,
                "correct": match is None,
            }
        )
        if match is not None:
            false_hits += 1
    positive_hits = [row for row in rows if row["expected"] is not None and row["hit"]]
    correct = [row for row in rows if row["expected"] is not None and row["correct"]]
    wrong_archetype = [row for row in positive_hits if not row["correct"]]
    return {
        "hit_rate": len(positive_hits) / len(holdout),
        "precision": len(correct) / len(positive_hits) if positive_hits else 0.0,
        "correct_rate": len(correct) / len(holdout),
        "wrong_archetype_rate": len(wrong_archetype) / len(holdout),
        "negative_false_hit_rate": false_hits / len(negative),
        "fallback_rate": 1 - (len(positive_hits) + false_hits) / (len(holdout) + len(negative)),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q-012 C2.2 原型路由评测")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "q12_archetype_eval.json")
    parser.add_argument("--thresholds", default=",".join(str(item) for item in DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--margins",
        default=",".join(str(item) for item in DEFAULT_MARGINS),
        help="Top-1 与第二名的差距下限；可扫多个值，与阈值两两组合",
    )
    args = parser.parse_args()

    _load_env()
    from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel

    holdout, negative = _load_golden(args.golden)
    embedder = CachingEmbedder(OpenAIEmbeddingModel.from_environment())
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    margins = [float(item) for item in args.margins.split(",") if item.strip()]

    by_threshold: dict[str, dict[str, object]] = {}
    for threshold in thresholds:
        for margin in margins:
            router = ArchetypeRouter(
                embed=embedder,
                config=ArchetypeRouterConfig(threshold=threshold, margin=margin),
            )
            report = _evaluate(router, holdout, negative)
            by_threshold[f"t{threshold:.2f}-m{margin:.2f}"] = report
            print(
                f"阈值 {threshold:.2f} margin {margin:.2f}：命中率 {report['hit_rate']:.3f} "
                f"命中里判对 {report['precision']:.3f} "
                f"判错原型 {report['wrong_archetype_rate']:.3f} "
                f"负例错命中 {report['negative_false_hit_rate']:.3f} "
                f"回退率 {report['fallback_rate']:.3f}"
            )

    default_key = "t0.62-m0.02" if "t0.62-m0.02" in by_threshold else next(iter(by_threshold))
    default = by_threshold[default_key]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in default["rows"]:
        expected = row["expected"] or "<negative>"
        confusion[str(expected)][str(row["predicted"])] += 1

    result = {
        "version": "q012-archetype-eval-v1",
        "golden": str(args.golden.relative_to(ROOT)).replace("\\", "/"),
        "library_version": ArchetypeRouter(
            embed=embedder, config=ArchetypeRouterConfig()
        ).library_version,
        "holdout_size": len(holdout),
        "negative_size": len(negative),
        "embedding_api_calls": embedder.api_calls,
        "embedding_api_texts": embedder.api_texts,
        "by_threshold": by_threshold,
        "default_key": default_key,
        "default_confusion": {key: dict(value) for key, value in confusion.items()},
        "default_rows": default["rows"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Embedding 调用：{embedder.api_calls} 次 / {embedder.api_texts} 段文本")
    print(f"默认配置（{default_key}）的混淆矩阵（行=期望，列=实际）：")
    for expected, counts in result["default_confusion"].items():
        pairs = "、".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        print(f"  {expected} -> {pairs}")
    print(f"已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
