"""Run deterministic retrieval evaluation on the 131-case fusion asset."""

from __future__ import annotations

import json
from pathlib import Path

from run_multilingual_retrieval_eval import build_comparison

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "multilingual_cases_fusion.json"
OUTPUT_PATH = (
    BACKEND_ROOT.parent / "outputs" / "evals" / "multilingual-fusion-retrieval-comparison.json"
)


def main() -> int:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    payload = build_comparison(document)
    payload["case_set"] = document["case_set"]
    payload["sources"] = document["sources"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
