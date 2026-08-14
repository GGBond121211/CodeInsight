"""Rescore stored Master-200 answers with chunk-compatible joint evidence coverage."""

from __future__ import annotations

import json
from types import SimpleNamespace

from codeinsight.domain.answer import AnswerCitation

try:
    from .run_master_200_eval import CASES_PATH, PROJECT_ROOT, _metrics, summarize
except ImportError:
    from run_master_200_eval import CASES_PATH, PROJECT_ROOT, _metrics, summarize


RUN_ROOT = PROJECT_ROOT / "outputs" / "evals" / "master-200" / "20260813T-master200-final-v2"


def _stored_result(payload: dict | None):
    if payload is None:
        return None
    citations = tuple(
        AnswerCitation(
            item["evidence_id"],
            item["relative_path"],
            item["start_line"],
            item["end_line"],
        )
        for item in payload["citations"]
    )
    return SimpleNamespace(
        outcome=payload["outcome"],
        answer=payload["answer"],
        citations=citations,
        subquestions=tuple(SimpleNamespace(**item) for item in payload.get("subquestions", ())),
    )


def main() -> int:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    repositories = {item["id"]: item for item in document["repositories"]}
    latest = {}
    for line in (RUN_ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["case_id"]] = row

    rescored = []
    for case in document["cases"]:
        row = dict(latest[case["id"]])
        root = PROJECT_ROOT / repositories[case["repository_id"]]["local_root"]
        row.update(_metrics(case, _stored_result(row.get("result")), root, row.get("error")))
        rescored.append(row)
    summary = summarize(rescored, list(repositories))
    payload = {
        "schema_version": 2,
        "metric_version": "chunk-compatible-joint-coverage-v2",
        "case_count": len(rescored),
        "summary": summary,
        "notes": [
            "Long evidence requirements are split at 80-line product chunk boundaries.",
            "Multiple citations may jointly cover one original evidence requirement.",
            "Citation precision accepts intersection with a required evidence region.",
            "Stored answers are rescored without new chat or embedding calls.",
        ],
    }
    output = RUN_ROOT / "summary-evidence-v2.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "micro": summary["micro"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
