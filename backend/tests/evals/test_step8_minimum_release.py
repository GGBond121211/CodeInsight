import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent


def test_step8_minimum_release_manifest_is_stratified_and_reproducible() -> None:
    manifest = json.loads(
        (EVAL_DIR / "step8_minimum_release_cases.json").read_text(encoding="utf-8")
    )
    master = json.loads(
        (EVAL_DIR / "master_200_cases.json").read_text(encoding="utf-8")
    )
    cases = {case["id"]: case for case in master["cases"]}
    case_ids = manifest["case_ids"]
    selected = [cases[case_id] for case_id in case_ids]

    assert manifest["case_count"] == 30
    assert len(case_ids) == 30
    assert len(set(case_ids)) == 30
    assert all(case_id in cases for case_id in case_ids)
    assert {case["category"] for case in selected} == {
        "ambiguity",
        "boundary_behavior",
        "code_document_conflict",
        "cross_file_trace",
        "data_flow",
        "insufficient_evidence",
        "multi_intent",
        "noisy_query",
        "semantic_paraphrase",
    }
    assert {case["repository_id"] for case in selected} == {
        "sample_repo",
        "httpx",
        "click",
        "requests",
    }
