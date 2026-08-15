"""Summarize Temporary-30 retrieval and structured Answer/Critic ablations."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .temporary_30_ablation import (
        ANSWER_GROUPS,
        MANIFEST_PATH,
        SEED,
        case_cluster_bootstrap,
        exact_mcnemar,
        load_manifest,
        majority,
        nearest_rank_percentile,
        run_directory,
        sha256_file,
        write_json_new,
    )
except ImportError:  # Direct script execution.
    from temporary_30_ablation import (
        ANSWER_GROUPS,
        MANIFEST_PATH,
        SEED,
        case_cluster_bootstrap,
        exact_mcnemar,
        load_manifest,
        majority,
        nearest_rank_percentile,
        run_directory,
        sha256_file,
        write_json_new,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _row_value(row: dict, metric: str) -> float:
    if row.get("error"):
        return 0.0
    value = row.get("metrics", {}).get(metric)
    if value is None:
        return 0.0
    return float(value)


def _case_values(rows: list[dict], group: str, metric: str) -> dict[str, list[float]]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["group"] == group:
            values[row["case_id"]].append(_row_value(row, metric))
    return dict(values)


def _paired_differences(
    rows: list[dict], left: str, right: str, metric: str
) -> dict[str, list[float]]:
    values = {}
    for row in rows:
        key = (row["case_id"], row["repeat"], row["group"])
        values[key] = _row_value(row, metric)
    differences: defaultdict[str, list[float]] = defaultdict(list)
    pairs = set()
    for row in rows:
        pairs.add((row["case_id"], row["repeat"]))
    pairs = sorted(pairs)
    for case_id, repeat in pairs:
        left_key = (case_id, repeat, left)
        right_key = (case_id, repeat, right)
        if left_key in values and right_key in values:
            differences[case_id].append(values[right_key] - values[left_key])
    return dict(differences)


def _case_binary(rows: list[dict], group: str, metric: str) -> dict[str, bool]:
    values = _case_values(rows, group, metric)
    result = {}
    for case_id, repeats in values.items():
        boolean_repeats = []
        for value in repeats:
            boolean_repeats.append(bool(value))
        result[case_id] = majority(tuple(boolean_repeats))
    return result


def _case_macro_metric(
    selected: list[dict], metric: str, *, omit_insufficient_recall: bool = False
) -> float | None:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for row in selected:
        if row.get("error"):
            if omit_insufficient_recall and row.get("category") == "insufficient_evidence":
                continue
            values[row["case_id"]].append(0.0)
            continue
        value = row.get("metrics", {}).get(metric)
        if value is not None:
            values[row["case_id"]].append(float(value))
    case_means = []
    for items in values.values():
        if items:
            case_means.append(sum(items) / len(items))
    return _mean(case_means)


def _effect(rows: list[dict], left: str, right: str, metric: str) -> dict[str, Any]:
    differences = _paired_differences(rows, left, right, metric)
    bootstrap = case_cluster_bootstrap(differences, seed=SEED)
    left_binary = _case_binary(rows, left, metric)
    right_binary = _case_binary(rows, right, metric)
    common = sorted(left_binary.keys() & right_binary.keys())
    left_values = []
    right_values = []
    for case_id in common:
        left_values.append(left_binary[case_id])
        right_values.append(right_binary[case_id])
    binary = exact_mcnemar(
        tuple(left_values),
        tuple(right_values),
    )
    return {
        "comparison": f"{right} - {left}",
        "cluster_bootstrap": bootstrap,
        "case_level_binary": binary,
        "case_repeat_differences": differences,
    }


def _group_summary(rows: list[dict], group: str) -> dict[str, Any]:
    selected = []
    for row in rows:
        if row["group"] == group:
            selected.append(row)
    elapsed = []
    for row in selected:
        if row.get("elapsed_milliseconds") is not None:
            elapsed.append(float(row["elapsed_milliseconds"]))
    automated = _case_binary(rows, group, "automated_grounded_pass")
    human_rows = []
    for row in selected:
        if row.get("metrics", {}).get("human_verified_complete_pass") is not None:
            human_rows.append(row)
    case_ids = set()
    for row in selected:
        case_ids.add(row["case_id"])
    human_values = []
    for row in human_rows:
        human_values.append(float(row["metrics"]["human_verified_complete_pass"]))
    automated_pass_count = 0
    for passed in automated.values():
        if passed:
            automated_pass_count += 1
    error_count = 0
    model_calls = 0
    input_tokens = 0
    output_tokens = 0
    for row in selected:
        if row.get("error"):
            error_count += 1
        model_calls += int(row.get("model_calls", 0))
        input_tokens += int(row.get("input_tokens", 0))
        output_tokens += int(row.get("output_tokens", 0))
    return {
        "case_run_count": len(selected),
        "case_count": len(case_ids),
        "automated_grounded_pass_case_rate": (
            automated_pass_count / len(automated) if automated else None
        ),
        "human_verified_complete_pass_case_rate": (
            _mean(human_values)
            if human_rows
            else None
        ),
        "human_review_status": "available" if human_rows else "not_human_reviewed",
        "outcome_accuracy": _case_macro_metric(selected, "outcome_accuracy"),
        "citation_precision": _case_macro_metric(selected, "citation_precision"),
        "evidence_recall": _case_macro_metric(
            selected, "evidence_recall", omit_insufficient_recall=True
        ),
        "errors": error_count,
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "mean_elapsed_milliseconds": _mean(elapsed),
        "p95_elapsed_milliseconds": nearest_rank_percentile(elapsed, 0.95),
        "max_elapsed_milliseconds": max(elapsed) if elapsed else None,
    }


def _rescue(rows: list[dict], baseline: str, critic: str) -> dict[str, Any]:
    lookup = {}
    for row in rows:
        key = (row["case_id"], row["repeat"], row["group"])
        lookup[key] = bool(_row_value(row, "automated_grounded_pass"))
    by_case: defaultdict[str, list[float]] = defaultdict(list)
    counts = {"fail_to_fail": 0, "rescue": 0, "regression": 0, "pass_to_pass": 0}
    pairs = set()
    for row in rows:
        pairs.add((row["case_id"], row["repeat"]))
    for case_id, repeat in sorted(pairs):
        before_key = (case_id, repeat, baseline)
        after_key = (case_id, repeat, critic)
        if before_key not in lookup or after_key not in lookup:
            continue
        before, after = lookup[before_key], lookup[after_key]
        if not before and not after:
            counts["fail_to_fail"] += 1
        elif not before and after:
            counts["rescue"] += 1
            by_case[case_id].append(1.0)
        elif before and not after:
            counts["regression"] += 1
            by_case[case_id].append(-1.0)
        else:
            counts["pass_to_pass"] += 1
            by_case[case_id].append(0.0)
        if not by_case[case_id]:
            by_case[case_id].append(0.0)
    counts["net_rescue"] = counts["rescue"] - counts["regression"]
    return {
        "comparison": f"{critic} - {baseline}",
        "automated": counts,
        "cluster_bootstrap_net_rescue_rate": case_cluster_bootstrap(by_case, seed=SEED),
        "human_verified": None,
        "human_review_status": "not_human_reviewed",
    }


def _interaction(rows: list[dict], metric: str) -> dict[str, Any]:
    sparse = _paired_differences(rows, "B00", "B01", metric)
    hybrid = _paired_differences(rows, "B10", "B11", metric)
    common = sorted(sparse.keys() & hybrid.keys())
    interaction = {}
    for case_id in common:
        if len(sparse[case_id]) != len(hybrid[case_id]):
            continue
        differences = []
        for left, right in zip(sparse[case_id], hybrid[case_id], strict=True):
            differences.append(right - left)
        interaction[case_id] = differences
    return case_cluster_bootstrap(interaction, seed=SEED)


def _breakdowns(rows: list[dict]) -> dict[str, Any]:
    breakdowns: dict[str, Any] = {}
    dimensions = {}
    for dimension in ("split", "language", "category"):
        values = set()
        for row in rows:
            values.add(row[dimension])
        dimensions[dimension] = sorted(values)
    for dimension, values in dimensions.items():
        dimension_breakdown = {}
        for value in values:
            value_rows = []
            for row in rows:
                if row[dimension] == value:
                    value_rows.append(row)
            group_summaries = {}
            for group in ANSWER_GROUPS:
                group_summaries[group] = _group_summary(value_rows, group)
            dimension_breakdown[value] = group_summaries
        breakdowns[dimension] = dimension_breakdown
    return breakdowns


def _snapshot_grounded_pass(row: dict, snapshot: dict, case: dict) -> bool:
    evidence_by_id = {}
    for group in row["evidence_groups"]:
        for item in group["evidence"]:
            evidence_by_id[item["evidence_id"]] = item
    outputs = {}
    for item in snapshot["subquestion_outputs"]:
        outputs[item["subquestion_id"]] = item
    for expected in case["subquestions"]:
        actual = outputs.get(expected["id"])
        if actual is None or actual["outcome"] != expected["expected_outcome"]:
            return False
        citations = []
        for item in actual["citations"]:
            citations.append(evidence_by_id[item])
        if expected["expected_outcome"] == "insufficient_evidence":
            if citations:
                return False
            continue
        for evidence in expected["expected_evidence"]:
            evidence_is_covered = False
            for citation in citations:
                if (
                    citation["path"] == evidence["path"]
                    and citation["start_line"] <= evidence["start_line"]
                    and citation["end_line"] >= evidence["end_line"]
                ):
                    evidence_is_covered = True
                    break
            if not evidence_is_covered:
                return False
    return True


def _revision_checkpoints(rows: list[dict], manifest: dict) -> dict[str, Any]:
    cases = {}
    for case in manifest["cases"]:
        cases[case["case_id"]] = case
    result = {}
    for group in ("B01", "B11"):
        selected = []
        for row in rows:
            if row["group"] == group and not row.get("error") and row.get("snapshots"):
                selected.append(row)
        checkpoints = {}
        previous_rate = None
        for checkpoint in ("0", "1", "3", "5"):
            passes = []
            for row in selected:
                passed = _snapshot_grounded_pass(
                    row, row["snapshots"][checkpoint], cases[row["case_id"]]
                )
                passes.append(passed)
            pass_count = 0
            for passed in passes:
                if passed:
                    pass_count += 1
            rate = pass_count / len(passes) if passes else None
            checkpoints[checkpoint] = {
                "case_run_count": len(passes),
                "automated_grounded_pass_rate": rate,
                "marginal_gain_from_previous": (
                    None if previous_rate is None or rate is None else rate - previous_rate
                ),
            }
            previous_rate = rate
        result[group] = {
            "checkpoints": checkpoints,
            "revision_count_distribution": {},
            "reached_max_revision_rate": None,
        }
        for count in range(0, 6):
            revision_count = 0
            for row in selected:
                if row.get("revise_calls", 0) == count:
                    revision_count += 1
            result[group]["revision_count_distribution"][str(count)] = revision_count
        max_revision_count = 0
        for row in selected:
            if row.get("revise_calls", 0) == 5:
                max_revision_count += 1
        result[group]["reached_max_revision_rate"] = (
            max_revision_count / len(selected) if selected else None
        )
    return result


def _blind_review_pack(rows: list[dict], manifest: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = {}
    for case in manifest["cases"]:
        cases[case["case_id"]] = case
    entries = []
    identities = []
    selected_stage = {"B00": "draft", "B01": "final", "B10": "draft", "B11": "final"}
    for row in rows:
        if row["split"] != "confirmation" or row.get("error"):
            continue
        stage = selected_stage[row["group"]]
        claim_rubric = []
        for subquestion in cases[row["case_id"]]["subquestions"]:
            claim_rubric.append(
                {
                    "subquestion_id": subquestion["id"],
                    "question": subquestion["question"],
                    "atomic_claims": subquestion["atomic_claims"],
                }
            )
        entries.append(
            {
                "identity_index": len(identities),
                "original_question": cases[row["case_id"]]["original_question"],
                "claim_rubric": claim_rubric,
                "answer": row[stage],
                "evidence_groups": row["evidence_groups"],
                "scores": None,
            }
        )
        identities.append(
            {
                "case_id": row["case_id"],
                "repeat": row["repeat"],
                "group": row["group"],
                "stage": stage,
            }
        )
    random.Random(SEED).shuffle(entries)
    mapping = []
    for index, entry in enumerate(entries, start=1):
        entry["review_id"] = f"R{index:04d}"
        identity = identities[entry.pop("identity_index")]
        mapping.append({"review_id": entry["review_id"], **identity})
    return (
        {
            "schema_version": 1,
            "instructions": {
                "claim_score": "0 incorrect/unsupported, 1 partial, 2 correct and complete",
                "unsupported_assertions": "Count factual assertions not supported by evidence.",
                "complete_pass": "All subquestions score 2 with zero unsupported assertions.",
            },
            "entries": entries,
        },
        {"schema_version": 1, "entries": mapping},
    )


def summarize_answer_rows(rows: list[dict], manifest: dict | None = None) -> dict[str, Any]:
    group_summaries = {}
    for group in ANSWER_GROUPS:
        group_summaries[group] = _group_summary(rows, group)
    payload = {
        "groups": group_summaries,
        "breakdowns": _breakdowns(rows),
        "effects": {
            "hybrid_without_critic": _effect(rows, "B00", "B10", "automated_grounded_pass"),
            "critic_without_semantic": _effect(rows, "B00", "B01", "automated_grounded_pass"),
            "critic_with_semantic": _effect(rows, "B10", "B11", "automated_grounded_pass"),
            "interaction": _interaction(rows, "automated_grounded_pass"),
        },
        "critic": {
            "sparse": _rescue(rows, "B00", "B01"),
            "hybrid": _rescue(rows, "B10", "B11"),
        },
    }
    if manifest is not None:
        payload["revision_checkpoints"] = _revision_checkpoints(rows, manifest)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-suffix")
    args = parser.parse_args()
    root = run_directory(args.run_id)
    all_rows = []
    retrieval = {}
    for split in ("diagnostic", "confirmation"):
        all_rows.extend(_read_jsonl(root / split / "structured-subquestion-outputs.jsonl"))
        retrieval_path = root / split / "retrieval-ablation.json"
        if retrieval_path.is_file():
            retrieval[split] = json.loads(retrieval_path.read_text(encoding="utf-8"))["summary"]
    if not all_rows:
        parser.error("no answer ablation outputs found for this run ID")
    manifest = load_manifest()
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "retrieval": retrieval,
        "answer": summarize_answer_rows(all_rows, manifest),
        "decisions": {
            "hybrid": "Keep but Unproven",
            "critic": "Keep Experimental",
            "status": "provisional_until_human_verified_complete_pass",
        },
    }
    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    summary_path = root / f"factorial-summary{suffix}.json"
    blind_path = root / f"blind-review-pack{suffix}.json"
    mapping_path = root / f"blind-review-mapping{suffix}.json"
    write_json_new(summary_path, summary)
    blind_pack, blind_mapping = _blind_review_pack(all_rows, manifest)
    write_json_new(blind_path, blind_pack)
    write_json_new(mapping_path, blind_mapping)
    print(
        json.dumps(
            {"output": str(summary_path), "decisions": summary["decisions"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
