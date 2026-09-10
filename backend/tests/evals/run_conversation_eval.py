"""用真实 Provider 运行 2.1.0 多轮 pilot。

这个 runner 明确拒绝 Fake Provider，只保存脱敏的运行指标，不保存用户问题、模型答案、
供应商原始响应或 reasoning。change 轮只走到等待审批，不自动 approve/apply。

示例（从 backend 目录运行）::

    $env:CODEINSIGHT_MAX_OUTPUT_TOKENS = "40960"
    python tests/evals/run_conversation_eval.py --split dev --split regression \\
        --case-id conv-020-four-round-lane-recheck
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from codeinsight.infrastructure.chat_endpoint import EXPECTED_CHAT_BASE_URL  # noqa: E402

DATASET_PATH = BACKEND_ROOT / "tests" / "evals" / "conversation_cases_2_1_0.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evals" / "conversation-real-pilot.json"
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "WAITING_APPROVAL"}
def _load_project_env() -> None:
    """为本地 pilot 读取项目根 .env；不打印、不持久化任何密钥。"""
    candidates = (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent.parent / ".env",
    )
    allowed = {"CODEINSIGHT_API_KEY", "CODEINSIGHT_MODEL", "CODEINSIGHT_BASE_URL"}
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key in allowed and value:
                os.environ[key] = value
        if os.environ.get("CODEINSIGHT_BASE_URL"):
            return


def _require_project_gateway() -> None:
    configured = os.environ.get("CODEINSIGHT_BASE_URL", "").rstrip("/")
    if configured != EXPECTED_CHAT_BASE_URL:
        raise SystemExit(
            "拒绝运行：conversation pilot 必须使用项目 Frontier 网关 "
            f"{EXPECTED_CHAT_BASE_URL}，当前为 {configured or '<missing>'}"
        )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _development_policy_snapshot() -> dict[str, object]:
    environment = os.environ.get("CODEINSIGHT_ENV", "development").strip().lower()
    enabled = _env_flag("CODEINSIGHT_DEV_MODE") and environment not in {
        "prod",
        "production",
        "release",
    }
    return {
        "environment": environment,
        "enabled": enabled,
        "auto_approve_changes": enabled and _env_flag("CODEINSIGHT_DEV_AUTO_APPROVE"),
        "skip_sandbox_validation": enabled and _env_flag("CODEINSIGHT_DEV_SKIP_SANDBOX_VALIDATION"),
    }


def load_cases(case_ids: set[str] | None, splits: set[str]) -> tuple[dict, list[dict]]:
    document = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = [
        case
        for case in document["cases"]
        if case["split"] in splits and (not case_ids or case["id"] in case_ids)
    ]
    if case_ids:
        found = {case["id"] for case in cases}
        missing = sorted(case_ids - found)
        if missing:
            raise ValueError(f"请求的 case 不在 pilot split 中：{missing}")
    if not cases:
        raise ValueError("没有可运行的 pilot case")
    return document, cases


def _citation_covers(citation: dict, evidence: dict) -> bool:
    return (
        citation.get("relative_path") == evidence.get("path")
        and isinstance(citation.get("start_line"), int)
        and isinstance(citation.get("end_line"), int)
        and citation["start_line"] <= evidence["start_line"]
        and citation["end_line"] >= evidence["end_line"]
    )


def _citation_is_valid(citation: dict) -> bool:
    relative = citation.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ".." in Path(relative).parts
    ):
        return False
    target = FIXTURE_ROOT / relative
    if not target.is_file():
        return False
    start = citation.get("start_line")
    end = citation.get("end_line")
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and start >= 1
        and start <= end
        and end <= len(target.read_text(encoding="utf-8").splitlines())
    )


def _evidence_metrics(expected: dict, result: dict | None) -> dict[str, object]:
    evidence = expected.get("evidence") or []
    citations = result.get("citations", []) if isinstance(result, dict) else []
    valid = sum(
        1
        for citation in citations
        if isinstance(citation, dict) and _citation_is_valid(citation)
    )
    if expected.get("outcome") == "approval_required":
        # 修改预览不输出代码回答 citations；其证据由工具探索、patch 范围和审批状态衡量。
        return {
            "evidence_applicable": False,
            "evidence_count": len(evidence),
            "citation_count": len(citations),
            "valid_citation_count": valid,
            "evidence_recall": None,
            "valid_citation_rate": None,
        }
    covered = sum(
        1
        for item in evidence
        if any(
            isinstance(citation, dict) and _citation_covers(citation, item)
            for citation in citations
        )
    )
    return {
        "evidence_applicable": True,
        "evidence_count": len(evidence),
        "citation_count": len(citations),
        "valid_citation_count": valid,
        "evidence_recall": covered / len(evidence) if evidence else 1.0,
        "valid_citation_rate": valid / len(citations) if citations else 1.0,
    }


def _usage(result: dict | None) -> dict[str, object]:
    if not isinstance(result, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_miss_tokens": 0,
            "cache_hit_ratio": 0.0,
            "usage_source": "none",
        }
    value = result.get("observability")
    if not isinstance(value, dict):
        value = {}
    return {
        "input_tokens": int(value.get("input_tokens", 0) or 0),
        "output_tokens": int(value.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(value.get("cache_read_tokens", 0) or 0),
        "cache_miss_tokens": int(value.get("cache_miss_tokens", 0) or 0),
        "cache_hit_ratio": float(value.get("cache_hit_ratio", 0.0) or 0.0),
        "usage_source": str(value.get("usage_source", "unknown")),
    }


def _event_metrics(client, run_id: str) -> dict[str, object]:
    service = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    events = service.runtime.event_log.read_events(run_id)
    model_called = [event for event in events if event.event_type == "model_called"]
    model_results = [event for event in events if event.event_type == "model_result"]
    context_events = [event for event in events if event.event_type == "context_assembled"]
    preflight_events = [event for event in events if event.event_type == "validation_preflight"]
    retrieval_events = [event for event in events if event.event_type == "retrieval_finished"]
    intent_events = [event for event in events if event.event_type == "intent_classified"]
    return {
        "model_call_count": len(model_called),
        "gateway_attempt_ids": [
            str(event.payload.get("attempt_id")) for event in model_called
        ],
        "gateway_models": sorted(
            {
                str(event.payload.get("model"))
                for event in model_called
                if event.payload.get("model")
            }
        ),
        "model_error_classes": sorted(
            {
                str(event.payload.get("error_class"))
                for event in model_results
                if event.payload.get("outcome") == "error"
            }
        ),
        "model_error_details": sorted(
            {
                str(event.payload.get("error_detail"))
                for event in model_results
                if event.payload.get("outcome") == "error" and event.payload.get("error_detail")
            }
        ),
        "model_finish_reasons": sorted(
            {
                str(event.payload.get("finish_reason"))
                for event in model_results
                if event.payload.get("finish_reason")
            }
        ),
        "event_types": sorted({event.event_type for event in events}),
        "execution_routes": [
            str(event.payload.get("execution_route"))
            for event in intent_events
            if event.payload.get("execution_route")
        ],
        "retrieval_outcomes": [
            str(event.payload.get("outcome"))
            for event in retrieval_events
            if event.payload.get("outcome")
        ],
        "retrieval_citation_counts": [
            int(event.payload.get("citations", "0"))
            for event in retrieval_events
            if str(event.payload.get("citations", "")).isdigit()
        ],
        "validation_error_classes": sorted(
            {
                str(event.payload.get("error_class"))
                for event in preflight_events
                if event.payload.get("error_class")
            }
        ),
        "validation_available": [
            str(event.payload.get("available"))
            for event in preflight_events
            if event.payload.get("available") is not None
        ],
        "context_history_turns": [
            int(event.payload.get("history_turns", "0")) for event in context_events
        ],
        "context_compacted": sum(1 for event in events if event.event_type == "context_compacted"),
        "intent_goal_actions": [
            str(event.payload.get("goal_action"))
            for event in events
            if event.event_type == "intent_classified" and event.payload.get("goal_action")
        ],
    }


def _wait_for_terminal(client, turn_id: str, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/v2/chat/turns/{turn_id}")
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") in TERMINAL_STATUSES:
            return payload
        time.sleep(0.05)
    raise TimeoutError("chat turn did not reach a terminal status")


def _run_case(
    client,
    case: dict,
    *,
    timeout_seconds: float,
    total_tokens: int,
    token_limit: int,
) -> tuple[dict, int]:
    session_response = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    )
    session_response.raise_for_status()
    session_id = session_response.json()["session_id"]
    turn_records: list[dict[str, object]] = []
    case_tokens = total_tokens
    stopped_reason: str | None = None

    for turn_spec in case["turns"]:
        expected = turn_spec["expected"]
        if case_tokens >= token_limit:
            stopped_reason = "token_limit_reached"
            break
        started = perf_counter()
        try:
            accepted = client.post(
                "/api/v2/chat/turns",
                json={
                    "session_id": session_id,
                    "repository_root": str(FIXTURE_ROOT),
                    "message": turn_spec["user_message"],
                    "client_turn_id": f"pilot-{case['id']}-{turn_spec['turn_id']}",
                    "limit": 5,
                    "show_debug_reasoning": False,
                },
            )
            accepted.raise_for_status()
            accepted_payload = accepted.json()
            actual = _wait_for_terminal(client, accepted_payload["turn_id"], timeout_seconds)
            result = actual.get("result")
            usage = _usage(result if isinstance(result, dict) else None)
            usage_total = int(usage["input_tokens"]) + int(usage["output_tokens"])
            case_tokens += usage_total
            events = _event_metrics(client, actual["run_id"])
            actual_goal = (
                events["intent_goal_actions"][-1]
                if events["intent_goal_actions"]
                else None
            )
            actual_task = actual.get("task_type")
            actual_status = actual.get("status")
            actual_outcome = result.get("outcome") if isinstance(result, dict) else None
            actual_error = actual.get("error")
            failure_class = None
            if actual_status == "FAILED":
                if events["validation_error_classes"]:
                    failure_class = events["validation_error_classes"][0]
                elif events["model_error_details"]:
                    failure_class = events["model_error_details"][0]
                elif events["model_call_count"]:
                    failure_class = "runtime_failed_after_model_call"
                else:
                    failure_class = "runtime_failed_before_model_call"
            turn_records.append(
                {
                    "turn_id": turn_spec["turn_id"],
                    "expected_task_type": expected["task_type"],
                    "actual_task_type": actual_task,
                    "task_type_match": actual_task == expected["task_type"],
                    "expected_goal_action": expected["goal_action"],
                    "actual_goal_action": actual_goal,
                    "goal_action_match": actual_goal == expected["goal_action"],
                    "expected_outcome": expected["outcome"],
                    "actual_outcome": actual_outcome,
                    "status": actual_status,
                    "failure_class": failure_class,
                    "failure_detail": (
                        str(actual_error)[:240]
                        if actual_status == "FAILED" and actual_error
                        else None
                    ),
                    "outcome_match": (
                        actual_status == "WAITING_APPROVAL"
                        if expected["outcome"] == "approval_required"
                        else actual_outcome == expected["outcome"]
                    ),
                    "model_called": events["model_call_count"] > 0,
                    "model": result.get("model") if isinstance(result, dict) else None,
                    "provider_host": urlsplit(os.environ.get("CODEINSIGHT_BASE_URL", "")).hostname,
                    "latency_milliseconds": round((perf_counter() - started) * 1000, 2),
                    "usage": usage,
                    "evidence": _evidence_metrics(
                        expected, result if isinstance(result, dict) else None
                    ),
                    "runtime": events,
                }
            )
            if actual_status == "WAITING_APPROVAL":
                # 不自动审批、不应用变更；数据集把 change 轮设计为链尾。
                break
        except Exception as error:  # noqa: BLE001 - pilot records safe failure class only
            turn_records.append(
                {
                    "turn_id": turn_spec["turn_id"],
                    "expected_task_type": expected["task_type"],
                    "expected_outcome": expected["outcome"],
                    "status": "PILOT_ERROR",
                    "failure_class": type(error).__name__,
                    "failure_detail": type(error).__name__,
                    "latency_milliseconds": round((perf_counter() - started) * 1000, 2),
                    "model_called": False,
                }
            )
            stopped_reason = type(error).__name__
            break
    return {
        "case_id": case["id"],
        "split": case["split"],
        "turn_count_requested": len(case["turns"]),
        "turn_count_executed": len(turn_records),
        "stopped_reason": stopped_reason,
        "turns": turn_records,
    }, case_tokens


def _summarize(case_records: list[dict]) -> dict[str, object]:
    turns = [turn for case in case_records for turn in case["turns"]]
    completed = [turn for turn in turns if turn.get("status") in {"COMPLETED", "WAITING_APPROVAL"}]
    failed = [turn for turn in turns if turn.get("status") not in {"COMPLETED", "WAITING_APPROVAL"}]
    route_hits = [turn for turn in completed if turn.get("task_type_match")]
    goal_hits = [turn for turn in completed if turn.get("goal_action_match")]
    outcome_hits = [turn for turn in completed if turn.get("outcome_match")]
    evidence = [
        turn["evidence"]
        for turn in completed
        if isinstance(turn.get("evidence"), dict)
        and turn["evidence"].get("evidence_applicable", True)
        and isinstance(turn["evidence"].get("evidence_recall"), (int, float))
    ]
    input_tokens = sum(int(turn.get("usage", {}).get("input_tokens", 0)) for turn in turns)
    output_tokens = sum(int(turn.get("usage", {}).get("output_tokens", 0)) for turn in turns)
    cache_read = sum(int(turn.get("usage", {}).get("cache_read_tokens", 0)) for turn in turns)
    return {
        "case_count": len(case_records),
        "turns_executed": len(turns),
        "terminal_turns": len(completed),
        "failed_turns": len(failed),
        "failure_rate": len(failed) / len(turns) if turns else 0.0,
        "route_accuracy": len(route_hits) / len(completed) if completed else 0.0,
        "goal_continuity_accuracy": len(goal_hits) / len(completed) if completed else 0.0,
        "outcome_accuracy": len(outcome_hits) / len(completed) if completed else 0.0,
        "evidence_recall_mean": (
            sum(float(item["evidence_recall"]) for item in evidence) / len(evidence)
            if evidence
            else 0.0
        ),
        "valid_citation_rate_mean": (
            sum(float(item["valid_citation_rate"]) for item in evidence) / len(evidence)
            if evidence
            else 0.0
        ),
        "evidence_applicable_turns": len(evidence),
        "model_calls": sum(
            int(turn.get("runtime", {}).get("model_call_count", 0)) for turn in turns
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_hit_ratio": cache_read / input_tokens if input_tokens else 0.0,
        "compaction_events": sum(
            int(turn.get("runtime", {}).get("context_compacted", 0)) for turn in turns
        ),
        "failure_classes": dict(
            Counter(str(turn["failure_class"]) for turn in turns if turn.get("failure_class"))
        ),
        "failed_cases": sorted({case["case_id"] for case in case_records if any(
            turn.get("status") not in {"COMPLETED", "WAITING_APPROVAL"}
            for turn in case["turns"]
        )}),
        "stopped_cases": [case["case_id"] for case in case_records if case.get("stopped_reason")],
        "semantic_fact_recall": "not_automated_source_review_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        choices=["dev", "regression", "golden", "holdout"],
    )
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=40960)
    parser.add_argument("--max-total-tokens", type=int, default=9_000_000)
    args = parser.parse_args()

    _load_project_env()
    _require_project_gateway()
    if os.environ.get("CODEINSIGHT_GATEWAY_FAKE_MODE") == "1":
        raise SystemExit("拒绝运行：CODEINSIGHT_GATEWAY_FAKE_MODE=1，pilot 必须使用真实 Provider")
    if not os.environ.get("CODEINSIGHT_API_KEY", "").strip():
        raise SystemExit("拒绝运行：未配置 CODEINSIGHT_API_KEY")
    if args.max_output_tokens < 256:
        raise SystemExit("--max-output-tokens 不能低于 256")
    os.environ["CODEINSIGHT_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)

    from fastapi.testclient import TestClient  # noqa: I001
    from codeinsight.api.app import create_app
    from codeinsight.infrastructure.model_gateway import default_gateway_from_environment

    selected_splits = set(args.split or ["dev", "regression", "golden"])
    document, cases = load_cases(set(args.case_ids) if args.case_ids else None, selected_splits)
    dataset_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    records: list[dict] = []
    total_tokens = 0
    for case in cases:
        # 每个 case 是一个独立实验单元：同一 case 内保持一个 Session，
        # case 之间不共享 tenant budget、Token Bucket 或语义缓存，避免前一条
        # 多轮会话把后一条的质量结果污染成“预算耗尽”。
        default_gateway_from_environment.cache_clear()
        with TestClient(create_app()) as client:
            record, total_tokens = _run_case(
                client,
                case,
                timeout_seconds=args.timeout_seconds,
                total_tokens=total_tokens,
                token_limit=args.max_total_tokens,
            )
        records.append(record)
        if total_tokens >= args.max_total_tokens:
            break

    payload = {
        "schema_version": 1,
        "dataset_id": document["dataset_id"],
        "dataset_version": document["dataset_version"],
        "dataset_sha256": dataset_hash,
        "source_commit": document["source_commit"],
        "provider_host": urlsplit(os.environ.get("CODEINSIGHT_BASE_URL", "")).hostname,
        "real_provider_only": True,
        "fake_provider_used": False,
        "model_configured": os.environ.get("CODEINSIGHT_MODEL"),
        "development_policy": _development_policy_snapshot(),
        "output_token_cap": args.max_output_tokens,
        "max_retries": 0,
        "splits": sorted(selected_splits),
        "summary": _summarize(records),
        "cases": records,
        "notes": [
            "只保存路由、证据、上下文、usage、cache、latency 和安全边界元数据。",
            "不保存 user_message、assistant_message、provider 原始响应或 reasoning_content。",
            "change 轮停在 WAITING_APPROVAL，不自动 approve/apply。",
            "required_fact_recall 仍需基于源码审查，不由本 runner 伪造 LLM judge 分数。",
            (
                "每个 case 使用独立应用实例；case 内 Session 连续，"
                "case 间不共享预算、Token Bucket、语义缓存或 Circuit Breaker。"
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": payload["summary"], "output": str(args.output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
