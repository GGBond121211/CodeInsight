"""2.1.0 数据集准入检查。

这个模块只做确定性准入，不调用模型，也不把模型输出写回数据集。它把“文件能
解析”“Evidence 行号存在”“数据没有被静默替换”“多轮样本没有跨 split 泄漏”
和“Agent 源码审查已经覆盖”分开报告，避免一个漂亮的自动分数掩盖数据问题。

用法::

    python experiments/dataset_quality.py validate \
        --policy experiments/configs/dataset_quality_policy_2_1_0.yaml \
        --baseline experiments/configs/baseline_2_1_0.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - the repository environment supplies PyYAML
    yaml = None  # type: ignore[assignment]

CONVERSATION_TASK_TYPES = frozenset(
    {"explain", "change", "general_chat", "scope_redirect", "clarify"}
)
CONVERSATION_GOAL_ACTIONS = frozenset({"create", "continue", "replace", "preserve"})
CONVERSATION_OUTCOMES = frozenset(
    {
        "answered",
        "insufficient_evidence",
        "clarification_required",
        "redirected",
        "approval_required",
    }
)
KNOWN_SPLIT_FIELDS = ("split", "evaluation_split")
FORBIDDEN_MODEL_CONTENT_KEYS = frozenset(
    {"assistant_answer", "raw_response", "provider_response", "reasoning", "hidden_reasoning"}
)


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    fatal: bool = True


@dataclass
class DatasetCheck:
    dataset_id: str
    path: str
    declared_count: int | None = None
    actual_count: int | None = None
    actual_sha256: str | None = None
    declared_turn_count: int | None = None
    actual_turn_count: int | None = None
    state: str = "DATASET_BLOCKED"
    issues: list[QualityIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return not any(item.fatal for item in self.issues)

    def error(self, code: str, message: str) -> None:
        self.issues.append(QualityIssue(code, message, True))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def finalize(self, configured_state: str) -> None:
        self.state = configured_state if self.accepted else "DATASET_BLOCKED"


@dataclass
class QualityReport:
    policy_path: str
    baseline_path: str | None
    checks: list[DatasetCheck] = field(default_factory=list)
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return not any(issue.fatal for issue in self.issues) and all(
            check.accepted for check in self.checks
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "policy_path": self.policy_path,
            "baseline_path": self.baseline_path,
            "issues": [asdict(issue) for issue in self.issues],
            "metrics": self.metrics,
            "datasets": [
                {
                    **asdict(check),
                    "issues": [asdict(issue) for issue in check.issues],
                }
                for check in self.checks
            ],
        }


def load_mapping(path: Path) -> dict[str, Any]:
    """加载 JSON 或 YAML 映射。配置解析失败必须显式中止。"""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        if yaml is None:
            raise ValueError(f"{path} 需要 PyYAML 才能解析")
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须解析为对象")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _safe_relative_path(raw: object) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    path = Path(raw)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in raw


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _normalize_message(message: str) -> str:
    # 只用于发现跨 case 的近似重复，不用于改写用户输入或生成答案。
    return re.sub(r"\s+", " ", message.casefold()).strip()


def _items_from_document(document: dict[str, Any], count_field: str) -> list[Any] | None:
    value = document.get(count_field)
    return value if isinstance(value, list) else None


def _item_id(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in ("id", "case_id", "task_id", "instance_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _validate_generic_dataset(
    check: DatasetCheck,
    document: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    if document.get("schema_version") not in {1, 2}:
        check.error("schema_version", "schema_version 必须是当前支持的 1 或 2")
    count_field = entry.get("count_field")
    if not isinstance(count_field, str):
        check.error("count_field", "策略缺少 count_field")
        return
    items = _items_from_document(document, count_field)
    if items is None:
        check.error("items", f"文档字段 {count_field} 必须是列表")
        return
    check.actual_count = len(items)
    expected_count = entry.get("expected_count")
    if isinstance(expected_count, int):
        check.declared_count = expected_count
        if len(items) != expected_count:
            check.error(
                "count_mismatch",
                f"声明数量 {expected_count} 与实际数量 {len(items)} 不一致",
            )
    ids = [_item_id(item) for item in items]
    if any(item_id is None for item_id in ids):
        check.error("missing_id", "至少有一条数据缺少非空 ID")
    present_ids = [item_id for item_id in ids if item_id is not None]
    if len(present_ids) != len(set(present_ids)):
        check.error("duplicate_id", "数据集内存在重复 ID")
    allowed_splits = entry.get("allowed_splits")
    if isinstance(allowed_splits, list):
        allowed = {str(item) for item in allowed_splits}
        for item in items:
            if not isinstance(item, dict):
                continue
            for field_name in KNOWN_SPLIT_FIELDS:
                split = item.get(field_name)
                if split is not None and split not in allowed:
                    check.error(
                        "unknown_split",
                        f"{_item_id(item) or '(无 ID)'} 的 {field_name}={split!r} 不在允许集合中",
                    )
    check.metrics["split_counts"] = dict(
        Counter(
            str(item[field_name])
            for item in items
            if isinstance(item, dict)
            for field_name in KNOWN_SPLIT_FIELDS
            if field_name in item
        )
    )


def _validate_evidence(
    check: DatasetCheck,
    evidence: object,
    fixture_root: Path,
    *,
    case_id: str,
    turn_id: str,
) -> int:
    if not isinstance(evidence, list):
        check.error("evidence_shape", f"{case_id}/{turn_id} 的 evidence 必须是列表")
        return 0
    valid_count = 0
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            check.error("evidence_item", f"{case_id}/{turn_id} evidence[{index}] 必须是对象")
            continue
        raw_path = item.get("path")
        if not _safe_relative_path(raw_path):
            check.error("evidence_path", f"{case_id}/{turn_id} evidence 路径不安全：{raw_path!r}")
            continue
        target = fixture_root / str(raw_path)
        if not target.is_file():
            check.error("evidence_missing", f"{case_id}/{turn_id} Evidence 文件不存在：{raw_path}")
            continue
        start = item.get("start_line")
        end = item.get("end_line")
        line_count = _line_count(target)
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or start > end:
            check.error("evidence_range", f"{case_id}/{turn_id} Evidence 行区间非法：{item}")
            continue
        if end > line_count:
            check.error(
                "evidence_out_of_range",
                f"{case_id}/{turn_id} Evidence {raw_path}:{start}-{end} 超出文件 {line_count} 行",
            )
            continue
        valid_count += 1
    return valid_count


def _contains_forbidden_key(value: object) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_MODEL_CONTENT_KEYS:
                return key
            found = _contains_forbidden_key(nested)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_forbidden_key(item)
            if found:
                return found
    return None


def validate_conversation_document(
    document: dict[str, Any],
    fixture_root: Path,
    *,
    expected_count: int | None = None,
    expected_turn_count: int | None = None,
    expected_sha256: str | None = None,
    actual_sha256: str | None = None,
    review_path: Path | None = None,
    split_manifest: dict[str, Any] | None = None,
    coverage_targets: dict[str, Any] | None = None,
) -> DatasetCheck:
    """校验独立多轮数据集，返回可供测试和报告复用的结果对象。"""
    check = DatasetCheck("conversation_2_1_0", "(document)")
    if document.get("schema_version") != 1:
        check.error("schema_version", "多轮数据集 schema_version 必须为 1")
    cases = document.get("cases")
    if not isinstance(cases, list):
        check.error("cases", "多轮数据集 cases 必须是列表")
        return check
    check.actual_count = len(cases)
    check.declared_count = expected_count
    if isinstance(expected_count, int) and len(cases) != expected_count:
        check.error("count_mismatch", f"多轮 case 数应为 {expected_count}，实际为 {len(cases)}")
    declared_case_count = document.get("case_count")
    if declared_case_count != len(cases):
        check.error("declared_case_count", "case_count 与 cases 实际数量不一致")
    actual_turn_count = 0
    check.declared_turn_count = expected_turn_count
    case_ids: list[str] = []
    families: list[str] = []
    split_by_id: dict[str, str] = {}
    message_locations: dict[str, tuple[str, str]] = {}
    evidence_turn_count = 0
    task_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    minimum_turns = int(document.get("generation_contract", {}).get("minimum_turns_per_case", 4))
    allowed_splits = {"dev", "regression", "golden", "holdout"}

    for case in cases:
        if not isinstance(case, dict):
            check.error("case_shape", "每条 case 必须是对象")
            continue
        case_id = case.get("id")
        family = case.get("family")
        split = case.get("split")
        if not isinstance(case_id, str) or not case_id.strip():
            check.error("missing_id", "多轮 case 缺少非空 id")
            case_id = f"(case-{len(case_ids) + 1})"
        case_ids.append(case_id)
        if not isinstance(family, str) or not family.strip():
            check.error("missing_family", f"{case_id} 缺少 family")
        else:
            families.append(family)
        if split not in allowed_splits:
            check.error("unknown_split", f"{case_id} 的 split={split!r} 非法")
        elif case_id in split_by_id and split_by_id[case_id] != split:
            check.error("split_collision", f"{case_id} 被放入多个 split")
        else:
            split_by_id[case_id] = str(split)
        language = case.get("language")
        if language not in {"zh", "zh-en"}:
            check.error("language", f"{case_id} 的 language 必须为 zh 或 zh-en")
        else:
            language_counts[str(language)] += 1
        scenario = case.get("scenario")
        if not isinstance(scenario, str) or not scenario.strip():
            check.error("scenario", f"{case_id} 缺少 scenario")
        else:
            scenario_counts[scenario] += 1
        tags = case.get("tags")
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag.strip() for tag in tags
        ):
            check.error("tags", f"{case_id} 的 tags 必须是非空字符串列表")
        else:
            tag_counts.update(str(tag) for tag in tags)
        turns = case.get("turns")
        if not isinstance(turns, list) or len(turns) < minimum_turns:
            check.error("turn_count", f"{case_id} 至少需要 {minimum_turns} 轮")
            continue
        actual_turn_count += len(turns)
        turn_ids: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                check.error("turn_shape", f"{case_id} 的 turn 必须是对象")
                continue
            turn_id = turn.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id.strip():
                check.error("missing_turn_id", f"{case_id} 存在缺少 turn_id 的轮次")
                turn_id = f"(turn-{len(turn_ids) + 1})"
            turn_ids.append(turn_id)
            message = turn.get("user_message")
            if not isinstance(message, str) or not message.strip():
                check.error("empty_user_message", f"{case_id}/{turn_id} user_message 为空")
            else:
                normalized = _normalize_message(message)
                previous = message_locations.get(normalized)
                if previous is not None and previous[0] != case_id:
                    check.error(
                        "cross_case_message_leak",
                        f"规范化问题与 {previous[0]}/{previous[1]} 重复：{case_id}/{turn_id}",
                    )
                else:
                    message_locations[normalized] = (case_id, turn_id)
            expected = turn.get("expected")
            if not isinstance(expected, dict):
                check.error("expected_shape", f"{case_id}/{turn_id} expected 必须是对象")
                continue
            task_type = expected.get("task_type")
            task_counts[str(task_type)] += 1
            if task_type not in CONVERSATION_TASK_TYPES:
                check.error("task_type", f"{case_id}/{turn_id} task_type={task_type!r} 非法")
            goal_action = expected.get("goal_action")
            if goal_action not in CONVERSATION_GOAL_ACTIONS:
                check.error("goal_action", f"{case_id}/{turn_id} goal_action={goal_action!r} 非法")
            outcome = expected.get("outcome")
            if outcome not in CONVERSATION_OUTCOMES:
                check.error("outcome", f"{case_id}/{turn_id} outcome={outcome!r} 非法")
            requires_model = expected.get("requires_model")
            if not isinstance(requires_model, bool):
                check.error("requires_model", f"{case_id}/{turn_id} requires_model 必须是 bool")
            elif task_type in {"scope_redirect", "clarify"} and requires_model:
                check.error("model_policy", f"{case_id}/{turn_id} 的 {task_type} 不应调用模型")
            elif task_type in {"explain", "change", "general_chat"} and not requires_model:
                check.error("model_policy", f"{case_id}/{turn_id} 的 {task_type} 应调用真实模型")
            facts = expected.get("expected_facts")
            if not isinstance(facts, list) or not facts or not all(
                isinstance(item, str) and item.strip() for item in facts
            ):
                check.error("expected_facts", f"{case_id}/{turn_id} expected_facts 不能为空")
            for field_name in ("must_retain", "allowed_to_forget"):
                values = expected.get(field_name)
                if not isinstance(values, list) or not values or not all(
                    isinstance(item, str) and item.strip() for item in values
                ):
                    check.error("context_contract", f"{case_id}/{turn_id} 缺少有效的 {field_name}")
            evidence = expected.get("evidence")
            if evidence:
                evidence_turn_count += 1
                _validate_evidence(
                    check,
                    evidence,
                    fixture_root,
                    case_id=case_id,
                    turn_id=turn_id,
                )
            elif task_type == "explain" and outcome in {"answered", "insufficient_evidence"}:
                check.error("missing_evidence", f"{case_id}/{turn_id} 的解释轮缺少 Evidence")
            if task_type == "change":
                contract = expected.get("change_contract")
                valid_contract = isinstance(contract, dict) and (
                    contract.get("must_require_approval") is True
                )
                if not valid_contract:
                    check.error("approval_contract", f"{case_id}/{turn_id} change 必须声明审批约束")
                if outcome != "approval_required":
                    check.error(
                        "change_outcome",
                        f"{case_id}/{turn_id} change 必须停在 approval_required",
                    )
            forbidden_key = _contains_forbidden_key(expected)
            if forbidden_key:
                check.error(
                    "raw_model_content",
                    f"{case_id}/{turn_id} 不得保存模型内容字段 {forbidden_key}",
                )
        if len(turn_ids) != len(set(turn_ids)):
            check.error("duplicate_turn_id", f"{case_id} 存在重复 turn_id")
        if any(
            isinstance(turn, dict)
            and isinstance(turn.get("expected"), dict)
            and turn["expected"].get("task_type") == "change"
            for turn in turns
        ):
            change_indexes = [
                index
                for index, turn in enumerate(turns)
                if isinstance(turn, dict)
                and isinstance(turn.get("expected"), dict)
                and turn["expected"].get("task_type") == "change"
            ]
            if change_indexes[-1] != len(turns) - 1:
                check.error(
                    "change_barrier",
                    f"{case_id} 的 change 轮必须是该链最后一轮，审批后才能继续",
                )
    if len(case_ids) != len(set(case_ids)):
        check.error("duplicate_id", "多轮数据集存在重复 case id")
    if len(families) != len(set(families)):
        check.error("duplicate_family", "同一 family 不能跨 case 重复，避免伪增样本")
    check.actual_turn_count = actual_turn_count
    if document.get("turn_count") != actual_turn_count:
        check.error("declared_turn_count", "turn_count 与实际轮次数不一致")
    if isinstance(expected_turn_count, int) and expected_turn_count != actual_turn_count:
        check.error(
            "turn_count_mismatch",
            f"多轮轮次应为 {expected_turn_count}，实际为 {actual_turn_count}",
        )
    if actual_sha256 and expected_sha256 and expected_sha256 != "pending_verify":
        if actual_sha256.lower() != expected_sha256.lower():
            check.error("hash_mismatch", "多轮数据集 SHA-256 与策略不一致")
    check.metrics.update(
        {
            "case_count": len(cases),
            "turn_count": actual_turn_count,
            "evidence_turn_count": evidence_turn_count,
            "task_type_counts": dict(task_counts),
            "scenario_counts": dict(scenario_counts),
            "language_counts": dict(language_counts),
            "tag_counts": dict(tag_counts),
        }
    )
    _validate_conversation_coverage(
        check,
        document,
        scenario_counts,
        language_counts,
        task_counts,
        tag_counts,
        coverage_targets,
    )
    if split_manifest is not None:
        _validate_split_manifest(check, split_manifest, case_ids, split_by_id, cases)
    if review_path is not None:
        _validate_review(
            check,
            review_path,
            set(case_ids),
            str(document.get("source_commit", "")),
            actual_turn_count,
        )
    return check


def _validate_conversation_coverage(
    check: DatasetCheck,
    document: dict[str, Any],
    scenario_counts: Counter[str],
    language_counts: Counter[str],
    task_counts: Counter[str],
    tag_counts: Counter[str],
    coverage_targets: dict[str, Any] | None,
) -> None:
    contract = document.get("generation_contract")
    if not isinstance(contract, dict):
        check.error("generation_contract", "缺少 generation_contract")
        return
    required_scenarios = contract.get("required_scenarios", [])
    if isinstance(required_scenarios, list):
        missing = sorted(set(map(str, required_scenarios)) - set(scenario_counts))
        if missing:
            check.error("scenario_coverage", f"缺少场景覆盖：{missing}")
    split_counts = contract.get("splits")
    if isinstance(split_counts, dict):
        actual = Counter(
            str(case.get("split"))
            for case in document.get("cases", [])
            if isinstance(case, dict)
        )
        for split, expected in split_counts.items():
            if actual[str(split)] != expected:
                check.error(
                    "split_coverage",
                    f"split={split} 应有 {expected} 条，实际 {actual[str(split)]} 条",
                )
    languages = contract.get("languages")
    if isinstance(languages, dict):
        for language, expected in languages.items():
            if language_counts[str(language)] != expected:
                check.error(
                    "language_coverage",
                    f"language={language} 应有 {expected} 条，实际 "
                    f"{language_counts[str(language)]} 条",
                )
    if not task_counts.get("explain"):
        check.error("task_coverage", "多轮数据集至少要有 explain 轮")
    if isinstance(coverage_targets, dict):
        case_targets = coverage_targets.get("case_targets")
        if isinstance(case_targets, dict):
            minimum_case_count = case_targets.get("minimum_case_count")
            case_count = len(document.get("cases", []))
            if isinstance(minimum_case_count, int) and case_count < minimum_case_count:
                check.error(
                    "case_quantity",
                    f"多轮 case 数低于覆盖目标 {minimum_case_count}",
                )
            for key, expected in (case_targets.get("minimum_cases_by_scenario") or {}).items():
                if isinstance(expected, int) and scenario_counts[str(key)] < expected:
                    check.error(
                        "scenario_quantity",
                        f"场景 {key} 至少需要 {expected} 条，实际 {scenario_counts[str(key)]} 条",
                    )
            for key, expected in (case_targets.get("minimum_cases_by_language") or {}).items():
                if isinstance(expected, int) and language_counts[str(key)] < expected:
                    check.error(
                        "language_quantity",
                        f"语言 {key} 至少需要 {expected} 条，实际 {language_counts[str(key)]} 条",
                    )
        turn_targets = coverage_targets.get("turn_targets")
        if isinstance(turn_targets, dict):
            for key, expected in (turn_targets.get("minimum_task_type_count") or {}).items():
                if isinstance(expected, int) and task_counts[str(key)] < expected:
                    check.error(
                        "task_quantity",
                        f"task_type={key} 至少需要 {expected} 轮，实际 {task_counts[str(key)]} 轮",
                    )
            for key, expected in (turn_targets.get("minimum_noise_tags") or {}).items():
                if isinstance(expected, int) and tag_counts[str(key)] < expected:
                    check.error(
                        "tag_quantity",
                        f"标签 {key} 至少需要 {expected} 个 case，实际 {tag_counts[str(key)]} 个",
                    )
            signal_counts = Counter()
            for case in document.get("cases", []):
                if not isinstance(case, dict):
                    continue
                for turn in case.get("turns", []):
                    if not isinstance(turn, dict) or not isinstance(turn.get("expected"), dict):
                        continue
                    expected = turn["expected"]
                    if expected.get("must_retain"):
                        signal_counts["must_retain"] += 1
                    if expected.get("allowed_to_forget"):
                        signal_counts["allowed_to_forget"] += 1
                    if expected.get("evidence"):
                        signal_counts["evidence_bearing_turns"] += 1
            for key, expected in (turn_targets.get("minimum_context_signal_count") or {}).items():
                if isinstance(expected, int) and signal_counts[str(key)] < expected:
                    check.error(
                        "context_signal_quantity",
                        f"上下文信号 {key} 至少需要 {expected} 轮，实际 "
                        f"{signal_counts[str(key)]} 轮",
                    )
    check.metrics["coverage_ready"] = not any(issue.fatal for issue in check.issues)


def _validate_split_manifest(
    check: DatasetCheck,
    manifest: dict[str, Any],
    case_ids: list[str],
    split_by_id: dict[str, str],
    cases: list[Any],
) -> None:
    if manifest.get("dataset_id") != "conversation_2_1_0":
        check.error("split_dataset", "split manifest dataset_id 不匹配")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        check.error("split_shape", "split manifest 缺少 splits")
        return
    listed: dict[str, str] = {}
    for split, payload in splits.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("case_ids"), list):
            check.error("split_shape", f"split={split} 必须有 case_ids 列表")
            continue
        for case_id in payload["case_ids"]:
            if not isinstance(case_id, str):
                check.error("split_id", f"split={split} 含非字符串 case id")
                continue
            if case_id in listed:
                check.error("split_leak", f"{case_id} 在 split manifest 中重复")
            listed[case_id] = str(split)
    if set(listed) != set(case_ids):
        check.error("split_membership", "split manifest 与数据集 case 集合不一致")
    for case_id, split in split_by_id.items():
        if listed.get(case_id) != split:
            check.error("split_assignment", f"{case_id} 的数据 split 与 manifest 不一致")
    for case in cases:
        is_holdout = isinstance(case, dict) and case.get("split") == "holdout"
        if is_holdout and case.get("holdout_policy") != "never_tune":
            check.error("holdout_policy", f"holdout case {case.get('id')} 缺少 never_tune 声明")
    tuning = manifest.get("split_policy", {}).get("tuning", [])
    if isinstance(tuning, list) and "holdout" in tuning:
        check.error("holdout_in_tuning", "holdout 不得出现在 tuning split")


def _validate_review(
    check: DatasetCheck,
    review_path: Path,
    case_ids: set[str],
    source_commit: str,
    turn_count: int,
) -> None:
    if not review_path.is_file():
        check.error("review_missing", f"语义审查记录不存在：{review_path}")
        return
    rows: list[dict[str, Any]] = []
    review_lines = review_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw in enumerate(review_lines, start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            check.error("review_json", f"审查记录第 {line_number} 行不是 JSON：{error.msg}")
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            check.error("review_shape", f"审查记录第 {line_number} 行必须是对象")
    review_ids = [str(row.get("case_id")) for row in rows]
    if set(review_ids) != case_ids:
        check.error("review_membership", "语义审查没有一一覆盖全部多轮 case")
    if len(review_ids) != len(set(review_ids)):
        check.error("review_duplicate", "语义审查记录含重复 case_id")
    for row in rows:
        case_id = row.get("case_id", "(unknown)")
        for field_name in (
            "reviewer",
            "review_version",
            "source_commit",
            "turn_count",
            "evidence_turn_count",
            "decision",
            "evidence_basis",
            "notes",
        ):
            if field_name not in row:
                check.error("review_field", f"{case_id} 缺少审查字段 {field_name}")
        if row.get("decision") != "pass":
            check.error("review_decision", f"{case_id} 不是 pass，不能进入真实模型 pilot")
        if row.get("source_commit") != source_commit:
            check.error("review_commit", f"{case_id} 审查源码 commit 不匹配")
        if not isinstance(row.get("turn_count"), int) or row.get("turn_count") < 1:
            check.error("review_turn_count", f"{case_id} turn_count 不是整数")
    checked_turns = sum(
        int(row.get("turn_count", 0))
        for row in rows
        if isinstance(row.get("turn_count"), int)
    )
    if checked_turns != turn_count:
        check.error(
            "review_turn_total",
            f"审查记录轮次总和 {checked_turns} 与数据集 {turn_count} 不一致",
        )
    check.metrics["semantic_review_case_count"] = len(review_ids)
    check.metrics["semantic_review_coverage"] = (
        len(set(review_ids)) / len(case_ids) if case_ids else 0.0
    )


def _resolve(root: Path, raw: object) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def validate_policy(
    policy_path: Path,
    *,
    baseline_path: Path | None = None,
    repo_root: Path | None = None,
) -> QualityReport:
    root = (repo_root or policy_path.resolve().parents[2]).resolve()
    report = QualityReport(str(policy_path), str(baseline_path) if baseline_path else None)
    try:
        policy = load_mapping(policy_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report.issues.append(QualityIssue("policy_parse", str(error)))
        return report
    baseline: dict[str, Any] | None = None
    if baseline_path is not None:
        try:
            baseline = load_mapping(baseline_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            report.issues.append(QualityIssue("baseline_parse", str(error)))
    entries = policy.get("datasets")
    if not isinstance(entries, list) or not entries:
        report.issues.append(QualityIssue("policy_datasets", "策略缺少 datasets 列表"))
        return report
    split_manifest: dict[str, Any] | None = None
    split_path = _resolve(root, policy.get("split_manifest"))
    if split_path is not None and split_path.is_file():
        try:
            split_manifest = load_mapping(split_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            report.issues.append(QualityIssue("split_parse", str(error)))
    elif split_path is not None:
        report.issues.append(QualityIssue("split_missing", f"split manifest 不存在：{split_path}"))
    coverage_targets: dict[str, Any] | None = None
    coverage_path = _resolve(root, policy.get("coverage_targets"))
    if coverage_path is not None and coverage_path.is_file():
        try:
            coverage_targets = load_mapping(coverage_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            report.issues.append(QualityIssue("coverage_parse", str(error)))
    elif coverage_path is not None:
        report.issues.append(
            QualityIssue("coverage_missing", f"coverage targets 不存在：{coverage_path}")
        )
    for entry in entries:
        if not isinstance(entry, dict):
            report.issues.append(QualityIssue("dataset_entry", "dataset 登记项必须是对象"))
            continue
        dataset_id = str(entry.get("id", "(missing-id)"))
        raw_path = entry.get("path")
        data_path = _resolve(root, raw_path)
        check = DatasetCheck(dataset_id, str(raw_path))
        report.checks.append(check)
        if data_path is None or not data_path.is_file():
            check.error("file_missing", f"数据文件不存在或路径越界：{raw_path}")
            check.finalize(str(entry.get("quality_status", "DATASET_BLOCKED")))
            continue
        check.actual_sha256 = sha256_file(data_path)
        expected_sha = entry.get("expected_sha256")
        sha_mismatch = (
            isinstance(expected_sha, str)
            and expected_sha != "pending_verify"
            and check.actual_sha256.lower() != expected_sha.lower()
        )
        if sha_mismatch:
            check.error("hash_mismatch", f"{dataset_id} SHA-256 与策略不一致")
        try:
            document = load_mapping(data_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            check.error("dataset_parse", str(error))
            check.finalize(str(entry.get("quality_status", "DATASET_BLOCKED")))
            continue
        if baseline is not None:
            baseline_entry = baseline.get("datasets", {}).get(dataset_id)
            if not isinstance(baseline_entry, dict):
                check.error("baseline_registration", f"{dataset_id} 未登记在 baseline datasets 中")
            else:
                if baseline_entry.get("path") != raw_path:
                    check.error("baseline_path", f"{dataset_id} 的 baseline path 与策略不一致")
                if baseline_entry.get("sha256", "").lower() != str(expected_sha).lower():
                    check.error("baseline_hash", f"{dataset_id} 的 baseline SHA-256 与策略不一致")
                expected_count = entry.get("expected_count")
                baseline_count = baseline_entry.get("cases", baseline_entry.get("tasks"))
                if isinstance(expected_count, int) and baseline_count != expected_count:
                    check.error("baseline_count", f"{dataset_id} 的 baseline 数量与策略不一致")
        if dataset_id == "conversation_2_1_0":
            fixture_raw = document.get("fixture")
            fixture_root = root / "backend" / str(fixture_raw)
            review_path = _resolve(root, entry.get("review_record"))
            conversation_check = validate_conversation_document(
                document,
                fixture_root,
                expected_count=entry.get("expected_count"),
                expected_turn_count=entry.get("expected_turn_count"),
                expected_sha256=expected_sha if isinstance(expected_sha, str) else None,
                actual_sha256=check.actual_sha256,
                review_path=review_path,
                split_manifest=split_manifest,
                coverage_targets=coverage_targets,
            )
            check.declared_count = conversation_check.declared_count
            check.actual_count = conversation_check.actual_count
            check.declared_turn_count = conversation_check.declared_turn_count
            check.actual_turn_count = conversation_check.actual_turn_count
            check.metrics.update(conversation_check.metrics)
            check.issues.extend(conversation_check.issues)
            check.warnings.extend(conversation_check.warnings)
        else:
            _validate_generic_dataset(check, document, entry)
        check.finalize(str(entry.get("quality_status", "DATASET_BLOCKED")))
    report.metrics["dataset_count"] = len(report.checks)
    report.metrics["states"] = dict(Counter(check.state for check in report.checks))
    return report


def _print_report(report: QualityReport) -> None:
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="校验登记的数据集")
    validate.add_argument("--policy", required=True, type=Path)
    validate.add_argument("--baseline", type=Path)
    validate.add_argument("--repo-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        report = validate_policy(args.policy, baseline_path=args.baseline, repo_root=args.repo_root)
        _print_report(report)
        return 0 if report.accepted else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
