"""按工具类型裁剪旧工具结果，为高成本的 full summary 争取空间。

通用的 head/tail 截断会破坏 JSON 工具结果：截掉一半的 JSON 既不能解析，
也看不出被截了什么。这里改成按工具语义投影——保留路径、行号、符号、
计数和错误结论，丢掉大段正文。

两条硬规则：

1. 失败结果永远完整保留。错误码、工具名和参数校验结论是模型下一步的
   唯一线索，裁掉它等于让模型在错误状态下继续猜。
2. `assistant tool_call` 与 `tool result` 必须成对。只替换 tool
   result 的内容，不删除、不重排、不合并消息。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from codeinsight.application.context_budget import estimate_tokens

TOOL_RESULT_PRUNE_VERSION = "tool-result-prune-v1"
PRUNE_MARKER = "[已按工具类型裁剪，完整结果仍在本次 Run 的工具产物中]"

# 这些工具的输出就是当前工作状态，裁剪会让模型在过期信息上做决定。
PROTECTED_TOOLS: frozenset[str] = frozenset(
    {"get_diff", "generate_patch", "validate_patch"}
)

_EXCERPT_CHARS = 160


@dataclass(frozen=True)
class PrunedToolResult:
    """一条被裁剪的工具结果及其前后 token 估算。"""

    call_id: str
    tool_name: str
    tokens_before: int
    tokens_after: int


@dataclass(frozen=True)
class PruneOutcome:
    """裁剪后的消息序列与可审计的裁剪记录。"""

    messages: tuple[Mapping[str, object], ...]
    pruned: tuple[PrunedToolResult, ...]
    protected_call_ids: tuple[str, ...]
    tokens_before: int
    tokens_after: int
    version: str = TOOL_RESULT_PRUNE_VERSION

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


class ToolResultPruner:
    """确定性的工具结果裁剪器；不调用模型，不发明内容。"""

    def __init__(
        self, *, keep_recent_results: int = 2, min_tokens_to_prune: int = 120
    ) -> None:
        if keep_recent_results < 0:
            raise ValueError("keep_recent_results 不能为负")
        if min_tokens_to_prune < 0:
            raise ValueError("min_tokens_to_prune 不能为负")
        self._keep_recent = keep_recent_results
        self._min_tokens = min_tokens_to_prune

    def prune(
        self,
        messages: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
        *,
        keep_recent: int | None = None,
    ) -> PruneOutcome:
        items = list(messages)
        keep = self._keep_recent if keep_recent is None else keep_recent
        if keep < 0:
            raise ValueError("keep_recent 不能为负")
        tool_indexes = [
            index
            for index, message in enumerate(items)
            if str(message.get("role", "")) == "tool"
        ]
        candidates = tool_indexes[: max(0, len(tool_indexes) - keep)]
        before = _messages_tokens(items)
        if not candidates:
            return PruneOutcome(
                tuple(items), (), tuple(_call_id(items[i]) for i in tool_indexes), before, before
            )

        pruned: list[PrunedToolResult] = []
        protected: list[str] = [
            _call_id(items[index]) for index in tool_indexes[len(tool_indexes) - keep :]
        ]
        result: list[Mapping[str, object]] = list(items)
        for index in candidates:
            message = items[index]
            call_id = _call_id(message)
            payload = _decode_payload(message.get("content"))
            if payload is None:
                protected.append(call_id)
                continue
            tool_name = str(payload.get("tool", ""))
            if payload.get("ok") is not True:
                # 失败结果完整保留：错误码和参数结论是下一步的唯一线索。
                protected.append(call_id)
                continue
            if tool_name in PROTECTED_TOOLS:
                protected.append(call_id)
                continue
            raw_tokens = estimate_tokens(str(message.get("content", "")))
            if raw_tokens < self._min_tokens:
                protected.append(call_id)
                continue
            new_payload = dict(payload)
            new_payload["data"] = _project(tool_name, payload.get("data"))
            new_payload["pruned"] = True
            new_payload["prune_version"] = TOOL_RESULT_PRUNE_VERSION
            content = json.dumps(new_payload, ensure_ascii=False, sort_keys=True)
            after_tokens = estimate_tokens(content)
            if after_tokens >= raw_tokens:
                protected.append(call_id)
                continue
            updated = dict(message)
            updated["content"] = content
            result[index] = updated
            pruned.append(
                PrunedToolResult(call_id, tool_name, raw_tokens, after_tokens)
            )
        return PruneOutcome(
            tuple(result),
            tuple(pruned),
            tuple(protected),
            before,
            _messages_tokens(result),
        )


def _messages_tokens(messages: list[Mapping[str, object]]) -> int:
    return estimate_tokens(json.dumps(messages, ensure_ascii=False, default=str))


def _call_id(message: Mapping[str, object]) -> str:
    return str(message.get("tool_call_id", ""))


def _decode_payload(content: object) -> dict[str, object] | None:
    if not isinstance(content, str):
        return None
    try:
        decoded = json.loads(content)
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _EXCERPT_CHARS:
        return collapsed
    return collapsed[:_EXCERPT_CHARS] + "…"


def _project(tool_name: str, data: object) -> object:
    """按工具语义投影成功结果的 data。未知工具走保守的通用收缩。"""
    if not isinstance(data, Mapping):
        return _shrink(data)
    if tool_name == "read_file":
        return {
            "path": data.get("path"),
            "start_line": data.get("start_line"),
            "end_line": data.get("end_line"),
            "text_excerpt": _excerpt(str(data.get("text", ""))),
            "untrusted": True,
            "note": "完整内容可用同一 path 与行范围重新调用 read_file",
            "marker": PRUNE_MARKER,
        }
    if tool_name in {"search_repository", "get_evidence_context"}:
        key = "results" if "results" in data else "evidence"
        items = data.get(key)
        projected: list[object] = []
        if isinstance(items, (list, tuple)):
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                projected.append(
                    {
                        "relative_path": item.get("relative_path"),
                        "start_line": item.get("start_line"),
                        "end_line": item.get("end_line"),
                        "symbol_path": item.get("symbol_path"),
                        "score": item.get("score"),
                        "rank": item.get("rank"),
                        "text_excerpt": _excerpt(str(item.get("text", ""))),
                        "untrusted": True,
                    }
                )
        return {
            key: projected,
            "untrusted": True,
            "note": "命中片段已截断，可用同一 question 重新检索或对具体路径调用 read_file",
            "marker": PRUNE_MARKER,
        }
    if tool_name == "get_repository_map":
        return _shrink_repository_map(data)
    if tool_name in {"lsp_definition", "scip_references"}:
        return _shrink(data, string_limit=_EXCERPT_CHARS)
    if tool_name == "get_run_events":
        events = data.get("events")
        types: dict[str, int] = {}
        if isinstance(events, (list, tuple)):
            for event in events:
                if isinstance(event, Mapping):
                    name = str(event.get("event_type", "unknown"))
                    types[name] = types.get(name, 0) + 1
        return {
            "event_type_counts": types,
            "note": "事件正文已裁剪，可用 after_sequence 重新读取需要的区段",
            "marker": PRUNE_MARKER,
        }
    return _shrink(data)


def _shrink_repository_map(data: Mapping[str, object]) -> dict[str, object]:
    def names(key: str, limit: int) -> list[str]:
        raw = data.get(key)
        found: list[str] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, Mapping):
                    value = item.get("path") or item.get("name") or item.get("symbol")
                    if value is not None:
                        found.append(str(value))
                elif isinstance(item, str):
                    found.append(item)
                if len(found) >= limit:
                    break
        return found

    return {
        "files": names("files", 50),
        "symbols": names("symbols", 50),
        "imports": names("imports", 50),
        "next_cursor": data.get("next_cursor"),
        "total_files": (
            len(data.get("files", ())) if isinstance(data.get("files"), (list, tuple)) else None
        ),
        "note": "地图只用于导航，不是证据；需要代码事实请调用 read_file 或检索",
        "marker": PRUNE_MARKER,
    }


def _shrink(value: object, *, string_limit: int = 200, depth: int = 0) -> object:
    if isinstance(value, str):
        return value if len(value) <= string_limit else value[:string_limit] + "…"
    if isinstance(value, Mapping):
        if depth >= 4:
            return PRUNE_MARKER
        items = list(value.items())
        projected: dict[str, object] = {}
        for key, item in items[:40]:
            projected[str(key)] = _shrink(item, string_limit=string_limit, depth=depth + 1)
        if len(items) > 40:
            projected["omitted_keys"] = len(items) - 40
        return projected
    if isinstance(value, (list, tuple)):
        if depth >= 4:
            return PRUNE_MARKER
        projected_items = [
            _shrink(item, string_limit=string_limit, depth=depth + 1)
            for item in list(value)[:20]
        ]
        if len(value) > 20:
            projected_items.append({"omitted_items": len(value) - 20})
        return projected_items
    return value
