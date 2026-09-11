from __future__ import annotations

import json

from codeinsight.application.tool_result_pruner import (
    PROTECTED_TOOLS,
    TOOL_RESULT_PRUNE_VERSION,
    ToolResultPruner,
)


def _tool_message(
    call_id: str,
    tool: str,
    data: object,
    *,
    ok: bool = True,
    fingerprint: str | None = "fp-1",
) -> dict[str, object]:
    payload: dict[str, object] = {"ok": ok, "tool": tool, "data": data}
    if fingerprint is not None:
        payload["state_fingerprint"] = fingerprint
    if not ok:
        payload["error"] = {"code": "VALIDATION", "message": "参数不合法"}
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def _assistant(call_id: str, tool: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": (
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": "{}"},
            },
        ),
    }


def _big_read(call_id: str) -> dict[str, object]:
    return _tool_message(
        call_id,
        "read_file",
        {
            "path": "src/app.py",
            "start_line": 1,
            "end_line": 200,
            "text": "def handler(): pass\n" * 200,
            "untrusted": True,
        },
    )


def _conversation() -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "safe"},
        {"role": "user", "content": "解释入口"},
    ]
    for index in range(1, 5):
        call_id = f"call-{index}"
        messages.append(_assistant(call_id, "read_file"))
        messages.append(_big_read(call_id))
    return messages


def _tool_content(messages: object, call_id: str) -> dict[str, object]:
    for message in messages:  # type: ignore[union-attr]
        if message.get("tool_call_id") == call_id:
            return json.loads(str(message["content"]))
    raise AssertionError(f"没有找到 {call_id} 的工具结果")


def test_old_successful_results_are_pruned_and_recent_ones_are_kept() -> None:
    outcome = ToolResultPruner(keep_recent_results=2).prune(_conversation())

    assert [item.call_id for item in outcome.pruned] == ["call-1", "call-2"]
    assert outcome.tokens_saved > 0
    assert outcome.version == TOOL_RESULT_PRUNE_VERSION


def test_pairing_and_order_survive_pruning() -> None:
    messages = _conversation()
    outcome = ToolResultPruner(keep_recent_results=2).prune(messages)

    assert [message.get("role") for message in outcome.messages] == [
        message.get("role") for message in messages
    ]
    surviving = [
        message.get("tool_call_id")
        for message in outcome.messages
        if message.get("role") == "tool"
    ]
    assert surviving == ["call-1", "call-2", "call-3", "call-4"]
    for index, message in enumerate(outcome.messages):
        if message.get("role") == "tool":
            assert messages[index].get("tool_call_id") == message.get("tool_call_id")


def test_read_file_keeps_location_and_fingerprint_but_drops_the_body() -> None:
    outcome = ToolResultPruner(keep_recent_results=2).prune(_conversation())
    payload = _tool_content(outcome.messages, "call-1")

    assert payload["ok"] is True
    assert payload["state_fingerprint"] == "fp-1"
    assert payload["data"]["path"] == "src/app.py"
    assert payload["data"]["start_line"] == 1
    assert payload["data"]["end_line"] == 200
    assert "text" not in payload["data"]
    assert len(str(payload["data"]["text_excerpt"])) < 400
    assert payload["pruned"] is True


def test_search_results_keep_hit_locations_and_short_excerpts() -> None:
    message = _tool_message(
        "call-search",
        "search_repository",
        {
            "results": [
                {
                    "relative_path": "src/a.py",
                    "start_line": 10,
                    "end_line": 20,
                    "symbol_path": "A.run",
                    "score": 0.9,
                    "rank": 1,
                    "text": "payload " * 500,
                    "untrusted": True,
                }
            ]
        },
    )
    messages = [
        {"role": "user", "content": "q"},
        _assistant("call-search", "search_repository"),
        message,
        _assistant("call-later", "read_file"),
        _big_read("call-later"),
    ]
    outcome = ToolResultPruner(keep_recent_results=1).prune(messages)
    payload = json.loads(str(outcome.messages[2]["content"]))
    hit = payload["data"]["results"][0]

    assert hit["relative_path"] == "src/a.py"
    assert hit["start_line"] == 10
    assert hit["end_line"] == 20
    assert hit["symbol_path"] == "A.run"
    assert len(hit["text_excerpt"]) < 400


def test_repository_map_is_reduced_to_navigation_names() -> None:
    message = _tool_message(
        "call-map",
        "get_repository_map",
        {
            "files": [{"path": f"src/m{index}.py"} for index in range(60)],
            "symbols": [
                {"name": f"Sym{index}", "path": "src/m.py"} for index in range(60)
            ],
            "imports": ["os", "sys"] * 40,
            "next_cursor": "c-2",
        },
    )
    messages = [
        {"role": "user", "content": "q"},
        _assistant("call-map", "get_repository_map"),
        message,
        _assistant("call-later", "read_file"),
        _big_read("call-later"),
    ]
    outcome = ToolResultPruner(keep_recent_results=1).prune(messages)
    payload = json.loads(str(outcome.messages[2]["content"]))

    assert payload["data"]["files"][0] == "src/m0.py"
    assert len(payload["data"]["files"]) == 50
    assert payload["data"]["next_cursor"] == "c-2"


def test_failed_results_are_never_pruned() -> None:
    failed = _tool_message("call-bad", "read_file", {}, ok=False, fingerprint=None)
    messages = [
        {"role": "user", "content": "q"},
        _assistant("call-bad", "read_file"),
        failed,
        _assistant("call-later", "read_file"),
        _big_read("call-later"),
    ]
    outcome = ToolResultPruner(keep_recent_results=1).prune(messages)

    assert outcome.pruned == ()
    assert outcome.messages[2] is messages[2]
    assert "call-bad" in outcome.protected_call_ids


def test_protected_tools_are_never_pruned() -> None:
    payload = "diff --git a/src/app.py b/src/app.py\n" * 300
    messages: list[dict[str, object]] = []
    for tool in sorted(PROTECTED_TOOLS):
        call_id = f"call-{tool}"
        messages.append(_assistant(call_id, tool))
        messages.append(_tool_message(call_id, tool, {"diff": payload}))
    messages.append(_assistant("call-later", "read_file"))
    messages.append(_big_read("call-later"))

    outcome = ToolResultPruner(keep_recent_results=1).prune(messages)

    assert outcome.pruned == ()
    protected = set(outcome.protected_call_ids)
    assert protected >= {f"call-{tool}" for tool in PROTECTED_TOOLS}


def test_small_results_and_malformed_payloads_are_left_alone() -> None:
    small = _tool_message("call-small", "read_file", {"path": "a.py", "text": "x"})
    broken = {
        "role": "tool",
        "tool_call_id": "call-broken",
        "content": "not json at all",
    }
    messages = [
        {"role": "user", "content": "q"},
        _assistant("call-small", "read_file"),
        small,
        _assistant("call-broken", "read_file"),
        broken,
        _assistant("call-later", "read_file"),
        _big_read("call-later"),
    ]
    outcome = ToolResultPruner(keep_recent_results=1).prune(messages)

    assert outcome.pruned == ()
    assert outcome.messages[2] is small
    assert outcome.messages[4] is broken


def test_everything_recent_is_left_untouched() -> None:
    messages = _conversation()
    outcome = ToolResultPruner(keep_recent_results=10).prune(messages)

    assert outcome.pruned == ()
    assert outcome.tokens_saved == 0
    assert list(outcome.messages) == messages
