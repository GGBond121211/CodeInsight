"""长会话上下文生命周期的评测入口（Q-010）。

两种模式：

* ``--mode fake``：脚本化 Provider + 脚本化 MCP Host。不花钱、可重复，
  用来验证「压缩 → 保留 → 恢复 → 复用」这条链路没坏。它**不产生质量结论**：
  回答内容、证据和步数都由脚本决定，不含模型判断。
* ``--mode real``：真实 Provider。只跑少量轮次，用来证明同一套链路在真实
  模型下同样成立；受 ``--max-total-tokens`` 约束，默认不跑全量。

两种模式的证据分开记录：Fake 结果不能写进「真实模型结论」。

示例（从 backend 目录运行）::

    python tests/evals/context_lifecycle_long_horizon.py --mode fake
    python tests/evals/context_lifecycle_long_horizon.py --mode real \
        --case-id lh-01-long-explain --max-total-tokens 400000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from codeinsight.infrastructure.chat_endpoint import EXPECTED_CHAT_BASE_URL  # noqa: E402
from codeinsight.infrastructure.model_profiles import DEFAULT_MODEL_ID  # noqa: E402

CASES_PATH = BACKEND_ROOT / "tests" / "evals" / "context_lifecycle_cases.json"
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evals" / "context-lifecycle-long-horizon.json"

# 评测用 Session 的保留轮数。线上默认 24 轮足以掩盖压缩；这里刻意调小，
# 让 20 轮以内的会话就能反复触发压缩，而不是等累积到几十轮才第一次执行。
EVAL_RECENT_TURNS = 4
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "WAITING_APPROVAL"}
EVIDENCE_INDEX_MARKER = "[evidence-index]"
EVIDENCE_ID_PATTERN = re.compile(r"\bE[1-9][0-9]*\b")
SOURCE_PATH_PATTERN = re.compile(r"(?<![\w/.-])([\w][\w./-]*\.py)\b")

# 只放行模型端点配置。MySQL/Redis 未启动时，公开 Store 必须继续走内存实现，
# 否则「本机没起数据库」会把评测变成一次连接失败。
ENV_ALLOWLIST = frozenset(
    {
        "CODEINSIGHT_API_KEY",
        "CODEINSIGHT_MODEL",
        "CODEINSIGHT_BASE_URL",
        # 检索链路自己的凭据。缺了它们，真实模式里 search_repository /
        # get_evidence_context 会因为构造不出 Embedding 而返回 INVALID_RESPONSE，
        # 于是每一轮都停在 STUCK，测到的只是「本机没配检索」而不是上下文生命周期。
        "CODEINSIGHT_EMBEDDING_API_KEY",
        "CODEINSIGHT_EMBEDDING_BASE_URL",
        "CODEINSIGHT_EMBEDDING_MODEL",
        "CODEINSIGHT_EMBEDDING_OUTPUT_TYPE",
        "CODEINSIGHT_RERANK_API_KEY",
        "CODEINSIGHT_RERANK_BASE_URL",
        "CODEINSIGHT_RERANK_MODEL",
    }
)

# 评测本地限额。20–50 轮的连续会话会耗尽 Gateway 默认的按租户 Token Bucket
# （容量 2,000,000，按「估算输入 + 输出预留」计费，补充速率 10,000/s）：
# 实测第 10 轮起就返回 BACKPRESSURE，测到的是限流而不是上下文生命周期。
# 抬高的是这两个**本地评测**限额，不是生产参数结论。
# 评测工作窗口。模型真实窗口是 1,000,000，但评测得先把会话历史压过预算才能验证
# 压缩；靠 message_padding_chars 铺够 1M 输入要烧掉几十万 token，而且需要生成
# 几百万字符的填充文本。这里显式把工作窗口压到 128K，让「压过预算」这条路径在
# 可承受的用量内可达。窗口大小不改变被测逻辑，只改变触发点；真实模式同样适用，
# 所以它同时也是一道省钱措施。
EVAL_CONTEXT_WINDOW_TOKENS = 128_000

EVAL_RATE_LIMIT_CAPACITY = 64_000_000
EVAL_RATE_LIMIT_REFILL_PER_SECOND = 2_000_000
EVAL_TENANT_TOKEN_LIMIT = 64_000_000

# 用例可以把 message_padding_chars 打开，给每条 user 消息追加一段固定填充。
# 用途只有一个：在没有真实长会话的前提下把 Session 历史推过 token 预算，从而
# 让「按 token 压缩」这条路径在评测里也能被触发。填充是确定性的，因此
# message_sha256 仍然可复现。
PADDING_UNIT = "def handle_request(payload): return ledger.append(payload)  # filler"


def _load_project_env() -> None:
    """为真实模式读取项目根 .env；不打印、不持久化任何密钥。"""
    candidates = (PROJECT_ROOT / ".env", PROJECT_ROOT.parent.parent / ".env")
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip(chr(34)).strip(chr(39))
            if key in ENV_ALLOWLIST and value:
                os.environ[key] = value
        if os.environ.get("CODEINSIGHT_BASE_URL"):
            return


def load_cases(case_ids: set[str] | None) -> dict:
    document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = document["cases"]
    if case_ids:
        selected = [case for case in cases if case["id"] in case_ids]
        missing = case_ids - {case["id"] for case in selected}
        if missing:
            raise SystemExit(f"未找到用例：{sorted(missing)}")
        cases = selected
    document["cases"] = cases
    return document


def _unavailable(name: str):
    """Fake 模式必须离线：任何未装配的依赖都要立刻失败，而不是偷偷走网络。"""

    def _raise():
        raise RuntimeError(f"评测未配置 {name}")

    return _raise


def _announced_evidence_ids(messages) -> tuple[str, ...]:
    """从可见消息里的 [evidence-index] 段落读出应用回填过的证据编号。

    脚本化 Provider 只能引用这里出现过的编号，和真实模型的约束一致。
    """
    found: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or EVIDENCE_INDEX_MARKER not in content:
            continue
        for token in EVIDENCE_ID_PATTERN.findall(content):
            if token not in found:
                found.append(token)
    return tuple(found)


def _mentioned_source_paths(messages) -> tuple[str, ...]:
    """从可见消息里抽出被提到的 .py 路径。

    证据评估按问题里出现的硬锚点比对路径，脚本化工具必须读「问题里那条
    路径」，否则用例会一直停在「证据没有覆盖硬锚点」。
    """
    found: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for token in SOURCE_PATH_PATTERN.findall(content):
            normalized = token.replace("\\", "/")
            if not normalized.startswith("/") and ".." not in normalized.split("/"):
                if normalized not in found:
                    found.append(normalized)
    return tuple(found)


def _fake_mcp_client_factory(payload_chars: int, fail_on_call: int):
    """脚本化 MCP Host：大工具输出 + 可复现的工具失败。"""

    class _Host:
        """每个 Run 新建一个实例，调用计数因此按轮重置，失败可复现。"""

        def __init__(self, root: str) -> None:
            self._root = root
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def list_tools(self):
            from codeinsight.infrastructure.tool_registry import build_default_registry

            return build_default_registry().list_tools()

        def call_tool(self, call):
            from codeinsight.agent.tool_loop import ToolResult

            self._calls += 1
            if self._calls == fail_on_call:
                return ToolResult.failure(
                    call.id, call.name, "NOT_FOUND", "脚本化的可恢复失败"
                )
            # 回显模型请求的路径：证据锚点按问题里出现的路径比对，路径对不上
            # 就会被判成「证据没覆盖硬锚点」，脚本用例会一直停在证据不足。
            arguments = getattr(call, "arguments", None) or {}
            path = str(arguments.get("path") or "src/service.py")
            start_line = arguments.get("start_line")
            end_line = arguments.get("end_line")
            if not isinstance(start_line, int) or start_line < 1:
                start_line = 1 + 40 * (self._calls - 1)
            if not isinstance(end_line, int) or end_line < start_line:
                end_line = start_line + 39
            return ToolResult.success(
                call.id,
                call.name,
                {
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": "def handler(): pass" * (payload_chars // 20),
                    "untrusted": True,
                },
                state_fingerprint=f"fp-{self._calls}",
            )

    return _Host


class _ScriptedEvidenceProvider:
    """脚本化 Provider：证据不够就继续读文件，凑够两条就按契约收口。

    Router 调用（不带 tools）返回一份合法的路由计划，使用例稳定落在 explain
    的只读 Tool Loop 上，而不是靠 Router 解析失败去走回退分支。
    """

    def __init__(self, *, model: str, evidence_target: int = 2) -> None:
        self.model = model
        self.evidence_target = evidence_target
        self.calls = 0

    def _router_response(self, model: str):
        from codeinsight.infrastructure.provider_adapters import ProviderResponse

        plan = json.dumps(
            {
                "language": "zh",
                "normalized_question": "脚本化问题",
                "subquestions": [
                    {
                        "question": "脚本化问题",
                        "intent": "implementation",
                        "retrieval_mode": "hybrid",
                    }
                ],
                "execution_route": "linear",
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )
        return ProviderResponse(plan, (), model, 9_000, 60, 7_000, "stop")

    def invoke(self, request):
        from codeinsight.agent.tool_loop import ToolCall
        from codeinsight.infrastructure.provider_adapters import ProviderResponse

        self.calls += 1
        model = request.model or self.model
        if not request.tools:
            return self._router_response(model)
        evidence_ids = _announced_evidence_ids(request.messages)
        if len(evidence_ids) >= self.evidence_target:
            content = json.dumps(
                {
                    "outcome": "answered",
                    "answer": "脚本化回答：可用证据已在本次上下文中。",
                    "citations": list(evidence_ids[: self.evidence_target]),
                },
                ensure_ascii=False,
            )
            return ProviderResponse(content, (), model, 12_000, 80, 9_000, "stop")
        call = ToolCall(
            f"call-{self.calls}",
            "read_file",
            self._read_arguments(request.messages),
        )
        return ProviderResponse(None, (call,), model, 12_000, 40, 9_000, "tool_calls")

    def _read_arguments(self, messages) -> dict[str, object]:
        """换一个行区间再读。

        每次都用同一组参数会被循环判成「重复调用」而提前停在 STUCK，而重复的
        行区间又会被证据台账去重；两者都会让脚本用例收不了口。
        """
        start_line = 1 + 40 * (self.calls - 1)
        return {
            "path": self._next_path(messages),
            "start_line": start_line,
            "end_line": start_line + 39,
        }

    def _next_path(self, messages) -> str:
        # 只看问题本身（第一条 user 消息）：证据锚点来自问题，从历史消息里
        # 挑路径会让脚本读到上一轮的文件，评估就永远判「锚点未覆盖」。
        question_only = [
            message
            for message in messages
            if str(message.get("role")) == "user"
        ][:1]
        candidates = _mentioned_source_paths(question_only) or _mentioned_source_paths(
            messages
        )
        if not candidates:
            return f"src/module{self.calls}.py"
        return candidates[(self.calls - 1) % len(candidates)]


def _wait_for_terminal(client, turn_id: str, timeout_seconds: float) -> dict:
    deadline = perf_counter() + timeout_seconds
    latest: dict = {}
    while perf_counter() < deadline:
        response = client.get(f"/api/v2/chat/turns/{turn_id}")
        response.raise_for_status()
        latest = response.json()
        if latest.get("status") in TERMINAL_STATUSES:
            return latest
        time.sleep(0.2)
    return latest


def _turn_events(client, run_id: str) -> tuple:
    """事件直接从 Runtime 的事件日志读。

    ``/api/v2/chat/turns/{id}/events`` 是 SSE 流，评测只需要事后计数，读日志更直接。
    """
    service = client.app.state.codeinsight_conversation
    return tuple(service.runtime.event_log.read_events(run_id))


def _event_counts(events) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        name = str(event.event_type)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _compaction_detail(events) -> tuple[list[str], dict]:
    """把一次轮次的压缩按触发点分类。

    Q-010 第一版里评测报出的「压缩触发率」其实全是「保留轮数」驱动的：评测把
    ``session_recent_turns`` 调到 4，第 3 轮起每轮都压缩，和 token 预算无关。
    这里把触发点单独记下来，读结果的人才能分清两种压缩。
    """
    triggers = sorted(
        {
            str(event.payload.get("trigger") or "retained_window")
            for event in events
            if event.event_type == "context_compacted"
        }
    )
    detail = next(
        (
            event.payload
            for event in reversed(events)
            if event.event_type == "context_compacted"
            and event.payload.get("trigger") == "session_history_budget"
        ),
        None,
    )
    if detail is None:
        return triggers, {}
    return triggers, {
        "outcome": str(detail.get("outcome") or ""),
        "surface_tokens_before": _as_int(detail.get("surface_tokens_before")),
        "surface_tokens_after": _as_int(detail.get("surface_tokens_after")),
        "limit_tokens": _as_int(detail.get("limit_tokens")),
        "dropped_turns": _as_int(detail.get("dropped_turns")),
    }


def _last_payload(events, event_type: str) -> dict:
    for event in reversed(events):
        if event.event_type == event_type:
            return dict(event.payload)
    return {}


def _session_snapshot(client, session_id: str) -> dict:
    response = client.get(
        f"/api/v2/chat/sessions/{session_id}",
        params={"repository_root": str(FIXTURE_ROOT)},
    )
    if response.status_code != 200:
        return {}
    return response.json()


def _create_session(client) -> str:
    created = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    )
    created.raise_for_status()
    return str(created.json()["session_id"])


def _run_case(
    case: dict,
    *,
    mode: str,
    timeout_seconds: float,
    limit: int,
    payload_chars: int,
    fail_on_call: int,
    recent_turns: int,
) -> tuple[dict, dict]:
    """跑一个 case：同一 Session 连续多轮，可选在中途重建应用实例。"""
    from contextlib import ExitStack  # noqa: I001

    from fastapi.testclient import TestClient

    from codeinsight.api.app import create_app
    from codeinsight.application.change_service import ChangeService
    from codeinsight.application.context_assembler import ContextAssembler
    from codeinsight.application.conversation_service import ConversationService
    from codeinsight.application.session_service import SessionService
    from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
    from codeinsight.infrastructure.model_gateway import (
        GatewayChatModel,
        InMemoryBudgetLedger,
        ModelGateway,
        TokenBucketRateLimiter,
        default_gateway_from_environment,
    )
    from codeinsight.infrastructure.run_store import InMemorySessionStore
    from codeinsight.infrastructure.redis_cache import InMemoryCache

    if mode == "real":
        from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
        from codeinsight.infrastructure.reranker import OpenAITextReranker

        # 进程内共享状态（配额、熔断、语义缓存）按 case 重建，避免前一条
        # 会话把后一条的预算耗尽，把质量结果污染成「预算用完了」。
        default_gateway_from_environment.cache_clear()
        gateway: ModelGateway = default_gateway_from_environment()
        embedding_factory = OpenAIEmbeddingModel.from_environment
        reranker_factory = OpenAITextReranker.from_environment
    else:
        gateway = ModelGateway(
            provider=_ScriptedEvidenceProvider(
                model=os.environ.get("CODEINSIGHT_MODEL", DEFAULT_MODEL_ID)
            ),
            budget_ledger=InMemoryBudgetLedger(default_limit=EVAL_TENANT_TOKEN_LIMIT),
            rate_limiter=TokenBucketRateLimiter(
                capacity=EVAL_RATE_LIMIT_CAPACITY,
                refill_tokens_per_second=EVAL_RATE_LIMIT_REFILL_PER_SECOND,
            ),
        )
        embedding_factory = _unavailable("Embedding")
        reranker_factory = _unavailable("Rerank")

    # 一个 case 共用一个 Session 事实层，但每次重建应用实例都换一个新的
    # SessionService 和一份新的内存缓存：重启就等于「新的进程内缓存 + 同一个
    # 事实 Store」，恢复只能来自 Store。真正的跨进程恢复需要 MySQL，由本机
    # 默认跳过的集成测试覆盖，这里不冒充。
    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    host_factory = (
        _fake_mcp_client_factory(payload_chars, fail_on_call)
        if mode == "fake"
        else None
    )

    def model_factory() -> GatewayChatModel:
        return GatewayChatModel(gateway, context_assembler=ContextAssembler())

    def open_client() -> TestClient:
        session_service = SessionService(
            session_store,
            memory_store,
            InMemoryCache(),
            max_recent_turns=recent_turns,
        )
        service = ConversationService(
            model_factory,
            embedding_factory,
            reranker_factory=reranker_factory,
            change_service=ChangeService(),
            session_service=session_service,
            mcp_client_factory=host_factory,
        )
        return TestClient(
            create_app(
                model_factory=model_factory,
                usage_gateway_factory=lambda: gateway,
                conversation_service=service,
            )
        )

    messages = list(case["turns"])
    repeat_of = {int(key): int(value) for key, value in (case.get("repeat_of") or {}).items()}
    restart_after = case.get("restart_after_turn")
    padding_chars = int(case.get("message_padding_chars", 0) or 0)
    padding = ""
    if padding_chars > 0:
        repeats = padding_chars // len(PADDING_UNIT) + 1
        padding = (PADDING_UNIT * repeats)[:padding_chars]
    turns: list[dict] = []
    latencies: list[float] = []
    restarts = 0

    stack = ExitStack()
    try:
        client = stack.enter_context(open_client())
        session_id = _create_session(client)
        for index, original_message in enumerate(messages, start=1):
            user_message = (
                messages[repeat_of[index] - 1] if index in repeat_of else original_message
            )
            if padding:
                user_message = f"{user_message} {padding}"
            started = perf_counter()
            record: dict = {
                "turn": index,
                "repeat_of": repeat_of.get(index),
                "message_sha256": hashlib.sha256(
                    user_message.encode("utf-8")
                ).hexdigest()[:16],
                "message_chars": len(user_message),
            }
            try:
                accepted = client.post(
                    "/api/v2/chat/turns",
                    json={
                        "session_id": session_id,
                        "repository_root": str(FIXTURE_ROOT),
                        "message": user_message,
                        "client_turn_id": f"lh-{case['id']}-{index}",
                        "limit": limit,
                        "show_debug_reasoning": False,
                    },
                )
                accepted.raise_for_status()
                payload = accepted.json()
                actual = _wait_for_terminal(
                    client, payload["turn_id"], timeout_seconds
                )
                latency = (perf_counter() - started) * 1000
                latencies.append(latency)
                events = _turn_events(client, actual["run_id"])
                counts = _event_counts(events)
                compaction_triggers, budget_compaction = _compaction_detail(events)
                assembled = _last_payload(events, "context_assembled")
                loaded = _last_payload(events, "session_loaded")
                result = actual.get("result")
                citations = (
                    result.get("citations") if isinstance(result, dict) else None
                )
                loop = (
                    result.get("tool_loop") if isinstance(result, dict) else None
                )
                loop = loop if isinstance(loop, dict) else {}
                model_results = [
                    event for event in events if event.event_type == "model_result"
                ]
                record.update(
                    {
                        "status": actual.get("status"),
                        "task_type": actual.get("task_type"),
                        "outcome": (
                            result.get("outcome") if isinstance(result, dict) else None
                        ),
                        "citation_count": (
                            len(citations) if isinstance(citations, list) else 0
                        ),
                        "answer_chars": len(str(actual.get("assistant_message") or "")),
                        "latency_ms": latency,
                        "history_turns": int(assembled.get("history_turns", 0) or 0),
                        "history_tokens": _as_int(assembled.get("history_tokens")),
                        "history_budget_tokens": _as_int(
                            assembled.get("history_budget_tokens")
                        ),
                        "summary_present": assembled.get("summary_present") == "true",
                        "session_cache_hit": loaded.get("cache_hit") == "true",
                        "compacted": counts.get("context_compacted", 0) > 0,
                        "compaction_triggers": compaction_triggers,
                        "budget_compaction": budget_compaction or None,
                        "overflow": counts.get("context_overflow", 0) > 0,
                        "model_calls": counts.get("model_called", 0),
                        "loop_status": loop.get("loop_status"),
                        "loop_termination_reason": loop.get("termination_reason"),
                        "loop_steps": loop.get("steps"),
                        "loop_tool_calls": loop.get("tool_calls"),
                        "loop_unknown_evidence_ids": loop.get("unknown_evidence_ids"),
                        "model_error_classes": sorted(
                            {
                                str(event.payload.get("error_class"))
                                for event in model_results
                                if event.payload.get("outcome") == "error"
                                and event.payload.get("error_class")
                            }
                        ),
                        "model_error_details": sorted(
                            {
                                str(event.payload.get("error_detail"))[:160]
                                for event in model_results
                                if event.payload.get("outcome") == "error"
                                and event.payload.get("error_detail")
                            }
                        ),
                        "event_types": sorted(counts),
                    }
                )
            except Exception as error:  # noqa: BLE001 - 评测只记录失败类别
                record.update(
                    {"status": "EVAL_ERROR", "error_class": type(error).__name__}
                )
                turns.append(record)
                break
            turns.append(record)
            if restart_after is not None and index == restart_after:
                stack.close()
                stack = ExitStack()
                client = stack.enter_context(open_client())
                restarts += 1
        snapshot = _session_snapshot(client, session_id)
    finally:
        stack.close()

    return (
        {
            "case_id": case["id"],
            "scenario": case.get("scenario"),
            "turns_requested": len(messages),
            "turns_executed": len(turns),
            "restarts": restarts,
            "compaction_turns": sum(1 for turn in turns if turn.get("compacted")),
            "restore_hits": sum(1 for turn in turns if turn.get("session_cache_hit")),
            "final_summary_present": bool(snapshot.get("summary")),
            "final_compacted_through_sequence": snapshot.get(
                "compacted_through_sequence"
            ),
            "final_recent_turns": len(snapshot.get("recent_turns") or []),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "turns": turns,
        },
        gateway.usage_summary(),
    )


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(ratio * (len(ordered) - 1))))
    return round(ordered[index], 2)


def _summarize(records: list[dict], usage: dict) -> dict[str, object]:
    """把逐轮记录汇总成计划第 4 节列出的指标。

    三个缓存口径分开报告，不合成一个数字：上游前缀命中、Gateway 语义缓存
    和「上下文原样装得下」是三件不同的事，混在一起就会把本地缓存省下的
    一次调用误读成上游前缀命中率高。
    """
    turns = [turn for record in records for turn in record["turns"]]
    completed = [turn for turn in turns if turn.get("status") == "COMPLETED"]
    failed = [turn for turn in turns if turn.get("status") != "COMPLETED"]
    latencies = [float(turn.get("latency_ms", 0.0)) for turn in turns]
    reasons: dict[str, int] = {}
    for turn in turns:
        reason = str(turn.get("loop_termination_reason") or "").strip()
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    trigger_counts: dict[str, int] = {}
    for turn in turns:
        for trigger in turn.get("compaction_triggers") or ():
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
    cache_read = int(usage.get("provider_prompt_cache_read_tokens", 0) or 0)
    cache_miss = int(usage.get("provider_prompt_cache_miss_tokens", 0) or 0)
    checked = int(usage.get("context_retention_checked", 0) or 0)
    overflow = int(usage.get("context_retention_overflow", 0) or 0)
    requests = int(usage.get("requests", 0) or 0)
    return {
        "case_count": len(records),
        "turns_executed": len(turns),
        "terminal_turns": len(completed),
        "failed_turns": len(failed),
        "failure_rate": len(failed) / len(turns) if turns else 0.0,
        "compaction_trigger_rate": (
            sum(1 for turn in turns if turn.get("compacted")) / len(turns)
            if turns
            else 0.0
        ),
        "compaction_events": sum(1 for turn in turns if turn.get("compacted")),
        # 轮数压缩和 token 预算压缩的触发原因完全不同，混成一个数字会把
        # 「评测把保留轮数调到 4」误读成「上下文预算在起作用」。
        "compaction_trigger_counts": dict(sorted(trigger_counts.items())),
        "budget_compacted_turns": sum(
            1 for turn in turns if turn.get("budget_compaction")
        ),
        "max_history_tokens": max(
            (int(turn.get("history_tokens", 0) or 0) for turn in turns), default=0
        ),
        "history_budget_tokens": next(
            (
                int(turn.get("history_budget_tokens", 0) or 0)
                for turn in turns
                if turn.get("history_budget_tokens")
            ),
            0,
        ),
        # 轮询状态 COMPLETED 只说明 HTTP 轮次跑完，不代表循环收口成功；把两者
        # 分开报，避免「26 轮 0 失败」掩盖 12 轮循环停在 STUCK 的事实。
        "answered_turns": sum(1 for turn in turns if turn.get("outcome") == "answered"),
        "insufficient_turns": sum(
            1 for turn in turns if turn.get("outcome") == "insufficient_evidence"
        ),
        "loop_completed_turns": sum(
            1 for turn in turns if turn.get("loop_status") == "COMPLETED"
        ),
        "loop_stuck_turns": sum(
            1 for turn in turns if turn.get("loop_status") == "STUCK"
        ),
        "loop_failed_turns": sum(
            1 for turn in turns if turn.get("loop_status") == "FAILED"
        ),
        "loop_termination_reasons": dict(
            sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        ),
        "summary_present_turns": sum(1 for turn in turns if turn.get("summary_present")),
        "summary_output_tokens_cap": 8_192,
        "session_restore_hit_turns": sum(
            1 for turn in turns if turn.get("session_cache_hit")
        ),
        "context_retention_checked": checked,
        "context_retention_pass": int(usage.get("context_retention_pass", 0) or 0),
        "context_retention_overflow": overflow,
        "overflow_rate": (overflow / checked) if checked else 0.0,
        "provider_prompt_cache_read_tokens": cache_read,
        "provider_prompt_cache_miss_tokens": cache_miss,
        "provider_prompt_cache_read_ratio": (
            cache_read / (cache_read + cache_miss) if cache_read + cache_miss else 0.0
        ),
        "semantic_response_cache_hits": int(
            usage.get("semantic_response_cache_hits", 0) or 0
        ),
        "semantic_response_cache_hit_ratio": (
            int(usage.get("semantic_response_cache_hits", 0) or 0) / requests
            if requests
            else 0.0
        ),
        "gateway_requests": requests,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "failed_cases": sorted(
            record["case_id"]
            for record in records
            if record["turns_executed"] < record["turns_requested"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fake", "real"], default="fake")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=40_960)
    parser.add_argument("--max-total-tokens", type=int, default=400_000)
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=EVAL_CONTEXT_WINDOW_TOKENS,
        help="评测工作窗口；默认 128K，见 experiments/configs/context_lifecycle_128k.yaml",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--fake-tool-payload-chars",
        type=int,
        default=6_000,
        help="fake 模式每次工具结果的正文字符数；调大用来压上下文",
    )
    parser.add_argument(
        "--fake-fail-on-call",
        type=int,
        default=2,
        help="fake 模式每个 Run 的第几次工具调用返回可恢复失败",
    )
    args = parser.parse_args()

    if args.mode == "real":
        _load_project_env()
        configured = os.environ.get("CODEINSIGHT_BASE_URL", "").rstrip("/")
        if configured != EXPECTED_CHAT_BASE_URL:
            raise SystemExit(
                "拒绝运行：真实模式必须使用项目 Frontier 网关 "
                f"{EXPECTED_CHAT_BASE_URL}，当前为 {configured or '<missing>'}"
            )
        if not os.environ.get("CODEINSIGHT_API_KEY", "").strip():
            raise SystemExit("拒绝运行：未配置 CODEINSIGHT_API_KEY")
        if os.environ.get("CODEINSIGHT_GATEWAY_FAKE_MODE") == "1":
            raise SystemExit("拒绝运行：CODEINSIGHT_GATEWAY_FAKE_MODE=1")
        if args.max_total_tokens > 9_000_000:
            raise SystemExit("--max-total-tokens 超过单次验收周期上限")
    os.environ["CODEINSIGHT_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)
    os.environ["CODEINSIGHT_CONTEXT_WINDOW_TOKENS"] = str(args.context_window_tokens)

    document = load_cases(set(args.case_ids) if args.case_ids else None)
    records: list[dict] = []
    usage_totals: dict[str, int] = {}
    tokens_used = 0
    for case in document["cases"]:
        record, usage = _run_case(
            case,
            mode=args.mode,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            payload_chars=int(
                case.get("fake_tool_payload_chars", args.fake_tool_payload_chars)
            ),
            fail_on_call=int(case.get("fake_fail_on_call", args.fake_fail_on_call)),
            recent_turns=int(case.get("session_recent_turns", EVAL_RECENT_TURNS)),
        )
        record["gateway_usage"] = {
            key: usage[key]
            for key in (
                "requests",
                "input_tokens",
                "output_tokens",
                "provider_prompt_cache_read_tokens",
                "provider_prompt_cache_miss_tokens",
                "provider_prompt_cache_hit_ratio",
                "semantic_response_cache_hits",
                "context_retention_checked",
                "context_retention_pass",
                "context_retention_overflow",
            )
        }
        records.append(record)
        for key in (
            "requests",
            "input_tokens",
            "output_tokens",
            "provider_prompt_cache_read_tokens",
            "provider_prompt_cache_miss_tokens",
            "semantic_response_cache_hits",
            "context_retention_checked",
            "context_retention_pass",
            "context_retention_overflow",
        ):
            usage_totals[key] = usage_totals.get(key, 0) + int(usage.get(key, 0) or 0)
        tokens_used += int(usage["input_tokens"]) + int(usage["output_tokens"])
        if args.mode == "real" and tokens_used >= args.max_total_tokens:
            break

    payload = {
        "schema_version": 1,
        "dataset_id": document["dataset_id"],
        "dataset_version": document["dataset_version"],
        "dataset_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "mode": args.mode,
        "real_provider_only": args.mode == "real",
        "fake_provider_used": args.mode == "fake",
        "provider_host": (
            urlsplit(os.environ.get("CODEINSIGHT_BASE_URL", "")).hostname
            if args.mode == "real"
            else None
        ),
        "model_configured": os.environ.get("CODEINSIGHT_MODEL"),
        "provider_client_max_retries": 0,
        "output_token_cap": args.max_output_tokens,
        "context_window_tokens": args.context_window_tokens,
        "max_total_tokens": args.max_total_tokens,
        "tokens_used": tokens_used,
        "eval_recent_turns": EVAL_RECENT_TURNS,
        "summary": _summarize(records, usage_totals),
        "cases": records,
        "notes": [
            "fake 模式只验证链路结构，不产生质量结论；回答、证据和步数都由脚本决定。",
            "真实模式只跑少量轮次，未做质量对照，也不代表参数已调优。",
            "context_window_tokens 是评测工作窗口，不是模型能力上限；生产默认是 1,000,000。",
            "只保存路由、压缩、缓存、usage 与延迟元数据，不保存用户问题正文和模型答案正文。",
            "进程内重建应用实例只证明 Session 从 Store 恢复；"
            "跨进程恢复需要 MySQL，不在本机默认范围内。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
