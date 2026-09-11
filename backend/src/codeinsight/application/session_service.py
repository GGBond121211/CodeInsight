"""Session、Goal 与三层 Memory 的应用层闭环。"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from codeinsight.application.context_budget import ContextLifecyclePolicy, estimate_tokens
from codeinsight.domain.change import (
    GOAL_ACTIVE,
    CodeGoal,
    ConversationSession,
    ConversationTurn,
    TenantScope,
)
from codeinsight.domain.memory import (
    LAYER_SEMANTIC,
    LAYER_SESSION,
    LAYER_WORKING,
    SESSION_SUMMARY_VERSION,
    MemoryCacheKey,
    MemoryRecord,
    SemanticMemory,
    SessionCompactionError,
    SessionCompactionResult,
    SessionCompactionSummary,
    SessionMemory,
    WorkingMemory,
)
from codeinsight.domain.ports import CacheStore, MemoryStore, SessionStore
from codeinsight.infrastructure.redis_cache import CacheUnavailableError, cache_aside

SESSION_MEMORY_SOURCE = "session_service"


@dataclass(frozen=True)
class SessionContext:
    """一次会话恢复的完整结果。"""

    session: ConversationSession
    memory: SessionMemory
    active_goal: CodeGoal | None
    cache_hit: bool = False
    cache_fallback: bool = False


class SessionService:
    """以 Session/Memory Store 为事实层，以 Redis 为可失效热点缓存。"""

    def __init__(
        self,
        session_store: SessionStore,
        memory_store: MemoryStore,
        cache: CacheStore | None = None,
        *,
        cache_ttl_seconds: int = 300,
        max_recent_turns: int = 24,
        summary_max_tokens: int | None = None,
    ) -> None:
        if cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds 必须为正")
        if max_recent_turns < 2:
            raise ValueError("max_recent_turns 至少为 2")
        self._session_store = session_store
        self._memory_store = memory_store
        self._cache = cache
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_recent_turns = max_recent_turns
        # 摘要上限来自生命周期策略的单一来源；显式传参只为测试和灰度。
        self._summary_max_tokens = (
            summary_max_tokens
            if summary_max_tokens is not None
            else ContextLifecyclePolicy().summary_max_tokens
        )
        if self._summary_max_tokens <= 0:
            raise ValueError("summary_max_tokens 必须为正")

    def get_or_create_session(
        self,
        *,
        session_id: str,
        scope: TenantScope,
        repo_id: str,
        repo_fingerprint: str,
        index_version: str,
    ) -> SessionContext:
        key = self._session_cache_key(
            session_id=session_id,
            scope=scope,
            repo_id=repo_id,
            repo_fingerprint=repo_fingerprint,
            index_version=index_version,
        )

        def load_from_stores() -> str:
            session = self._session_store.get_session(session_id)
            if session is None:
                session = ConversationSession(
                    session_id=session_id,
                    scope=scope,
                    repo_id=repo_id,
                )
                self._session_store.save_session(session)
            self._assert_session_scope(session, scope=scope, repo_id=repo_id)

            memory = self.load_session_memory(
                session_id=session_id,
                scope=scope,
                repo_id=repo_id,
            )
            if memory is None:
                memory = SessionMemory(
                    session_id=session_id,
                    recent_turns=session.recent_turns,
                    active_goal_id=session.active_goal_id,
                    summary=session.summary,
                    compacted_through_sequence=session.compacted_through_sequence,
                )
                self._save_session_memory(memory, scope=scope, repo_id=repo_id)
            session = replace(
                session,
                recent_turns=memory.recent_turns,
                summary=memory.summary,
                active_goal_id=memory.active_goal_id,
                compacted_through_sequence=memory.compacted_through_sequence,
            )
            active_goal = self._load_active_goal(session)
            return _context_to_json(
                SessionContext(session=session, memory=memory, active_goal=active_goal)
            )

        cached = cache_aside(
            self._cache,
            key,
            load_from_stores,
            ttl_seconds=self._cache_ttl_seconds,
        )
        context = _context_from_json(cached.value)
        self._assert_session_scope(context.session, scope=scope, repo_id=repo_id)
        return replace(
            context,
            cache_hit=cached.hit,
            cache_fallback=cached.used_fallback,
        )

    def load_session(self, session_id: str) -> ConversationSession | None:
        """按 session_id 读回会话事实。

        Worker 在另一个进程里只有 session_id；它就是靠这一条知道该读哪个仓库。
        """

        return self._session_store.get_session(session_id)

    def bind_repository_root(self, session_id: str, repo_root: str) -> bool:
        """把会话绑定的仓库根落成事实。返回是否真的写了一次。

        这是受理路径上的热调用，因此只在第一次绑定或根目录变化时写库。
        """

        normalized = str(Path(repo_root).resolve())
        session = self._session_store.get_session(session_id)
        if session is None:
            raise ValueError(f"会话不存在：{session_id}")
        if session.repo_root == normalized:
            return False
        self._session_store.save_session(replace(session, repo_root=normalized))
        return True

    def append_turn(
        self,
        context: SessionContext,
        *,
        role: str,
        content: str,
        repo_fingerprint: str,
        index_version: str,
    ) -> SessionContext:
        session = context.session.with_turn(role, content)
        memory = replace(context.memory, recent_turns=session.recent_turns)
        compacted_context, _ = self._compact_context(
            replace(context, session=session, memory=memory, cache_hit=False),
            max_recent_turns=self._max_recent_turns,
        )
        return self._persist_context(
            compacted_context,
            repo_fingerprint=repo_fingerprint,
            index_version=index_version,
        )

    def compact_session(
        self,
        context: SessionContext,
        *,
        repo_fingerprint: str,
        index_version: str,
        max_recent_turns: int | None = None,
        max_context_tokens: int | None = None,
    ) -> tuple[SessionContext, SessionCompactionResult]:
        """显式压缩 Session 历史，并返回可审计的丢弃范围。

        压缩只生成逐轮截断摘要，不调用模型、不发明结论；真正的代码事实仍需
        重新检索并进入 Evidence。默认 append_turn 已按 ``max_recent_turns`` 自动
        触发，本方法供 Gateway/恢复流程在 token 预算更紧时主动触发。
        """
        if max_recent_turns is not None and max_recent_turns < 2:
            raise ValueError("max_recent_turns 至少为 2")
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens 必须为正")
        compacted, result = self._compact_context(
            context,
            max_recent_turns=max_recent_turns,
            max_context_tokens=max_context_tokens,
        )
        if result.dropped_turn_sequences:
            compacted = self._persist_context(
                compacted,
                repo_fingerprint=repo_fingerprint,
                index_version=index_version,
            )
        return compacted, result

    def update_session_memory(
        self,
        context: SessionContext,
        *,
        summary: str | None = None,
        confirmed_conclusions: tuple[str, ...] | None = None,
        rejected_approaches: tuple[str, ...] | None = None,
        user_preferences: dict[str, str] | None = None,
        repo_fingerprint: str,
        index_version: str,
    ) -> SessionContext:
        memory = replace(
            context.memory,
            summary=summary if summary is not None else context.memory.summary,
            confirmed_conclusions=(
                confirmed_conclusions
                if confirmed_conclusions is not None
                else context.memory.confirmed_conclusions
            ),
            rejected_approaches=(
                rejected_approaches
                if rejected_approaches is not None
                else context.memory.rejected_approaches
            ),
            user_preferences=(
                user_preferences
                if user_preferences is not None
                else context.memory.user_preferences
            ),
        )
        session = replace(context.session, summary=memory.summary)
        return self._persist_context(
            replace(context, session=session, memory=memory, cache_hit=False),
            repo_fingerprint=repo_fingerprint,
            index_version=index_version,
        )

    def continue_or_create_goal(
        self,
        context: SessionContext,
        *,
        user_goal: str,
        task_type: str,
        mode: str,
        target_scope: tuple[str, ...] = (),
        validation_profile: str | None = None,
        start_new: bool = False,
        goal_id: str | None = None,
        repo_fingerprint: str,
        index_version: str,
    ) -> SessionContext:
        current = context.active_goal
        if not start_new and current is not None and current.status == GOAL_ACTIVE:
            return context

        new_goal = CodeGoal(
            goal_id=goal_id or str(uuid.uuid4()),
            session_id=context.session.session_id,
            scope=context.session.scope,
            repo_id=context.session.repo_id,
            task_type=task_type,
            user_goal=user_goal,
            mode=mode,
            target_scope=target_scope,
            validation_profile=validation_profile,
        )
        self._session_store.save_goal(new_goal)
        session = context.session.with_active_goal(new_goal.goal_id)
        memory = replace(context.memory, active_goal_id=new_goal.goal_id)
        return self._persist_context(
            replace(
                context,
                session=session,
                memory=memory,
                active_goal=new_goal,
                cache_hit=False,
            ),
            repo_fingerprint=repo_fingerprint,
            index_version=index_version,
        )

    def save_working_memory(
        self,
        memory: WorkingMemory,
        *,
        scope: TenantScope,
        repo_id: str,
    ) -> None:
        self._save_memory_record(
            layer=LAYER_WORKING,
            owner_id=memory.run_id,
            payload=asdict(memory),
            scope=scope,
            repo_id=repo_id,
        )

    def load_working_memory(
        self,
        *,
        run_id: str,
        scope: TenantScope,
        repo_id: str,
    ) -> WorkingMemory | None:
        record = self._load_memory_record(
            layer=LAYER_WORKING,
            owner_id=run_id,
            scope=scope,
            repo_id=repo_id,
        )
        if record is None:
            return None
        payload = json.loads(record.content)
        return WorkingMemory(
            run_id=str(payload["run_id"]),
            user_goal=str(payload["user_goal"]),
            selected_evidence_ids=tuple(payload.get("selected_evidence_ids", ())),
            rejected_paths=tuple(payload.get("rejected_paths", ())),
            steps_remaining=int(payload.get("steps_remaining", 0)),
            structured_problem=(
                str(payload["structured_problem"])
                if payload.get("structured_problem") is not None
                else None
            ),
            pending_tool=(
                str(payload["pending_tool"])
                if payload.get("pending_tool") is not None
                else None
            ),
            patch_ref=(
                str(payload["patch_ref"]) if payload.get("patch_ref") is not None else None
            ),
            check_status=(
                str(payload["check_status"])
                if payload.get("check_status") is not None
                else None
            ),
        )

    def save_semantic_memory(
        self,
        memory: SemanticMemory,
        *,
        scope: TenantScope,
        source: str,
        confidence: float,
        consent: bool,
        expires_at_epoch_ms: int | None = None,
    ) -> None:
        self._save_memory_record(
            layer=LAYER_SEMANTIC,
            owner_id=memory.index_version,
            payload=asdict(memory),
            scope=scope,
            repo_id=memory.repo_id,
            source=source,
            confidence=confidence,
            consent=consent,
            expires_at_epoch_ms=expires_at_epoch_ms,
        )

    def load_semantic_memory(
        self,
        *,
        repo_id: str,
        index_version: str,
        scope: TenantScope,
    ) -> SemanticMemory | None:
        record = self._load_memory_record(
            layer=LAYER_SEMANTIC,
            owner_id=index_version,
            scope=scope,
            repo_id=repo_id,
        )
        if record is None:
            return None
        payload = json.loads(record.content)
        return SemanticMemory(
            repo_id=str(payload["repo_id"]),
            index_version=str(payload["index_version"]),
            module_summaries=tuple(
                (str(path), str(summary))
                for path, summary in payload.get("module_summaries", ())
            ),
            symbol_names=tuple(str(name) for name in payload.get("symbol_names", ())),
        )

    def load_session_memory(
        self,
        *,
        session_id: str,
        scope: TenantScope,
        repo_id: str,
    ) -> SessionMemory | None:
        record = self._load_memory_record(
            layer=LAYER_SESSION,
            owner_id=session_id,
            scope=scope,
            repo_id=repo_id,
        )
        if record is None:
            return None
        return _session_memory_from_payload(json.loads(record.content))

    def forget_working_memory(
        self,
        *,
        run_id: str,
        scope: TenantScope,
        repo_id: str,
    ) -> None:
        self._delete_memory_record(
            layer=LAYER_WORKING,
            owner_id=run_id,
            scope=scope,
            repo_id=repo_id,
        )

    def forget_semantic_memory(
        self,
        *,
        repo_id: str,
        index_version: str,
        scope: TenantScope,
    ) -> None:
        self._delete_memory_record(
            layer=LAYER_SEMANTIC,
            owner_id=index_version,
            scope=scope,
            repo_id=repo_id,
        )

    def forget_session_memory(
        self,
        *,
        session_id: str,
        scope: TenantScope,
        repo_id: str,
        repo_fingerprint: str,
        index_version: str,
    ) -> None:
        self._delete_memory_record(
            layer=LAYER_SESSION,
            owner_id=session_id,
            scope=scope,
            repo_id=repo_id,
        )
        key = self._session_cache_key(
            session_id=session_id,
            scope=scope,
            repo_id=repo_id,
            repo_fingerprint=repo_fingerprint,
            index_version=index_version,
        )
        if self._cache is not None:
            try:
                self._cache.delete(key.value)
            except CacheUnavailableError:
                pass

    def _compact_context(
        self,
        context: SessionContext,
        *,
        max_recent_turns: int | None = None,
        max_context_tokens: int | None = None,
    ) -> tuple[SessionContext, SessionCompactionResult]:
        memory = context.memory
        original_turns = list(memory.recent_turns)
        before = estimate_tokens(_session_history_text(memory))
        if not original_turns:
            return context, SessionCompactionResult((), (), before, before, False)

        kept = list(original_turns)
        dropped: list[ConversationTurn] = []

        def over_budget() -> bool:
            if max_context_tokens is None:
                return False
            # 候选 Memory 必须带上「如果现在停下」的边界，否则剩余轮次的
            # sequence 与 compacted_through_sequence 对不上，SessionMemory 的
            # 连续性校验会直接拒绝构造。这条路径在 Q-010 第一版从未被执行，
            # 因此这个错误一直没有暴露。
            boundary = (
                dropped[-1].sequence if dropped else memory.compacted_through_sequence
            )
            candidate = replace(
                memory,
                recent_turns=tuple(kept),
                compacted_through_sequence=boundary,
            )
            return estimate_tokens(_session_history_text(candidate)) > max_context_tokens

        while kept and (
            (max_recent_turns is not None and len(kept) > max_recent_turns)
            or over_budget()
        ):
            dropped.append(kept.pop(0))

        if not dropped:
            sequences = tuple(turn.sequence for turn in original_turns)
            return context, SessionCompactionResult(
                (), sequences, before, before, False
            )

        last_sequence = dropped[-1].sequence
        compaction_count = memory.compaction_count + 1
        boundary_id = _compaction_boundary_id(
            memory.session_id, last_sequence, compaction_count
        )
        try:
            structured = _build_structured_summary(
                boundary_id=boundary_id,
                previous=memory.structured_summary,
                existing_summary=memory.summary,
                dropped=dropped,
                active_goal=_active_goal_text(context.active_goal),
                confirmed_conclusions=memory.confirmed_conclusions,
                rejected_approaches=memory.rejected_approaches,
            )
        except SessionCompactionError:
            # 摘要建不起来时保持历史完整，并把失败类别交给调用方观测。
            sequences = tuple(turn.sequence for turn in original_turns)
            return context, SessionCompactionResult(
                (),
                sequences,
                before,
                before,
                False,
                None,
                "SUMMARY_BUILD_FAILED",
            )
        structured = _enforce_summary_budget(structured, self._summary_max_tokens)
        summary = structured.render()
        compacted_memory = replace(
            memory,
            recent_turns=tuple(kept),
            summary=summary,
            compacted_through_sequence=last_sequence,
            compaction_count=compaction_count,
            compaction_boundary_id=boundary_id,
            summary_hash=structured.summary_hash,
            structured_summary=structured,
        )
        compacted_session = replace(
            context.session,
            recent_turns=tuple(kept),
            summary=summary,
            compacted_through_sequence=last_sequence,
        )
        compacted_context = replace(
            context,
            session=compacted_session,
            memory=compacted_memory,
            cache_hit=False,
        )
        after = estimate_tokens(_session_history_text(compacted_memory))
        result = SessionCompactionResult(
            dropped_turn_sequences=tuple(turn.sequence for turn in dropped),
            kept_turn_sequences=tuple(turn.sequence for turn in kept),
            tokens_before=before,
            tokens_after=after,
            summary_updated=True,
            boundary_id=boundary_id,
        )
        return compacted_context, result

    def _persist_context(
        self,
        context: SessionContext,
        *,
        repo_fingerprint: str,
        index_version: str,
    ) -> SessionContext:
        self._session_store.save_session(context.session)
        self._save_session_memory(
            context.memory,
            scope=context.session.scope,
            repo_id=context.session.repo_id,
        )
        key = self._session_cache_key(
            session_id=context.session.session_id,
            scope=context.session.scope,
            repo_id=context.session.repo_id,
            repo_fingerprint=repo_fingerprint,
            index_version=index_version,
        )
        cache_fallback = self._cache is None
        if self._cache is not None:
            try:
                self._cache.set(
                    key.value,
                    _context_to_json(context),
                    ttl_seconds=self._cache_ttl_seconds,
                )
            except CacheUnavailableError:
                cache_fallback = True
        return replace(context, cache_fallback=cache_fallback)

    def _save_session_memory(
        self,
        memory: SessionMemory,
        *,
        scope: TenantScope,
        repo_id: str,
    ) -> None:
        self._save_memory_record(
            layer=LAYER_SESSION,
            owner_id=memory.session_id,
            payload=_session_memory_to_payload(memory),
            scope=scope,
            repo_id=repo_id,
        )

    def _save_memory_record(
        self,
        *,
        layer: str,
        owner_id: str,
        payload: dict[str, object],
        scope: TenantScope,
        repo_id: str,
        source: str = SESSION_MEMORY_SOURCE,
        confidence: float | None = None,
        consent: bool = True,
        expires_at_epoch_ms: int | None = None,
    ) -> None:
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._memory_store.save(
            MemoryRecord(
                record_id=_record_id(layer, owner_id, scope=scope, repo_id=repo_id),
                layer=layer,
                owner_id=owner_id,
                content=content,
                token_estimate=estimate_tokens(content),
                is_trusted=False,
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                repo_id=repo_id,
                source=source,
                confidence=confidence,
                consent=consent,
                expires_at_epoch_ms=expires_at_epoch_ms,
            )
        )

    def _load_memory_record(
        self,
        *,
        layer: str,
        owner_id: str,
        scope: TenantScope,
        repo_id: str,
    ) -> MemoryRecord | None:
        record_id = _record_id(layer, owner_id, scope=scope, repo_id=repo_id)
        record = self._memory_store.get(
            layer,
            owner_id,
            record_id,
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            repo_id=repo_id,
        )
        if record is None or record.status != "active":
            return None
        if record.expires_at_epoch_ms is not None:
            now_epoch_ms = int(time.time() * 1000)
            if now_epoch_ms >= record.expires_at_epoch_ms:
                self._delete_memory_record(
                    layer=layer,
                    owner_id=owner_id,
                    scope=scope,
                    repo_id=repo_id,
                )
                return None
        return record

    def _delete_memory_record(
        self,
        *,
        layer: str,
        owner_id: str,
        scope: TenantScope,
        repo_id: str,
    ) -> None:
        self._memory_store.delete(
            layer,
            owner_id,
            _record_id(layer, owner_id, scope=scope, repo_id=repo_id),
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            repo_id=repo_id,
        )

    def _session_cache_key(
        self,
        *,
        session_id: str,
        scope: TenantScope,
        repo_id: str,
        repo_fingerprint: str,
        index_version: str,
    ) -> MemoryCacheKey:
        filter_hash = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()
        return MemoryCacheKey.for_query(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            session_id=session_id,
            repo_fingerprint=repo_fingerprint,
            index_version=index_version,
            query="session-context",
            top_k=1,
            filter_hash=filter_hash,
        )

    def _load_active_goal(self, session: ConversationSession) -> CodeGoal | None:
        if session.active_goal_id is None:
            return None
        goal = self._session_store.get_goal(session.active_goal_id)
        if goal is None or goal.status != GOAL_ACTIVE:
            return None
        return goal

    @staticmethod
    def _assert_session_scope(
        session: ConversationSession,
        *,
        scope: TenantScope,
        repo_id: str,
    ) -> None:
        if session.scope != scope or session.repo_id != repo_id:
            raise ValueError("session_id 已被其他 tenant/user/repo 使用")


def _record_id(layer: str, owner_id: str, *, scope: TenantScope, repo_id: str) -> str:
    raw = ":".join((layer, scope.tenant_id, scope.user_id, repo_id, owner_id))
    return f"{layer}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _session_history_text(memory: SessionMemory) -> str:
    parts: list[str] = []
    if memory.summary:
        parts.append(f"summary: {memory.summary}")
    if memory.confirmed_conclusions:
        parts.append("confirmed: " + " | ".join(memory.confirmed_conclusions))
    if memory.rejected_approaches:
        parts.append("rejected: " + " | ".join(memory.rejected_approaches))
    for turn in memory.recent_turns:
        parts.append(f"turn {turn.sequence} {turn.role}: {turn.content}")
    return "\n".join(parts)


def _enforce_summary_budget(
    summary: SessionCompactionSummary, max_tokens: int
) -> SessionCompactionSummary:
    """把渲染后的摘要压到 token 上限以内。

    先丢最旧的逐轮片段——它们是体积主体，也是信息密度最低的部分；仍然超限
    再截断上一段已定稿摘要。目标、已确认结论、失败和待办保留到最后，它们
    才是恢复对话必需的线索。没有这个上限时，previous_text 会随每次压缩累积，
    摘要本身最终撑成超预算的正文，压缩就白做了。
    """
    if max_tokens <= 0:
        raise ValueError("summary_max_tokens 必须为正")
    if estimate_tokens(summary.render()) <= max_tokens:
        return summary
    snippets = list(summary.turn_snippets)
    while snippets:
        snippets.pop(0)
        candidate = replace(summary, turn_snippets=tuple(snippets))
        if estimate_tokens(candidate.render()) <= max_tokens:
            return candidate
    trimmed = replace(summary, turn_snippets=())
    if estimate_tokens(trimmed.render()) <= max_tokens or not summary.previous_text:
        return trimmed
    without_previous = replace(trimmed, previous_text=None)
    remaining = max_tokens - estimate_tokens(without_previous.render())
    if remaining <= 0:
        return without_previous
    # estimate_tokens 是 utf8 字节 / 4；按剩余预算反推可保留的字符数，并从
    # 尾部保留——尾部是最近一次边界的内容，比更早的边界更值得留。
    keep = remaining * 4
    previous = summary.previous_text
    while keep > 0:
        candidate = replace(without_previous, previous_text=previous[-keep:])
        if estimate_tokens(candidate.render()) <= max_tokens:
            return candidate
        keep = int(keep * 0.8)
    return without_previous


# 重复次数必须带上限：无界的 [A-Za-z0-9_./\\-]+ 遇到「一整段没有点的长词
# 字符」时，会在每个起始位置一路回溯到结尾，退化成二次复杂度。压缩要扫描
# 被丢弃的全部正文，只要正文里出现一大段 base64、压缩 JSON 或长标识符，
# 一次压缩就会从毫秒级涨到几十秒。实测 40KB 的连续词字符这一处就要 30 秒。
_PATH_PATTERN = re.compile(
    r"[A-Za-z0-9_./\\-]{1,200}\.(?:py|ts|tsx|js|jsx|json|md|toml|ya?ml|sql|css|html)"
)
# 压缩摘要要扫描被丢弃的全部正文。给扫描量一个上界，使压缩成本与正文形状
# 无关：文件名和证据编号是恢复线索，不是证据本身，扫到前 200K 字符足够。
_SUMMARY_SCAN_CHARS = 200_000
_EVIDENCE_ID_PATTERN = re.compile(r"\bE[0-9]{1,4}\b")
_FAILURE_LINE_PATTERN = re.compile(
    r"^(?:FAILED|ERROR|AssertionError|Traceback|[A-Za-z_.]+Error:).*"
)


def _extract_in_order(pattern: re.Pattern[str], text: str, *, limit: int) -> list[str]:
    """按出现顺序去重取前 limit 个匹配，保持可复现。"""
    found: list[str] = []
    for match in pattern.finditer(text):
        value = match.group(0).strip()
        if value and value not in found:
            found.append(value)
        if len(found) >= limit:
            break
    return found


def _build_structured_summary(
    *,
    boundary_id: str,
    previous: SessionCompactionSummary | None,
    existing_summary: str | None,
    dropped: list[ConversationTurn],
    active_goal: str | None,
    confirmed_conclusions: tuple[str, ...],
    rejected_approaches: tuple[str, ...],
) -> SessionCompactionSummary:
    """把被丢弃的轮次压成一个确定性的结构化摘要。

    这里只用规则提取，不调用模型：压缩是恢复线索，不是结论。调用模型生成
    摘要会同时引入第二次成本和一个新的、无法审计的事实来源。
    """
    if not dropped:
        raise SessionCompactionError("没有可压缩的轮次")
    text = "\n".join(turn.content for turn in dropped)
    # 正则只扫有界长度；摘要里的文件名与证据编号是恢复线索，不是证据。
    scan_text = text[:_SUMMARY_SCAN_CHARS]
    failures: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _FAILURE_LINE_PATTERN.match(line) and line not in failures:
            failures.append(line[:160])
        if len(failures) >= 8:
            break
    questions: list[str] = []
    for turn in dropped:
        if turn.role != "user":
            continue
        stripped = " ".join(turn.content.split())
        if not stripped:
            continue
        if stripped.endswith(("?", "？")) and stripped not in questions:
            questions.append(stripped[:160])
        if len(questions) >= 6:
            break
    snippets: list[str] = []
    for turn in dropped:
        collapsed = " ".join(turn.content.split())
        snippets.append(f"第 {turn.sequence} 轮 {turn.role}：{collapsed[:120]}")
    return SessionCompactionSummary(
        boundary_id=boundary_id,
        source_sequence_range=(dropped[0].sequence, dropped[-1].sequence),
        previous_text=(
            previous.render() if previous is not None else existing_summary or None
        ),
        active_goal=active_goal,
        confirmed_decisions=confirmed_conclusions,
        rejected_approaches=rejected_approaches,
        important_files=tuple(
            _extract_in_order(_PATH_PATTERN, scan_text, limit=20)
        ),
        evidence_ids=tuple(
            _extract_in_order(_EVIDENCE_ID_PATTERN, scan_text, limit=40)
        ),
        test_failures=tuple(failures),
        pending_actions=(),
        open_questions=tuple(questions),
        turn_snippets=tuple(snippets),
    )


def _active_goal_text(goal: CodeGoal | None) -> str | None:
    """摘要里只放目标正文，不放 goal_id 这类内部标识。"""
    if goal is None:
        return None
    text = " ".join(goal.user_goal.split())
    return text[:200] if text else None


def _compaction_boundary_id(session_id: str, last_sequence: int, count: int) -> str:
    """确定性边界 ID：同一 Session 的同一段历史永远得到同一个 ID。"""
    return f"cmp-{session_id[:12]}-{last_sequence}-{count}"


def _structured_summary_from_payload(
    raw: object,
) -> SessionCompactionSummary | None:
    """从持久化载荷恢复结构化摘要；缺字段按版本默认值补齐。"""
    if not isinstance(raw, dict):
        return None
    sequence_range = raw.get("source_sequence_range")
    if not isinstance(sequence_range, (list, tuple)) or len(sequence_range) != 2:
        return None
    return SessionCompactionSummary(
        boundary_id=str(raw["boundary_id"]),
        source_sequence_range=(int(sequence_range[0]), int(sequence_range[1])),
        previous_text=(
            str(raw["previous_text"]) if raw.get("previous_text") is not None else None
        ),
        active_goal=(
            str(raw["active_goal"]) if raw.get("active_goal") is not None else None
        ),
        confirmed_decisions=tuple(
            str(value) for value in raw.get("confirmed_decisions", ())
        ),
        rejected_approaches=tuple(
            str(value) for value in raw.get("rejected_approaches", ())
        ),
        important_files=tuple(str(value) for value in raw.get("important_files", ())),
        evidence_ids=tuple(str(value) for value in raw.get("evidence_ids", ())),
        test_failures=tuple(str(value) for value in raw.get("test_failures", ())),
        pending_actions=tuple(str(value) for value in raw.get("pending_actions", ())),
        open_questions=tuple(str(value) for value in raw.get("open_questions", ())),
        turn_snippets=tuple(str(value) for value in raw.get("turn_snippets", ())),
        summary_version=str(
            raw.get("summary_version", SESSION_SUMMARY_VERSION)
        ),
    )


def _session_memory_to_payload(memory: SessionMemory) -> dict[str, object]:
    return {
        "session_id": memory.session_id,
        "recent_turns": [asdict(turn) for turn in memory.recent_turns],
        "active_goal_id": memory.active_goal_id,
        "confirmed_conclusions": list(memory.confirmed_conclusions),
        "rejected_approaches": list(memory.rejected_approaches),
        "user_preferences": memory.user_preferences,
        "summary": memory.summary,
        "compacted_through_sequence": memory.compacted_through_sequence,
        "compaction_count": memory.compaction_count,
        "compaction_boundary_id": memory.compaction_boundary_id,
        "summary_hash": memory.summary_hash,
        "structured_summary": (
            asdict(memory.structured_summary)
            if memory.structured_summary is not None
            else None
        ),
    }


def _session_memory_from_payload(payload: dict[str, object]) -> SessionMemory:
    raw_turns = payload.get("recent_turns", [])
    turns = tuple(
        ConversationTurn(
            sequence=int(turn["sequence"]),
            role=str(turn["role"]),
            content=str(turn["content"]),
        )
        for turn in raw_turns
        if isinstance(turn, dict)
    )
    raw_preferences = payload.get("user_preferences", {})
    preferences = {
        str(key): str(value)
        for key, value in raw_preferences.items()
    } if isinstance(raw_preferences, dict) else {}
    return SessionMemory(
        session_id=str(payload["session_id"]),
        recent_turns=turns,
        active_goal_id=(
            str(payload["active_goal_id"])
            if payload.get("active_goal_id") is not None
            else None
        ),
        confirmed_conclusions=tuple(
            str(value) for value in payload.get("confirmed_conclusions", ())
        ),
        rejected_approaches=tuple(
            str(value) for value in payload.get("rejected_approaches", ())
        ),
        user_preferences=preferences,
        summary=str(payload["summary"]) if payload.get("summary") is not None else None,
        compacted_through_sequence=int(payload.get("compacted_through_sequence", 0)),
        compaction_count=int(payload.get("compaction_count", 0)),
        compaction_boundary_id=(
            str(payload["compaction_boundary_id"])
            if payload.get("compaction_boundary_id") is not None
            else None
        ),
        summary_hash=(
            str(payload["summary_hash"])
            if payload.get("summary_hash") is not None
            else None
        ),
        structured_summary=_structured_summary_from_payload(
            payload.get("structured_summary")
        ),
    )


def _context_to_json(context: SessionContext) -> str:
    session = context.session
    goal = context.active_goal
    payload: dict[str, object] = {
        "session": {
            "session_id": session.session_id,
            "tenant_id": session.scope.tenant_id,
            "user_id": session.scope.user_id,
            "repo_id": session.repo_id,
            "summary": session.summary,
            "active_goal_id": session.active_goal_id,
            "compacted_through_sequence": session.compacted_through_sequence,
        },
        "memory": _session_memory_to_payload(context.memory),
        "active_goal": asdict(goal) if goal is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _context_from_json(raw: str) -> SessionContext:
    payload = json.loads(raw)
    session_payload = payload["session"]
    memory = _session_memory_from_payload(payload["memory"])
    scope = TenantScope(
        tenant_id=str(session_payload["tenant_id"]),
        user_id=str(session_payload["user_id"]),
    )
    session = ConversationSession(
        session_id=str(session_payload["session_id"]),
        scope=scope,
        repo_id=str(session_payload["repo_id"]),
        recent_turns=memory.recent_turns,
        summary=(
            str(session_payload["summary"])
            if session_payload.get("summary") is not None
            else None
        ),
        active_goal_id=(
            str(session_payload["active_goal_id"])
            if session_payload.get("active_goal_id") is not None
            else None
        ),
        compacted_through_sequence=int(
            session_payload.get(
                "compacted_through_sequence", memory.compacted_through_sequence
            )
        ),
    )
    goal_payload = payload.get("active_goal")
    active_goal = None
    if isinstance(goal_payload, dict):
        active_goal = CodeGoal(
            goal_id=str(goal_payload["goal_id"]),
            session_id=str(goal_payload["session_id"]),
            scope=TenantScope(
                tenant_id=str(goal_payload["scope"]["tenant_id"]),
                user_id=str(goal_payload["scope"]["user_id"]),
            ),
            repo_id=str(goal_payload["repo_id"]),
            task_type=str(goal_payload["task_type"]),
            user_goal=str(goal_payload["user_goal"]),
            mode=str(goal_payload["mode"]),
            target_scope=tuple(goal_payload.get("target_scope", ())),
            validation_profile=(
                str(goal_payload["validation_profile"])
                if goal_payload.get("validation_profile") is not None
                else None
            ),
            status=str(goal_payload["status"]),
        )
    return SessionContext(session=session, memory=memory, active_goal=active_goal)
