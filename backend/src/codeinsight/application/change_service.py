"""Step 6 代码变更闭环：preview、approval、隔离应用、检查和回滚。"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from codeinsight.application.quality_guard import (
    FileChange,
    apply_change_set,
    build_change_set,
    get_validation_profile,
    reconcile_change_set,
    validate_change_set_base,
)
from codeinsight.domain.change import (
    ChangeApproval,
    PatchArtifact,
    TestFailureDigest,
    ValidationRun,
)
from codeinsight.domain.trace import (
    APPROVAL_GRANTED,
    APPROVAL_REQUESTED,
    CHECKPOINT_CREATED,
    PATCH_APPLIED,
    RECONCILE_PERFORMED,
    ROLLBACK_PERFORMED,
    RUN_FINISHED,
    VALIDATION_FINISHED,
    VALIDATION_STARTED,
    VERDICT_APPLIED,
    AuditRecord,
    RunEvent,
)
from codeinsight.infrastructure.persistent_change_state import (
    JsonObjectStore,
    PersistentApprovalStore,
    PersistentAuditLog,
    PersistentEventLog,
)
from codeinsight.infrastructure.redaction import redact_sensitive
from codeinsight.infrastructure.run_store import (
    ApprovalAlreadyConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    InMemoryApprovalStore,
)
from codeinsight.infrastructure.runtime_policy import DevelopmentPolicy
from codeinsight.infrastructure.sandbox import (
    DockerSandbox,
    SandboxCleanupError,
    SandboxPreflightResult,
    SandboxResult,
    SandboxRunner,
)
from codeinsight.infrastructure.workspace import WorkspaceManager, fingerprint_artifact


class ChangeRequestError(ValueError):
    """用户请求或状态不满足变更门禁。"""


@dataclass(frozen=True)
class PreviewResult:
    run_id: str
    patch_id: str
    path: str
    diff: str
    diff_hash: str
    base_fingerprint: str
    validation_profile: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ChangeResult:
    status: str
    run_id: str
    patch_id: str
    diff: str
    diff_hash: str
    base_fingerprint: str
    workspace_id: str | None = None
    checkpoint_id: str | None = None
    validation: dict[str, object] | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "patch_id": self.patch_id,
            "diff": self.diff,
            "diff_hash": self.diff_hash,
            "base_fingerprint": self.base_fingerprint,
            "workspace_id": self.workspace_id,
            "checkpoint_id": self.checkpoint_id,
            "validation": self.validation,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _Proposal:
    preview: PreviewResult
    artifact: PatchArtifact
    changes: tuple[FileChange, ...]
    base_fingerprints: dict[str, str]
    repository_root: str


class Repairer(Protocol):
    def __call__(
        self, digest: TestFailureDigest, attempt: int
    ) -> Mapping[str, str | None] | None: ...


class ChangeService:
    """单进程本地持久变更服务；源仓库只读，不支持多进程共享状态目录。"""

    def __init__(
        self,
        *,
        workspace_manager: WorkspaceManager | None = None,
        sandbox: SandboxRunner | None = None,
        approval_store: InMemoryApprovalStore | None = None,
        state_dir: str | Path | None = None,
        event_sink: Callable[[RunEvent], None] | None = None,
        development_policy: DevelopmentPolicy | None = None,
    ) -> None:
        self.workspaces = workspace_manager or WorkspaceManager()
        self.sandbox = sandbox or DockerSandbox()
        root = Path(state_dir or (self.workspaces.base_dir / ".change-state"))
        self.state = JsonObjectStore(root)
        self.approvals = approval_store or PersistentApprovalStore(root)
        self.events = PersistentEventLog(root)
        self.audit = PersistentAuditLog(root)
        self.development_policy = development_policy or DevelopmentPolicy.from_environment()
        self._proposals: dict[tuple[str, str], _Proposal] = {}
        self._results: dict[str, ChangeResult] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._event_sinks: list[Callable[[RunEvent], None]] = []
        if event_sink is not None:
            self._event_sinks.append(event_sink)
        self._load_state()

    def add_event_sink(self, sink: Callable[[RunEvent], None]) -> None:
        """增加非持久化的实时镜像；镜像失败不能影响变更事实写入。"""
        self._event_sinks.append(sink)

    def preview(
        self,
        repository_root: str | Path,
        *,
        run_id: str | None,
        path: str,
        new_content: str,
        validation_profile: str,
    ) -> PreviewResult:
        return self.preview_many(
            repository_root,
            run_id=run_id,
            changes={path: new_content},
            validation_profile=validation_profile,
        )

    def preview_many(
        self,
        repository_root: str | Path,
        *,
        run_id: str | None,
        changes: Mapping[str, str | None],
        validation_profile: str = "python_compile",
    ) -> PreviewResult:
        with self._lock:
            return self._preview_many(
                repository_root, run_id=run_id, changes=changes,
                validation_profile=validation_profile,
            )

    def _preview_many(
        self,
        repository_root: str | Path,
        *,
        run_id: str | None,
        changes: Mapping[str, str | None],
        validation_profile: str = "python_compile",
    ) -> PreviewResult:
        get_validation_profile(validation_profile)
        resolved_run_id = run_id or f"run-{uuid4().hex[:16]}"
        self._require_known_run(resolved_run_id)
        managed = self.workspaces.get(resolved_run_id)
        source_root = Path(repository_root).resolve()
        if managed is not None:
            if str(source_root) != managed.run.source_repo_path:
                if str(source_root) != managed.run.workspace_path:
                    raise ChangeRequestError("同一 Run 不能绑定不同的源仓库")
                source_root = Path(managed.run.source_repo_path)
            repository_root = managed.run.workspace_path
        patch_id = f"patch-{uuid4().hex[:16]}"
        artifact, normalized, bases = build_change_set(
            repository_root,
            run_id=resolved_run_id,
            patch_id=patch_id,
            changes=changes,
        )
        preview = PreviewResult(
            resolved_run_id,
            patch_id,
            normalized[0].path,
            artifact.diff_text,
            artifact.diff_hash,
            artifact.base_fingerprint,
            validation_profile,
        )
        proposal = _Proposal(
            preview,
            artifact,
            normalized,
            bases,
            str(source_root),
        )
        self._proposals[(resolved_run_id, patch_id)] = proposal
        self._persist_proposal(proposal)
        self._emit(
            resolved_run_id,
            "patch_generated",
            {"patch_id": patch_id, "file_count": str(len(normalized))},
        )
        return preview

    def preflight_validation(self, profile: str) -> SandboxPreflightResult:
        """在模型生成补丁前确认固定校验环境可用。"""
        get_validation_profile(profile)
        if self.development_policy.skip_sandbox_validation:
            return SandboxPreflightResult(
                profile,
                True,
                "DEV_MODE_VALIDATION_SKIPPED",
                "开发模式已显式跳过 Docker 固定校验",
            )
        checker = getattr(self.sandbox, "preflight", None)
        if not callable(checker):
            # 测试替身和外部注入的 Sandbox 仍由其 run() 契约负责校验。
            return SandboxPreflightResult(profile, True)
        result = checker(profile)
        if not isinstance(result, SandboxPreflightResult):
            raise TypeError("Sandbox preflight 必须返回 SandboxPreflightResult")
        return result

    def approve(
        self,
        run_id: str,
        patch_id: str,
        *,
        expires_in_seconds: int = 300,
        actor: str = "user",
        source: str = "user",
    ) -> str:
        with self._lock:
            return self._approve(
                run_id,
                patch_id,
                expires_in_seconds=expires_in_seconds,
                actor=actor,
                source=source,
            )

    def _approve(
        self,
        run_id: str,
        patch_id: str,
        *,
        expires_in_seconds: int,
        actor: str,
        source: str,
    ) -> str:
        proposal = self._proposal(run_id, patch_id)
        validate_change_set_base(
            self._proposal_base(proposal), proposal.artifact, proposal.base_fingerprints
        )
        if not 30 <= expires_in_seconds <= 3600:
            raise ChangeRequestError("审批有效期必须在 30 到 3600 秒之间")
        token = secrets.token_urlsafe(24)
        approval = ChangeApproval(
            token=token,
            run_id=run_id,
            diff_hash=proposal.artifact.diff_hash,
            base_fingerprint=proposal.artifact.base_fingerprint,
            scope=proposal.artifact.touched_files,
            expires_at_epoch_ms=_now_ms() + expires_in_seconds * 1000,
        )
        self.approvals.issue(approval)
        self._emit(run_id, APPROVAL_REQUESTED, {"patch_id": patch_id, "source": source})
        self._emit(
            run_id,
            APPROVAL_GRANTED,
            {
                "patch_id": patch_id,
                "scope_count": str(len(proposal.artifact.touched_files)),
                "source": source,
            },
        )
        self.audit.record(
            AuditRecord(
                audit_id=f"audit-{uuid4().hex[:16]}",
                run_id=run_id,
                event_type=APPROVAL_GRANTED,
                actor=actor,
                occurred_at_epoch_ms=_now_ms(),
                subject=patch_id,
                outcome="granted",
                details={
                    "scope_count": str(len(proposal.artifact.touched_files)),
                    "source": source,
                },
            )
        )
        return token

    def apply(
        self,
        run_id: str,
        patch_id: str,
        approval_token: str,
        *,
        defer_validation: bool = False,
    ) -> ChangeResult:
        """应用已批准的补丁。

        ``defer_validation=True`` 时只做到「补丁已应用并核对」：不碰 Sandbox，
        结果停在 WAITING_VALIDATION，等独立的 ValidationWorker 拿着登记过的
        固定 profile 去跑。默认仍然是同步跑完校验，因为开发模式或本地直连的
        调用方没有第二个 Worker 可等。
        """

        with self._lock:
            return self._apply(
                run_id, patch_id, approval_token, defer_validation=defer_validation
            )

    def _apply(
        self,
        run_id: str,
        patch_id: str,
        approval_token: str,
        *,
        defer_validation: bool = False,
    ) -> ChangeResult:
        proposal = self._proposal(run_id, patch_id)
        if run_id in self._cancelled:
            raise ChangeRequestError("Run 已收到取消请求，未开始应用")
        previous = self._results.get(f"{run_id}:{patch_id}")
        if previous is not None:
            return previous
        self._require_validation_environment(proposal.preview.validation_profile)
        validate_change_set_base(
            self._proposal_base(proposal), proposal.artifact, proposal.base_fingerprints
        )
        approval = self.approvals.get(approval_token)
        if approval is None:
            raise ChangeRequestError("审批令牌不存在")
        if (
            approval.run_id != run_id
            or approval.diff_hash != proposal.artifact.diff_hash
            or approval.base_fingerprint != proposal.artifact.base_fingerprint
            or set(proposal.artifact.touched_files) - set(approval.scope)
        ):
            raise ChangeRequestError("审批令牌与当前 Run、diff、基线或范围不匹配")
        # 在消耗审批和写工作区之前落盘，重启后才能识别中断的执行。
        self._store_result(self._result(proposal, None, None, "RUNNING", None))
        managed = None
        checkpoint = None
        try:
            try:
                self.approvals.consume(approval_token, now_epoch_ms=_now_ms())
            except (
                ApprovalNotFoundError, ApprovalAlreadyConsumedError, ApprovalExpiredError
            ) as error:
                raise ChangeRequestError(str(error)) from error
            managed = self.workspaces.create(proposal.repository_root, run_id)
            checkpoint = self.workspaces.create_checkpoint(run_id)
            self._store_result(self._result(
                proposal, managed.run.workspace_id, checkpoint.checkpoint_id, "RUNNING", None
            ))
            self._emit(run_id, CHECKPOINT_CREATED, {"checkpoint_id": checkpoint.checkpoint_id})
            apply_change_set(
                managed.run.workspace_path,
                proposal.artifact,
                proposal.changes,
                proposal.base_fingerprints,
            )
            self._emit(run_id, PATCH_APPLIED, {"patch_id": patch_id})
            reconciliation = reconcile_change_set(
                managed.run.workspace_path,
                proposal.artifact,
                proposal.changes,
                proposal.base_fingerprints,
                run_id=run_id,
            )
            self._emit(run_id, RECONCILE_PERFORMED, {"verdict": reconciliation.verdict})
            if reconciliation.verdict != VERDICT_APPLIED:
                self.workspaces.rollback(run_id, checkpoint.checkpoint_id)
                result = self._result(
                    proposal,
                    managed.run.workspace_id,
                    checkpoint.checkpoint_id,
                    "ROLLED_BACK",
                    reconciliation.evidence,
                )
                self._store_result(result)
                return result
            if defer_validation:
                # 校验交给 ValidationWorker：改到这里的补丁已经落盘并核对过，
                # 接下来该由谁跑沙箱、跑哪个 profile 是登记好的事实，不是这次
                # 调用能顺手决定的事情。
                result = self._result(
                    proposal,
                    managed.run.workspace_id,
                    checkpoint.checkpoint_id,
                    "WAITING_VALIDATION",
                    "补丁已应用并通过核对，等待固定校验环境执行。",
                )
                self._store_result(result)
                return result
            self._emit(
                run_id,
                VALIDATION_STARTED,
                {
                    "profile": proposal.preview.validation_profile,
                    "mode": (
                        "development_skipped"
                        if self.development_policy.skip_sandbox_validation
                        else "docker"
                    ),
                },
            )
            expected_tree = fingerprint_artifact(managed.run.workspace_path)
            if self.development_policy.skip_sandbox_validation:
                validation = self._skipped_validation_payload(
                    proposal.preview.validation_profile
                )
                self._emit(
                    run_id,
                    VALIDATION_FINISHED,
                    {
                        "profile": proposal.preview.validation_profile,
                        "passed": "skipped",
                        "skipped": "true",
                    },
                )
            else:
                checked = self.sandbox.run(
                    proposal.preview.validation_profile, managed.run.workspace_path
                )
                self._emit(
                    run_id,
                    VALIDATION_FINISHED,
                    {
                        "profile": proposal.preview.validation_profile,
                        "passed": str(checked.passed).lower(),
                    },
                )
                validation = self._validation_payload(checked, run_id)
            if fingerprint_artifact(managed.run.workspace_path) != expected_tree:
                raise ChangeRequestError("检查程序修改了批准范围之外或批准后的产物")
            self._store_validation(run_id, validation)
            if validation.get("skipped"):
                result = self._result(
                    proposal,
                    managed.run.workspace_id,
                    checkpoint.checkpoint_id,
                    "COMPLETED",
                    "开发模式已应用修改，但跳过了 Docker 固定校验；上线前必须关闭开发模式。",
                    validation,
                )
            elif not bool(validation["passed"]):
                result = self._result(
                    proposal,
                    managed.run.workspace_id,
                    checkpoint.checkpoint_id,
                    "REVIEW_REQUIRED",
                    "固定检查失败，等待有限修复或人工处理",
                    validation,
                )
            else:
                # 应用后已核对 diff；全树指纹相等说明检查没有改变该产物，无需重复核对。
                result = self._result(
                    proposal,
                    managed.run.workspace_id,
                    checkpoint.checkpoint_id,
                    "COMPLETED",
                    None,
                    validation,
                )
            self._store_result(result)
            self._emit(run_id, RUN_FINISHED, {"status": result.status})
            return result
        except Exception as error:
            status = "FAILED"
            reason = f"执行异常：{type(error).__name__}；请重新 preview 和审批"
            if isinstance(error, SandboxCleanupError):
                status = "UNKNOWN"
                reason = "无法确认 Sandbox 已停止；禁止继续写入，需人工核对容器和 workspace"
            elif checkpoint is not None:
                try:
                    self.workspaces.rollback(run_id, checkpoint.checkpoint_id)
                    status = "ROLLED_BACK"
                    reason = f"执行异常，已回滚：{type(error).__name__}"
                except (OSError, RuntimeError):
                    status = "UNKNOWN"
                    reason = "执行异常且回滚未确认完成，需要人工核对 workspace"
            result = self._result(
                proposal,
                managed.run.workspace_id if managed else None,
                checkpoint.checkpoint_id if checkpoint else None,
                status,
                reason,
            )
            self._store_result(result)
            self._emit(run_id, RUN_FINISHED, {"status": result.status})
            raise

    def retry_validation(
        self,
        run_id: str,
        patch_id: str,
        *,
        event_run_id: str | None = None,
    ) -> ChangeResult:
        """重新校验已应用但因基础设施失败而进入 REVIEW_REQUIRED 的补丁。

        该路径不重新生成补丁，也不重复消费审批令牌；它先确认隔离 workspace
        仍与已批准内容一致，再重新执行同一个固定 profile。这样 Docker 短暂不可用
        时，用户输入“继续”不会触发重复补丁或再次写入。
        """

        with self._lock:
            return self._validate_applied_patch(
                run_id,
                patch_id,
                event_run_id=event_run_id,
                required_status="REVIEW_REQUIRED",
                retry=True,
            )

    def run_registered_validation(
        self,
        run_id: str,
        patch_id: str,
        *,
        event_run_id: str | None = None,
    ) -> ChangeResult:
        """ValidationWorker 的入口：跑登记过的固定校验，并把结果写成事实。

        它只能在 WAITING_VALIDATION 上启动：apply 已经发生这件事不会被再执行一次；
        profile 取自补丁登记时固定的那一个，调用方无法临时改。
        """

        with self._lock:
            return self._validate_applied_patch(
                run_id,
                patch_id,
                event_run_id=event_run_id,
                required_status="WAITING_VALIDATION",
                retry=False,
            )

    def _validate_applied_patch(
        self,
        run_id: str,
        patch_id: str,
        *,
        event_run_id: str | None,
        required_status: str,
        retry: bool,
    ) -> ChangeResult:
        proposal = self._proposal(run_id, patch_id)
        previous = self._results.get(f"{run_id}:{patch_id}")
        if previous is None or previous.status != required_status:
            raise ChangeRequestError(
                f"当前补丁没有停在 {required_status} 的结果，不能执行固定校验"
            )
        retry_flag = {"retry": "true"} if retry else {}
        managed = self.workspaces.get(run_id)
        if managed is None or managed.run.latest_checkpoint is None:
            raise ChangeRequestError("当前 Run 没有可重新校验的隔离 workspace")

        self._require_validation_environment(proposal.preview.validation_profile)
        event_id = event_run_id or run_id
        reconciliation = reconcile_change_set(
            managed.run.workspace_path,
            proposal.artifact,
            proposal.changes,
            proposal.base_fingerprints,
            run_id=event_id,
        )
        self._emit(event_id, RECONCILE_PERFORMED, {"verdict": reconciliation.verdict})
        if reconciliation.verdict != VERDICT_APPLIED:
            raise ChangeRequestError(
                "当前隔离 workspace 已不再匹配已批准补丁，不能直接重新校验；"
                "请重新生成并审批新的补丁。"
            )

        self._emit(
            event_id,
            VALIDATION_STARTED,
            {
                "profile": proposal.preview.validation_profile,
                **retry_flag,
                "mode": (
                    "development_skipped"
                    if self.development_policy.skip_sandbox_validation
                    else "docker"
                ),
            },
        )
        if self.development_policy.skip_sandbox_validation:
            validation = self._skipped_validation_payload(
                proposal.preview.validation_profile
            )
            self._emit(
                event_id,
                VALIDATION_FINISHED,
                {
                    "profile": proposal.preview.validation_profile,
                    "passed": "skipped",
                    "skipped": "true",
                    **retry_flag,
                },
            )
        else:
            checked = self.sandbox.run(
                proposal.preview.validation_profile, managed.run.workspace_path
            )
            self._emit(
                event_id,
                VALIDATION_FINISHED,
                {
                    "profile": proposal.preview.validation_profile,
                    "passed": str(checked.passed).lower(),
                    **retry_flag,
                },
            )
            validation = self._validation_payload(checked, run_id)
        self._store_validation(run_id, validation)
        if validation.get("skipped") or bool(validation["passed"]):
            result = self._result(
                proposal,
                managed.run.workspace_id,
                managed.run.latest_checkpoint.checkpoint_id,
                "COMPLETED",
                (
                    "开发模式已完成修改，但跳过了 Docker 固定校验；上线前必须关闭开发模式。"
                    if validation.get("skipped")
                    else None
                ),
                validation,
            )
        else:
            result = self._result(
                proposal,
                managed.run.workspace_id,
                managed.run.latest_checkpoint.checkpoint_id,
                "REVIEW_REQUIRED",
                "固定检查失败，等待有限修复或人工处理",
                validation,
            )
        self._store_result(result)
        self._emit(event_id, RUN_FINISHED, {"status": result.status, **retry_flag})
        return result

    def apply_with_repairs(
        self,
        run_id: str,
        patch_id: str,
        approval_token: str,
        *,
        repairer: Repairer,
        approve_repair: Callable[[PreviewResult], str],
        max_repairs: int = 2,
    ) -> ChangeResult:
        """每个修复 diff 都重新 preview 和审批，最多两次，绝不复用旧 token。"""
        if not 0 <= max_repairs <= 2:
            raise ValueError("自动修复次数必须在 0 到 2 之间")
        result = self.apply(run_id, patch_id, approval_token)
        for attempt in range(1, max_repairs + 1):
            if result.status != "REVIEW_REQUIRED" or not result.validation:
                return result
            digest = TestFailureDigest(
                digest_id=str(result.validation["failure_digest_id"]),
                run_id=run_id,
                failed_test_ids=(str(result.validation["profile"]),),
                error_class=str(result.validation.get("error_class", "CHECK_FAILED")),
                message_excerpt=str(result.validation.get("message_excerpt", ""))[:2000],
            )
            repaired = repairer(digest, attempt)
            if not repaired:
                return result
            original = self._proposal(run_id, patch_id)
            managed = self.workspaces.get(run_id)
            if managed is None:
                raise ChangeRequestError("Run 的隔离 workspace 不存在")
            preview = self.preview_many(
                managed.run.workspace_path,
                run_id=run_id,
                changes=repaired,
                validation_profile=original.preview.validation_profile,
            )
            token = approve_repair(preview)
            self.state.put(
                "repair-attempts",
                f"{run_id}:{attempt}",
                {"run_id": run_id, "attempt": attempt, "patch_id": preview.patch_id},
            )
            result = self.apply(run_id, preview.patch_id, token)
            patch_id = preview.patch_id
        return result

    def rollback(self, run_id: str, patch_id: str) -> ChangeResult:
        with self._lock:
            return self._rollback(run_id, patch_id)

    def _rollback(self, run_id: str, patch_id: str) -> ChangeResult:
        proposal = self._proposal(run_id, patch_id)
        managed = self.workspaces.get(run_id)
        previous = self._results.get(f"{run_id}:{patch_id}")
        checkpoint_id = previous.checkpoint_id if previous else None
        if managed is None or checkpoint_id is None:
            raise ChangeRequestError("当前 Run 没有可回滚的 checkpoint")
        self._require_known_run(run_id)
        if previous.status == "ROLLED_BACK":
            return previous
        checkpoints = [item.checkpoint_id for item in managed.run.checkpoints]
        affected = checkpoints[checkpoints.index(checkpoint_id):]
        # 恢复早期 checkpoint 会使后续结果一起失效，先记录进行中状态。
        for old in tuple(self._results.values()):
            if old.run_id == run_id and old.checkpoint_id in affected:
                self._store_result(replace(old, status="RUNNING", reason="正在回滚 checkpoint"))
        try:
            self.workspaces.rollback(run_id, checkpoint_id)
        except (OSError, RuntimeError):
            for old in tuple(self._results.values()):
                if old.run_id == run_id and old.checkpoint_id in affected:
                    self._store_result(replace(old, status="UNKNOWN", reason="回滚未确认完成"))
            raise
        for old in tuple(self._results.values()):
            if old.run_id == run_id and old.checkpoint_id in affected:
                self._store_result(
                    replace(old, status="ROLLED_BACK", reason="已恢复更早的 checkpoint")
                )
        result = self._result(
            proposal,
            managed.run.workspace_id,
            checkpoint_id,
            "ROLLED_BACK",
            "已恢复到 checkpoint",
        )
        self._store_result(result)
        self._emit(run_id, ROLLBACK_PERFORMED, {"checkpoint_id": checkpoint_id})
        self.audit.record(
            AuditRecord(
                audit_id=f"audit-{uuid4().hex[:16]}",
                run_id=run_id,
                event_type=ROLLBACK_PERFORMED,
                actor="system",
                occurred_at_epoch_ms=_now_ms(),
                subject=patch_id,
                outcome="rolled_back",
                details={"checkpoint_id": checkpoint_id},
            )
        )
        return result

    def cancel(self, run_id: str) -> tuple[str, bool]:
        """请求取消；正在执行的不可中断写操作不会被粗暴杀死。"""
        with self._lock:
            if any(key.startswith(f"{run_id}:") for key in self._results):
                return "already_finished", False
            if not any(key[0] == run_id for key in self._proposals):
                raise ChangeRequestError("Run 不存在")
            self._cancelled.add(run_id)
            self.state.put("cancelled", run_id, {"run_id": run_id, "cancelled": True})
            self._emit(run_id, "cancel_requested", {"reason": "user_request"})
            return "cancel_requested", True

    def events_after(self, run_id: str, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        return self.events.read_events(run_id, after_sequence=after_sequence)

    def last_result(self, run_id: str) -> ChangeResult | None:
        """这个 Run 最近写入的一条结果事实。

        校验结论、失败摘要和「需要人决定」都在结果里；续跑与收尾只读它，
        不去问「哪一次调用还记得什么」。
        """

        for result in reversed(list(self._results.values())):
            if result.run_id == run_id:
                return result
        return None

    def last_review_required(self, run_id: str) -> ChangeResult | None:
        """最近一次「校验没通过、等人决定」的结果事实。"""

        for result in reversed(list(self._results.values())):
            if result.run_id == run_id and result.status == "REVIEW_REQUIRED":
                return result
        return None

    def repair_rounds(self, run_id: str) -> int:
        """这个 Run 已经占用过几次自动修复预算（只数已落盘的事实）。"""

        return sum(
            1
            for raw in self.state.list("repair-attempts")
            if str(raw.get("run_id")) == run_id
        )

    def record_repair_round(self, run_id: str, *, patch_id: str | None = None) -> int:
        """先记账再动手：修复预算必须在真的开始新一轮之前落盘。

        反过来的顺序（跑完再记）在崩溃时会让同一个 Run 反复重修，
        而「最多修几次」正是要挡住的自动循环。
        """

        attempt = self.repair_rounds(run_id) + 1
        self.state.put(
            "repair-attempts",
            f"{run_id}:{attempt}",
            {
                "run_id": run_id,
                "attempt": attempt,
                "patch_id": patch_id or "",
                "reserved_by": "agent-run-worker",
            },
        )
        return attempt

    def registered_validation(self, run_id: str, patch_id: str) -> tuple[str, str]:
        """这条补丁登记过的固定校验：profile 与要校验的隔离 workspace 路径。

        两者都来自登记时的事实：模型不能临时决定跑什么命令，Worker 也不能
        自己挑一个目录。
        """

        proposal = self._proposal(run_id, patch_id)
        managed = self.workspaces.get(run_id)
        if managed is None:
            raise ChangeRequestError("当前 Run 没有可校验的隔离 workspace")
        return proposal.preview.validation_profile, managed.run.workspace_path
    def get_result(self, run_id: str, patch_id: str) -> ChangeResult | None:
        self._proposal(run_id, patch_id)
        return self._results.get(f"{run_id}:{patch_id}")

    def _require_known_run(self, run_id: str) -> None:
        if any(
            item.run_id == run_id and item.status == "UNKNOWN"
            for item in self._results.values()
        ):
            raise ChangeRequestError("Run 有未确认状态，必须人工核对后使用新 Run")

    def _proposal_base(self, proposal: _Proposal) -> str:
        self._require_known_run(proposal.preview.run_id)
        managed = self.workspaces.get(proposal.preview.run_id)
        return managed.run.workspace_path if managed else proposal.repository_root

    def _proposal(self, run_id: str, patch_id: str) -> _Proposal:
        try:
            return self._proposals[(run_id, patch_id)]
        except KeyError as error:
            raise ChangeRequestError("Patch 提案不存在或不属于当前 Run") from error

    def _require_validation_environment(self, profile: str) -> None:
        preflight = self.preflight_validation(profile)
        if not preflight.available:
            detail = preflight.message_excerpt or "Sandbox 不可用"
            raise ChangeRequestError(
                "固定校验环境不可用："
                f"{preflight.error_class or 'SANDBOX_UNAVAILABLE'}；{detail}"
            )

    def _persist_proposal(self, proposal: _Proposal) -> None:
        self.state.put(
            "proposals",
            f"{proposal.preview.run_id}:{proposal.preview.patch_id}",
            {
                "preview": proposal.preview.as_dict(),
                "artifact": {
                    "patch_id": proposal.artifact.patch_id,
                    "goal_id": proposal.artifact.goal_id,
                    "diff_text": proposal.artifact.diff_text,
                    "diff_hash": proposal.artifact.diff_hash,
                    "base_fingerprint": proposal.artifact.base_fingerprint,
                    "touched_files": list(proposal.artifact.touched_files),
                },
                "changes": [
                    {"path": item.path, "new_content": item.new_content}
                    for item in proposal.changes
                ],
                "base_fingerprints": proposal.base_fingerprints,
                "repository_root": proposal.repository_root,
            },
        )

    def _store_result(self, result: ChangeResult) -> None:
        self.state.put("results", f"{result.run_id}:{result.patch_id}", result.as_dict())
        self._results[f"{result.run_id}:{result.patch_id}"] = result

    def _store_validation(self, run_id: str, payload: dict[str, object]) -> None:
        skipped = bool(payload.get("skipped", False))
        passed = bool(payload.get("passed", False))
        digest = None
        if not passed and not skipped:
            digest = TestFailureDigest(
                str(payload["failure_digest_id"]),
                run_id,
                (str(payload["profile"]),),
                str(payload["error_class"]),
                str(payload["message_excerpt"]),
            )
        validation = ValidationRun(
            f"validation-{uuid4().hex[:16]}",
            run_id,
            str(payload["profile"]),
            tuple(str(item) for item in payload["commands"]),
            passed,
            digest,
            skipped,
        )
        self.state.put(
            "validations",
            validation.validation_id,
            {
                "validation_id": validation.validation_id,
                "run_id": validation.run_id,
                "profile": validation.profile,
                "commands": list(validation.commands),
                "passed": validation.passed,
                "skipped": validation.skipped,
                "digest": (
                    {
                        "digest_id": digest.digest_id,
                        "run_id": digest.run_id,
                        "failed_test_ids": list(digest.failed_test_ids),
                        "error_class": digest.error_class,
                        "message_excerpt": digest.message_excerpt,
                    }
                    if digest
                    else None
                ),
            },
        )

    def _load_state(self) -> None:
        for raw in self.state.list("proposals"):
            preview = PreviewResult(**raw["preview"])
            artifact_raw = raw["artifact"]
            artifact_raw["touched_files"] = tuple(artifact_raw["touched_files"])
            artifact = PatchArtifact(**artifact_raw)
            changes = tuple(FileChange(**item) for item in raw["changes"])
            proposal = _Proposal(
                preview,
                artifact,
                changes,
                dict(raw["base_fingerprints"]),
                raw["repository_root"],
            )
            self._proposals[(preview.run_id, preview.patch_id)] = proposal
        for raw in self.state.list("results"):
            result = ChangeResult(**raw)
            if result.status == "RUNNING":
                result = replace(result, status="UNKNOWN", reason="上次执行中断，禁止自动重放")
                self.state.put("results", f"{result.run_id}:{result.patch_id}", result.as_dict())
            self._results[f"{result.run_id}:{result.patch_id}"] = result
        for raw in self.state.list("cancelled"):
            if raw.get("cancelled"):
                self._cancelled.add(str(raw["run_id"]))

    def _result(
        self,
        proposal: _Proposal,
        workspace_id: str | None,
        checkpoint_id: str | None,
        status: str,
        reason: str | None,
        validation: dict[str, object] | None = None,
    ) -> ChangeResult:
        return ChangeResult(
            status,
            proposal.preview.run_id,
            proposal.preview.patch_id,
            proposal.preview.diff,
            proposal.preview.diff_hash,
            proposal.preview.base_fingerprint,
            workspace_id,
            checkpoint_id,
            validation,
            reason,
        )

    @staticmethod
    def _validation_payload(result: SandboxResult, run_id: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "profile": result.profile,
            "commands": list(result.commands),
            "passed": result.passed,
            "skipped": False,
        }
        if not result.passed:
            digest = TestFailureDigest(
                digest_id=f"digest-{uuid4().hex[:16]}",
                run_id=run_id,
                failed_test_ids=(result.profile,),
                error_class=result.error_class or "CHECK_FAILED",
                message_excerpt=redact_sensitive(result.message_excerpt)[:2000],
            )
            payload.update(
                {
                    "error_class": digest.error_class,
                    "message_excerpt": digest.message_excerpt,
                    "failure_digest_id": digest.digest_id,
                }
            )
        return payload

    def _skipped_validation_payload(self, profile: str) -> dict[str, object]:
        return {
            "profile": profile,
            "commands": [],
            "passed": None,
            "skipped": True,
            "reason": "CODEINSIGHT_DEV_SKIP_SANDBOX_VALIDATION",
            "development_mode": self.development_policy.as_dict(),
        }

    def _emit(self, run_id: str, event_type: str, payload: dict[str, str]) -> None:
        event = RunEvent(
            event_id=f"event-{uuid4().hex[:16]}",
            run_id=run_id,
            sequence=self.events.next_sequence(run_id),
            event_type=event_type,
            occurred_at_epoch_ms=_now_ms(),
            payload=payload,
        )
        self.events.append(event)
        for sink in tuple(self._event_sinks):
            try:
                sink(event)
            except Exception:
                # 实时 UI 是旁路能力，不能让它的连接故障破坏变更事实。
                pass
        if event_type == PATCH_APPLIED:
            self.audit.record(
                AuditRecord(
                    audit_id=f"audit-{uuid4().hex[:16]}",
                    run_id=run_id,
                    event_type=PATCH_APPLIED,
                    actor="system",
                    occurred_at_epoch_ms=_now_ms(),
                    subject=payload.get("patch_id", "unknown"),
                    outcome="applied",
                    details={"verified": "true"},
                )
            )


def _now_ms() -> int:
    return int(time.time() * 1000)
