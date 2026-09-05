"""Step 6 代码变更闭环：preview、approval、隔离应用、检查和回滚。"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    VERDICT_APPLIED,
    AuditRecord,
    IdempotencyKey,
    RunEvent,
)
from codeinsight.infrastructure.persistent_change_state import (
    JsonObjectStore,
    PersistentApprovalStore,
    PersistentAuditLog,
    PersistentEventLog,
    PersistentIdempotencyStore,
)
from codeinsight.infrastructure.redaction import redact_sensitive
from codeinsight.infrastructure.run_store import (
    ApprovalAlreadyConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    InMemoryApprovalStore,
)
from codeinsight.infrastructure.sandbox import DockerSandbox, SandboxResult, SandboxRunner
from codeinsight.infrastructure.workspace import WorkspaceManager


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
    """默认使用内存状态，便于本地演示；业务事实不写回源仓库。"""

    def __init__(
        self,
        *,
        workspace_manager: WorkspaceManager | None = None,
        sandbox: SandboxRunner | None = None,
        approval_store: InMemoryApprovalStore | None = None,
        state_dir: str | Path | None = None,
    ) -> None:
        self.workspaces = workspace_manager or WorkspaceManager()
        self.sandbox = sandbox or DockerSandbox()
        root = Path(state_dir or (self.workspaces.base_dir / ".change-state"))
        self.state = JsonObjectStore(root)
        self.approvals = approval_store or PersistentApprovalStore(root)
        self.idempotency = PersistentIdempotencyStore(root)
        self.events = PersistentEventLog(root)
        self.audit = PersistentAuditLog(root)
        self._proposals: dict[tuple[str, str], _Proposal] = {}
        self._results: dict[str, ChangeResult] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._load_state()

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
        selected = get_validation_profile(validation_profile)
        del selected  # 只在这里确认 profile 已登记；执行时再次解析。
        resolved_run_id = run_id or f"run-{uuid4().hex[:16]}"
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
            str(Path(repository_root).resolve()),
        )
        self._proposals[(resolved_run_id, patch_id)] = proposal
        self._persist_proposal(proposal)
        self._emit(
            resolved_run_id,
            "patch_generated",
            {"patch_id": patch_id, "file_count": str(len(normalized))},
        )
        return preview

    def approve(self, run_id: str, patch_id: str, *, expires_in_seconds: int = 300) -> str:
        proposal = self._proposal(run_id, patch_id)
        validate_change_set_base(
            proposal.repository_root, proposal.artifact, proposal.base_fingerprints
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
        self._emit(run_id, APPROVAL_REQUESTED, {"patch_id": patch_id})
        self._emit(
            run_id,
            APPROVAL_GRANTED,
            {"patch_id": patch_id, "scope_count": str(len(proposal.artifact.touched_files))},
        )
        self.audit.record(
            AuditRecord(
                audit_id=f"audit-{uuid4().hex[:16]}",
                run_id=run_id,
                event_type=APPROVAL_GRANTED,
                actor="user",
                occurred_at_epoch_ms=_now_ms(),
                subject=patch_id,
                outcome="granted",
                details={"scope_count": str(len(proposal.artifact.touched_files))},
            )
        )
        return token

    def apply(self, run_id: str, patch_id: str, approval_token: str) -> ChangeResult:
        proposal = self._proposal(run_id, patch_id)
        if run_id in self._cancelled:
            raise ChangeRequestError("Run 已收到取消请求，未开始应用")
        previous = self._results.get(f"{run_id}:{patch_id}")
        if previous is not None:
            return previous
        validate_change_set_base(
            proposal.repository_root, proposal.artifact, proposal.base_fingerprints
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
        try:
            self.approvals.consume(approval_token, now_epoch_ms=_now_ms())
        except (ApprovalNotFoundError, ApprovalAlreadyConsumedError, ApprovalExpiredError) as error:
            raise ChangeRequestError(str(error)) from error

        key = IdempotencyKey.for_action(
            goal_id=run_id, action="apply_patch", fingerprint=proposal.artifact.diff_hash
        )
        if not self.idempotency.register(key, result_ref=f"{run_id}:{patch_id}"):
            existing = self._results.get(f"{run_id}:{patch_id}")
            if existing is not None:
                return existing
            raise ChangeRequestError("该补丁正在被另一个请求处理，请稍后读取 Run 状态")

        managed = self.workspaces.create(proposal.repository_root, run_id)
        checkpoint = self.workspaces.create_checkpoint(run_id)
        self._emit(run_id, CHECKPOINT_CREATED, {"checkpoint_id": checkpoint.checkpoint_id})
        try:
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
            checked = self.sandbox.run(
                proposal.preview.validation_profile, managed.run.workspace_path
            )
            validation = self._validation_payload(checked, run_id)
            self._store_validation(run_id, validation)
            if not checked.passed:
                result = self._result(
                    proposal,
                    managed.run.workspace_id,
                    checkpoint.checkpoint_id,
                    "REVIEW_REQUIRED",
                    "固定检查失败，等待有限修复或人工处理",
                    validation,
                )
            else:
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
        except Exception:
            self.workspaces.rollback(run_id, checkpoint.checkpoint_id)
            raise

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
            preview = self.preview_many(
                original.repository_root,
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
        proposal = self._proposal(run_id, patch_id)
        managed = self.workspaces.get(run_id)
        if managed is None or managed.run.latest_checkpoint is None:
            raise ChangeRequestError("当前 Run 没有可回滚的 checkpoint")
        checkpoint = managed.run.latest_checkpoint
        self.workspaces.rollback(run_id, checkpoint.checkpoint_id)
        result = self._result(
            proposal,
            managed.run.workspace_id,
            checkpoint.checkpoint_id,
            "ROLLED_BACK",
            "已恢复到 checkpoint",
        )
        self._store_result(result)
        self._emit(run_id, ROLLBACK_PERFORMED, {"checkpoint_id": checkpoint.checkpoint_id})
        self.audit.record(
            AuditRecord(
                audit_id=f"audit-{uuid4().hex[:16]}",
                run_id=run_id,
                event_type=ROLLBACK_PERFORMED,
                actor="system",
                occurred_at_epoch_ms=_now_ms(),
                subject=patch_id,
                outcome="rolled_back",
                details={"checkpoint_id": checkpoint.checkpoint_id},
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

    def _proposal(self, run_id: str, patch_id: str) -> _Proposal:
        try:
            return self._proposals[(run_id, patch_id)]
        except KeyError as error:
            raise ChangeRequestError("Patch 提案不存在或不属于当前 Run") from error

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
        self._results[f"{result.run_id}:{result.patch_id}"] = result
        self.state.put("results", f"{result.run_id}:{result.patch_id}", result.as_dict())

    def _store_validation(self, run_id: str, payload: dict[str, object]) -> None:
        digest = None
        if not bool(payload["passed"]):
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
            bool(payload["passed"]),
            digest,
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
            self._results[f"{result.run_id}:{result.patch_id}"] = result
        for raw in self.state.list("cancelled"):
            if raw.get("cancelled"):
                self._cancelled.add(str(raw["run_id"]))

    def _result(
        self,
        proposal: _Proposal,
        workspace_id: str,
        checkpoint_id: str,
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
