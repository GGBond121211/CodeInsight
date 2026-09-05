from pathlib import Path

import pytest

from codeinsight.agent.change_workflow import run_change_workflow_to_preview
from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse, ToolResult
from codeinsight.application.change_service import ChangeService
from codeinsight.infrastructure.redaction import redact_sensitive
from codeinsight.infrastructure.sandbox import SandboxResult
from codeinsight.infrastructure.workspace import WorkspaceManager


class PassingSandbox:
    def run(self, profile, workspace_path):
        return SandboxResult(profile, ("python -m pytest -q",), True)


class ExplodingSandbox:
    def run(self, profile, workspace_path):
        raise RuntimeError("worker died")


class RepairSandbox:
    def run(self, profile, workspace_path):
        passed = "value = 3" in (Path(workspace_path) / "app.py").read_text(encoding="utf-8")
        return SandboxResult(
            profile,
            ("python -m pytest -q",),
            passed,
            None if passed else "CHECK_FAILED",
            "C:\\Users\\Alex\\repo\\test.py Authorization: Bearer sk-secret password=hunter2",
        )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    (repo / "remove.py").write_text("remove = True\n", encoding="utf-8")
    return repo


def test_multifile_update_create_delete_is_atomic_and_rollbackable(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=PassingSandbox()
    )
    preview = service.preview_many(
        repo,
        run_id="run-multi",
        changes={"a.py": "a = 2\n", "new.py": "new = True\n", "remove.py": None},
        validation_profile="pytest",
    )
    token = service.approve("run-multi", preview.patch_id)
    result = service.apply("run-multi", preview.patch_id, token)
    workspace = Path(service.workspaces.get("run-multi").run.workspace_path)  # type: ignore[union-attr]
    assert result.status == "COMPLETED"
    assert (workspace / "a.py").read_text(encoding="utf-8") == "a = 2\n"
    assert (workspace / "new.py").is_file()
    assert not (workspace / "remove.py").exists()
    assert (repo / "a.py").read_text(encoding="utf-8") == "a = 1\n"
    service.rollback("run-multi", preview.patch_id)
    assert (workspace / "a.py").read_text(encoding="utf-8") == "a = 1\n"
    assert not (workspace / "new.py").exists()
    assert (workspace / "remove.py").is_file()


def test_exception_during_validation_rolls_back_entire_change_set(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=ExplodingSandbox()
    )
    preview = service.preview_many(
        repo, run_id="run-atomic", changes={"a.py": "a = 2\n", "new.py": "x = 1\n"}
    )
    token = service.approve("run-atomic", preview.patch_id)
    with pytest.raises(RuntimeError, match="worker died"):
        service.apply("run-atomic", preview.patch_id, token)
    workspace = Path(service.workspaces.get("run-atomic").run.workspace_path)  # type: ignore[union-attr]
    assert (workspace / "a.py").read_text(encoding="utf-8") == "a = 1\n"
    assert not (workspace / "new.py").exists()


def test_service_restart_recovers_persistent_change_state(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    managed = tmp_path / "managed"
    first = ChangeService(workspace_manager=WorkspaceManager(managed), sandbox=PassingSandbox())
    preview = first.preview(
        repo,
        run_id="run-restart",
        path="a.py",
        new_content="a = 2\n",
        validation_profile="pytest",
    )
    token = first.approve("run-restart", preview.patch_id)

    second = ChangeService(workspace_manager=WorkspaceManager(managed), sandbox=PassingSandbox())
    result = second.apply("run-restart", preview.patch_id, token)
    third = ChangeService(workspace_manager=WorkspaceManager(managed), sandbox=PassingSandbox())

    assert result.status == "COMPLETED"
    assert third.apply("run-restart", preview.patch_id, token) == result
    assert len(third.events_after("run-restart")) >= 5
    restored = third.workspaces.get("run-restart")
    assert restored is not None and restored.run.latest_checkpoint is not None
    assert len(third.state.list("validations")) == 1
    persisted = "".join(path.read_text(encoding="utf-8") for path in managed.rglob("*.json"))
    assert token not in persisted


def test_failure_digest_redacts_credentials_private_key_env_and_absolute_paths() -> None:
    raw = """Authorization: Bearer sk-live
API_KEY=abc123
password=hunter2
C:\\Users\\Alex\\secret.txt
/home/alex/secret.txt
-----BEGIN PRIVATE KEY-----
abc
-----END PRIVATE KEY-----"""
    cleaned = redact_sensitive(raw)
    for secret in ("sk-live", "abc123", "hunter2", "C:\\Users", "/home/alex", "abc"):
        assert secret not in cleaned


def test_failed_check_can_generate_bounded_repairs_with_new_approvals(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=RepairSandbox()
    )
    preview = service.preview(
        repo,
        run_id="run-repair",
        path="app.py",
        new_content="value = 2\n",
        validation_profile="pytest",
    )
    approvals = 0

    def approve_repair(item):
        nonlocal approvals
        approvals += 1
        return service.approve(item.run_id, item.patch_id)

    result = service.apply_with_repairs(
        "run-repair",
        preview.patch_id,
        service.approve("run-repair", preview.patch_id),
        repairer=lambda digest, attempt: {"app.py": "value = 3\n"},
        approve_repair=approve_repair,
    )
    assert result.status == "COMPLETED"
    assert approvals == 1


def test_tool_loop_generated_patch_becomes_change_service_preview(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")

    class Host:
        def list_tools(self):
            return ({"name": "generate_patch", "inputSchema": {"type": "object"}},)

        def call_tool(self, call):
            return ToolResult.success(call.id, call.name, {"changed": True})

    class Model:
        round = 0

        def complete_with_tools(self, messages, tools):
            self.round += 1
            if self.round == 1:
                return ToolModelResponse(
                    None,
                    (
                        ToolCall(
                            "patch-1",
                            "generate_patch",
                            {"path": "app.py", "new_content": "value = 2\n"},
                        ),
                    ),
                    "fixed",
                )
            return ToolModelResponse("done", (), "fixed")

    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=PassingSandbox()
    )
    result = run_change_workflow_to_preview(
        "change value",
        str(repo),
        model=Model(),
        mcp_client=Host(),
        change_service=service,
        run_id="run-tool-loop",
    )
    assert result.preview.diff_hash
    assert service.events_after("run-tool-loop")[0].event_type == "patch_generated"
