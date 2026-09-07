"""审视发现的真实失败路径：增量基线、中断、越权产物与重复执行。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from codeinsight.application.change_service import ChangeRequestError, ChangeService
from codeinsight.infrastructure.sandbox import SandboxCleanupError, SandboxResult
from codeinsight.infrastructure.workspace import WorkspaceManager


class PassingSandbox:
    def run(self, profile, workspace_path):
        return SandboxResult(profile, ("python -m compileall -q .",), True)


def setup_service(tmp_path, sandbox=None):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=sandbox or PassingSandbox(),
    )
    return repo, service


def propose(service, repo, changes):
    preview = service.preview_many(repo, run_id="run", changes=changes)
    return preview, service.approve("run", preview.patch_id)


def test_incremental_repair_preserves_other_files_and_early_rollback_invalidates_later(tmp_path):
    repo, service = setup_service(tmp_path)
    first, token = propose(service, repo, {"a.py": "a = 2\n", "b.py": "b = 2\n"})
    service.apply("run", first.patch_id, token)
    second, token = propose(service, repo, {"a.py": "a = 3\n"})
    service.apply("run", second.patch_id, token)
    workspace = Path(service.workspaces.get("run").run.workspace_path)
    assert (workspace / "b.py").read_text() == "b = 2\n"
    assert (workspace / "a.py").read_text() == "a = 3\n"
    with pytest.raises(ChangeRequestError, match="不同"):
        propose(service, tmp_path, {"a.py": "a = 4\n"})
    service.rollback("run", first.patch_id)
    assert service.get_result("run", second.patch_id).status == "ROLLED_BACK"
    service.rollback("run", second.patch_id)
    assert (workspace / "a.py").read_text() == "a = 1\n"
    assert not (workspace / "b.py").exists()


def test_checkpoint_failure_is_durable_and_does_not_reconsume_token(tmp_path, monkeypatch):
    repo, service = setup_service(tmp_path)
    preview, token = propose(service, repo, {"a.py": "a = 2\n"})
    original = service.workspaces.create_checkpoint

    def fail(_run):
        raise OSError("disk full")

    monkeypatch.setattr(service.workspaces, "create_checkpoint", fail)
    with pytest.raises(OSError, match="disk full"):
        service.apply("run", preview.patch_id, token)
    assert service.apply("run", preview.patch_id, token).status == "FAILED"
    monkeypatch.setattr(service.workspaces, "create_checkpoint", original)
    new, new_token = propose(service, repo, {"a.py": "a = 2\n"})
    assert service.apply("run", new.patch_id, new_token).status == "COMPLETED"


def test_restart_of_running_operation_is_unknown_and_blocks_new_writes(tmp_path):
    repo, service = setup_service(tmp_path)
    preview, token = propose(service, repo, {"a.py": "a = 2\n"})
    result = service.apply("run", preview.patch_id, token)
    raw = result.as_dict()
    raw["status"] = "RUNNING"
    service.state.put("results", f"run:{preview.patch_id}", raw)
    restarted = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=PassingSandbox()
    )
    assert restarted.apply("run", preview.patch_id, token).status == "UNKNOWN"
    with pytest.raises(ChangeRequestError, match="未确认"):
        propose(restarted, repo, {"a.py": "a = 3\n"})
    with pytest.raises(ChangeRequestError, match="未确认"):
        restarted.rollback("run", preview.patch_id)


@pytest.mark.parametrize("filename", ["unapproved.py", ".env"])
def test_validation_cannot_smuggle_unapproved_artifacts(tmp_path, filename):
    class MutatingSandbox:
        def run(self, profile, workspace_path):
            (Path(workspace_path) / filename).write_text("unexpected", encoding="utf-8")
            return SandboxResult(profile, ("python -m compileall -q .",), True)

    repo, service = setup_service(tmp_path, MutatingSandbox())
    preview, token = propose(service, repo, {"a.py": "a = 2\n"})
    with pytest.raises(ChangeRequestError, match="产物"):
        service.apply("run", preview.patch_id, token)
    assert service.get_result("run", preview.patch_id).status == "ROLLED_BACK"
    workspace = Path(service.workspaces.get("run").run.workspace_path)
    assert not (workspace / filename).exists()
    assert (repo / "a.py").read_text() == "a = 1\n"


def test_unconfirmed_container_lifetime_does_not_race_rollback(tmp_path, monkeypatch):
    class UnconfirmedSandbox:
        def run(self, profile, workspace_path):
            raise SandboxCleanupError("container still running")

    repo, service = setup_service(tmp_path, UnconfirmedSandbox())
    preview, token = propose(service, repo, {"a.py": "a = 2\n"})

    def forbidden(*args):
        pytest.fail("不能在容器存活状态不明时回滚")

    monkeypatch.setattr(service.workspaces, "rollback", forbidden)
    with pytest.raises(SandboxCleanupError):
        service.apply("run", preview.patch_id, token)
    assert service.get_result("run", preview.patch_id).status == "UNKNOWN"


def test_concurrent_duplicate_apply_executes_sandbox_once(tmp_path):
    started, release = Event(), Event()

    class BlockingSandbox:
        calls = 0

        def run(self, profile, workspace_path):
            self.calls += 1
            started.set()
            assert release.wait(5)
            return SandboxResult(profile, ("python -m compileall -q .",), True)

    sandbox = BlockingSandbox()
    repo, service = setup_service(tmp_path, sandbox)
    preview, token = propose(service, repo, {"a.py": "a = 2\n"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.apply, "run", preview.patch_id, token)
        try:
            assert started.wait(5)
            second = pool.submit(service.apply, "run", preview.patch_id, token)
        finally:
            release.set()
        assert first.result() == second.result()
    assert sandbox.calls == 1


def test_workspace_inside_source_is_rejected_before_copy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = WorkspaceManager(repo / "managed")
    with pytest.raises(ValueError, match="之外"):
        manager.create(repo, "run")
    assert not list(manager.base_dir.glob("ws-*"))
