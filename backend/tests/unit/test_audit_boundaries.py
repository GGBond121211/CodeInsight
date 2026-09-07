import subprocess
from pathlib import Path

import pytest

from codeinsight.cli.change_demo import run_demo
from codeinsight.domain.change import ChangeApproval
from codeinsight.infrastructure.persistent_change_state import PersistentApprovalStore
from codeinsight.infrastructure.run_store import ApprovalAlreadyConsumedError
from codeinsight.infrastructure.sandbox import DockerSandbox, SandboxCleanupError, SandboxResult
from codeinsight.ingestion.path_policy import safe_relative_path


@pytest.mark.parametrize(
    "path", ["C:secret.py", "a.py:secret", "..\\secret.py", ".GIT/config", ".env.dev"]
)
def test_shared_policy_rejects_unsafe_paths(path):
    with pytest.raises(PermissionError):
        safe_relative_path(path)


def test_consumed_approval_wins_if_crash_leaves_active_record(tmp_path, monkeypatch):
    store = PersistentApprovalStore(tmp_path)
    store.issue(ChangeApproval("token", "run", "diff", "base", ("a.py",), 1000))
    original = Path.unlink

    def fail_active_delete(path, *args, **kwargs):
        if path.parent.name == "active":
            raise OSError("interrupted")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_active_delete)
    with pytest.raises(OSError, match="interrupted"):
        store.consume("token", now_epoch_ms=10)
    restarted = PersistentApprovalStore(tmp_path)
    with pytest.raises(ApprovalAlreadyConsumedError):
        restarted.consume("token", now_epoch_ms=20)


@pytest.mark.parametrize("cleanup_confirmed", [True, False])
def test_docker_timeout_requires_confirmed_container_cleanup(
    tmp_path, monkeypatch, cleanup_confirmed
):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(argv, 0 if cleanup_confirmed else 1, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    if cleanup_confirmed:
        assert DockerSandbox().run("pytest", tmp_path).error_class == "TIMEOUT"
    else:
        with pytest.raises(SandboxCleanupError):
            DockerSandbox().run("pytest", tmp_path)
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "--force", name]


@pytest.mark.parametrize("approved", [False, True])
def test_demo_requires_explicit_approval_and_can_rollback(tmp_path, monkeypatch, capsys, approved):
    class Sandbox:
        calls = 0

        def run(self, profile, workspace_path):
            self.calls += 1
            assert "a + b" in (Path(workspace_path) / "app.py").read_text()
            return SandboxResult(profile, ("python -m pytest -q",), True)

    inputs = iter(["APPLY", "ROLLBACK"] if approved else ["no"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    sandbox = Sandbox()
    assert run_demo(tmp_path, sandbox=sandbox) == 0
    assert sandbox.calls == int(approved)
    demo = next(tmp_path.glob("demo-*"))
    assert "a - b" in (demo / "source" / "app.py").read_text()
    if approved:
        workspace = next((demo / "managed").glob("ws-*"))
        assert "a - b" in (workspace / "app.py").read_text()
        assert "ROLLED_BACK" in capsys.readouterr().out
