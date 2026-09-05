"""Step 6 安全边界与失败结果测试。"""

from pathlib import Path

import pytest

from codeinsight.application.change_service import ChangeRequestError, ChangeService
from codeinsight.infrastructure.sandbox import SandboxResult
from codeinsight.infrastructure.workspace import WorkspaceManager


class PassingSandbox:
    def run(self, profile, workspace_path):
        return SandboxResult(profile, ("python -m compileall -q .",), True)


class FailingSandbox:
    def run(self, profile, workspace_path):
        return SandboxResult(
            profile, ("python -m compileall -q .",), False, "CHECK_FAILED", "syntax error"
        )


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "app.py"
    path.write_text("value = 1\n", encoding="utf-8")
    return repo, path


def test_apply_isolated_and_idempotent(tmp_path: Path) -> None:
    repo, source = _repo(tmp_path)
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=PassingSandbox()
    )
    preview = service.preview(
        repo,
        run_id="run-1",
        path="app.py",
        new_content="value = 2\n",
        validation_profile="python_compile",
    )
    token = service.approve("run-1", preview.patch_id)
    first = service.apply("run-1", preview.patch_id, token)
    second = service.apply("run-1", preview.patch_id, token)
    assert first.status == "COMPLETED"
    assert second == first
    assert source.read_text(encoding="utf-8") == "value = 1\n"
    managed = service.workspaces.get("run-1")
    assert managed is not None
    assert (
        Path(managed.run.workspace_path) / "app.py"
    ).read_text(encoding="utf-8") == "value = 2\n"


def test_failed_check_returns_review_required_with_redacted_digest(tmp_path: Path) -> None:
    repo, source = _repo(tmp_path)
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=FailingSandbox()
    )
    preview = service.preview(
        repo,
        run_id="run-fail",
        path="app.py",
        new_content="value = 2\n",
        validation_profile="python_compile",
    )
    token = service.approve("run-fail", preview.patch_id)
    result = service.apply("run-fail", preview.patch_id, token)
    assert result.status == "REVIEW_REQUIRED"
    assert result.validation is not None
    assert result.validation["failure_digest_id"]
    assert source.read_text(encoding="utf-8") == "value = 1\n"


def test_wrong_approval_and_baseline_change_cannot_apply(tmp_path: Path) -> None:
    repo, source = _repo(tmp_path)
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=PassingSandbox()
    )
    preview = service.preview(
        repo,
        run_id="run-safe",
        path="app.py",
        new_content="value = 2\n",
        validation_profile="python_compile",
    )
    with pytest.raises(ChangeRequestError, match="不存在"):
        service.apply("run-safe", preview.patch_id, "wrong-token")
    token = service.approve("run-safe", preview.patch_id)
    source.write_text("value = 99\n", encoding="utf-8")
    with pytest.raises((ChangeRequestError, RuntimeError), match="指纹|基线"):
        service.apply("run-safe", preview.patch_id, token)


def test_path_and_profile_are_allowlisted(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    service = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=PassingSandbox()
    )
    with pytest.raises(PermissionError):
        service.preview(
            repo,
            run_id="run-path",
            path="../outside.py",
            new_content="x = 1\n",
            validation_profile="python_compile",
        )
    with pytest.raises(ValueError, match="profile"):
        service.preview(
            repo,
            run_id="run-profile",
            path="app.py",
            new_content="x = 1\n",
            validation_profile="curl_anywhere",
        )
