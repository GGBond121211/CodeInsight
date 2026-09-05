"""Step 6 HTTP 闭环：API 只对临时 workspace 写入。"""

from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.application.change_service import ChangeService
from codeinsight.infrastructure.sandbox import SandboxResult
from codeinsight.infrastructure.workspace import WorkspaceManager


class PassingSandbox:
    def run(self, profile, workspace_path):
        return SandboxResult(profile, ("python -m compileall -q .",), True)


class FakeModel:
    pass


def _service(tmp_path: Path) -> ChangeService:
    return ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"), sandbox=PassingSandbox()
    )


def test_preview_approve_apply_and_rollback_are_separate_steps(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_file = repo / "app.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    service = _service(tmp_path)
    client = TestClient(create_app(lambda: FakeModel(), lambda: None, service))  # type: ignore[arg-type]

    preview = client.post(
        "/api/v2/change/preview",
        json={
            "repository_root": str(repo),
            "run_id": "run-api-1",
            "path": "app.py",
            "new_content": "value = 2\n",
            "validation_profile": "python_compile",
        },
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["status"] == "preview_ready"
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"

    approved = client.post(
        "/api/v2/change/approve",
        json={"run_id": "run-api-1", "patch_id": preview_payload["patch_id"]},
    )
    assert approved.status_code == 200
    token = approved.json()["approval_token"]

    applied = client.post(
        "/api/v2/change/apply",
        json={
            "run_id": "run-api-1",
            "patch_id": preview_payload["patch_id"],
            "approval_token": token,
        },
    )
    assert applied.status_code == 200
    assert applied.json()["status"] == "COMPLETED"
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"

    rolled_back = client.post(
        "/api/v2/change/rollback",
        json={"run_id": "run-api-1", "patch_id": preview_payload["patch_id"]},
    )
    assert rolled_back.status_code == 200
    assert rolled_back.json()["status"] == "ROLLED_BACK"

    events = client.get("/api/v2/change/run-api-1/events")
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "patch_generated" in events.text
    assert "checkpoint_created" in events.text
    assert "run_finished" in events.text


def test_cancel_is_recorded_and_prevents_apply(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    service = _service(tmp_path)
    client = TestClient(create_app(lambda: FakeModel(), lambda: None, service))  # type: ignore[arg-type]
    preview = client.post(
        "/api/v2/change/preview",
        json={
            "repository_root": str(repo),
            "run_id": "run-cancel-1",
            "path": "app.py",
            "new_content": "value = 2\n",
        },
    ).json()
    approved = client.post(
        "/api/v2/change/approve",
        json={"run_id": "run-cancel-1", "patch_id": preview["patch_id"]},
    )
    assert approved.status_code == 200
    cancelled = client.post("/api/v2/change/run-cancel-1/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json() == {
        "status": "cancel_requested",
        "run_id": "run-cancel-1",
        "immediate": True,
    }
    blocked = client.post(
        "/api/v2/change/apply",
        json={
            "run_id": "run-cancel-1",
            "patch_id": preview["patch_id"],
            "approval_token": approved.json()["approval_token"],
        },
    )
    assert blocked.status_code == 409
