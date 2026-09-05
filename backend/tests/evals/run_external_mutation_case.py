"""固定 requests commit 的外部仓库 FAIL_TO_PASS / PASS_TO_PASS 验收。"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from codeinsight.infrastructure.sandbox import DockerSandbox
from codeinsight.infrastructure.workspace import WorkspaceManager

EXPECTED_COMMIT = "6e83187b8feb273ed4c6cdab5efd8d54901dfab3"
TARGET = Path("src/requests/status_codes.py")
MUTATION = '418: ("im_a_teapot", "teapot", "i_am_a_teapot"),'
BROKEN = '419: ("im_a_teapot", "teapot", "i_am_a_teapot"),'

TEST_FILE = '''from pathlib import Path

SOURCE = Path("src/requests/status_codes.py").read_text(encoding="utf-8")

def test_fail_to_pass_teapot_status_is_418():
    assert '418: ("im_a_teapot", "teapot", "i_am_a_teapot"),' in SOURCE

def test_pass_to_pass_ok_status_stays_200():
    assert '200: ("ok", "okay", "all_ok"' in SOURCE
'''


def main() -> int:
    project = Path(__file__).resolve().parents[3]
    source = project / "work" / "benchmarks" / "requests"
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"requests commit 漂移：{commit}")
    source_target = source / TARGET
    source_bytes_before = source_target.read_bytes()
    source_status_before = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout

    with tempfile.TemporaryDirectory(prefix="codeinsight-external-") as temporary:
        manager = WorkspaceManager(Path(temporary) / "managed")
        managed = manager.create(source, "external-requests-418")
        workspace = Path(managed.run.workspace_path)
        target = workspace / TARGET
        original = target.read_text(encoding="utf-8")
        if MUTATION not in original:
            raise RuntimeError("固定 mutation 位置不存在")
        (workspace / "codeinsight_mutation_test.py").write_text(TEST_FILE, encoding="utf-8")
        target.write_text(original.replace(MUTATION, BROKEN, 1), encoding="utf-8")
        sandbox = DockerSandbox()
        broken = sandbox.run("requests_mutation_pytest", workspace)
        target.write_text(original, encoding="utf-8")
        repaired = sandbox.run("requests_mutation_pytest", workspace)

    source_status_after = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    source_repository_modified = (
        source_target.read_bytes() != source_bytes_before
        or source_status_after != source_status_before
    )

    payload = {
        "case": "requests-status-codes-418",
        "repository": "psf/requests",
        "commit": commit,
        "mutation_file": TARGET.as_posix(),
        "fail_to_pass": {
            "before_repair_passed": broken.passed,
            "after_repair_passed": repaired.passed,
        },
        "pass_to_pass": {"after_repair_passed": repaired.passed},
        "source_repository_modified": source_repository_modified,
        "sandbox_error_before": broken.error_class,
        "sandbox_error_after": repaired.error_class,
    }
    output = project / "experiments" / "results" / "STEP8-CHANGE-CLOSURE"
    output.mkdir(parents=True, exist_ok=True)
    (output / "external-mutation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if (not broken.passed and repaired.passed and not source_repository_modified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
