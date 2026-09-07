"""固定样例契约演示：不调用付费模型，不修改用户源仓库。"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from codeinsight.agent.tool_loop import ToolCall
from codeinsight.application.change_service import ChangeService
from codeinsight.infrastructure.sandbox import SandboxRunner
from codeinsight.infrastructure.tool_executor import ToolExecutor
from codeinsight.infrastructure.workspace import WorkspaceManager


def run_demo(work_dir: str | Path, *, sandbox: SandboxRunner | None = None) -> int:
    directory = Path(work_dir).resolve() / f"demo-{uuid4().hex[:12]}"
    repo = directory / "source"
    repo.mkdir(parents=True)
    source = repo / "app.py"
    source.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "from app import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    service = ChangeService(
        workspace_manager=WorkspaceManager(directory / "managed"), sandbox=sandbox
    )
    print("固定样例契约演示：补丁由程序给定，不证明真实模型修复能力。")
    evidence = ToolExecutor(repo).execute(ToolCall("read", "read_file", {"path": "app.py"}))
    print(json.dumps(evidence.data, ensure_ascii=False))
    preview = service.preview(
        repo, run_id=None, path="app.py",
        new_content="def add(a, b):\n    return a + b\n", validation_profile="pytest",
    )
    print(preview.diff)
    print(f"状态保留在：{directory}")
    try:
        approved = input("输入 APPLY 批准此 diff 并在 Docker 中运行固定 pytest：") == "APPLY"
    except EOFError:
        approved = False
    if not approved:
        print("未批准；没有应用补丁或运行 Sandbox。")
        return 0
    try:
        token = service.approve(preview.run_id, preview.patch_id)
        result = service.apply(preview.run_id, preview.patch_id, token)
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    except (OSError, RuntimeError, ValueError) as error:
        result = service.get_result(preview.run_id, preview.patch_id)
        print(f"执行未完成：{type(error).__name__}")
        if result:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 1
    if result.status == "UNKNOWN":
        return 1
    try:
        rollback = input("输入 ROLLBACK 恢复隔离 workspace 的 checkpoint：") == "ROLLBACK"
    except EOFError:
        rollback = False
    if rollback:
        try:
            restored = service.rollback(preview.run_id, preview.patch_id)
            print(restored.status)
        except (OSError, RuntimeError, ValueError):
            print("回滚未确认完成；请人工核对保留的工作区和结果。")
            return 1
    print("源样例保持只读：", source.read_text(encoding="utf-8").strip())
    return 0 if result.status == "COMPLETED" else 1
