"""MCP 工具的唯一执行闸门。

这里的输入可能来自模型，也可能来自恶意仓库文本，因此不能相信 Prompt、
文件名或模型自报的权限。所有调用都要经过登记、参数、路径和能力检查。
Step 5 只提供补丁提案和验证；真正 Sandbox 应用闭环属于 Step 6。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeinsight.agent.tool_loop import ToolCall, ToolResult
from codeinsight.application.search_repository import search_repository
from codeinsight.infrastructure.event_log import InMemoryEventLog
from codeinsight.infrastructure.tool_registry import ToolRegistry, build_default_registry
from codeinsight.retrieval.repository_map import build_repository_map

SENSITIVE_BASENAMES = frozenset({".env", ".env.local", ".env.production", "id_rsa", "id_dsa"})
SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


@dataclass(frozen=True)
class ExecutorCapabilities:
    """Step 5 的能力开关；默认全是 False，避免误把预览当执行。"""

    checkpoint: bool = False
    sandbox: bool = False
    approval: bool = False


class ToolExecutor:
    def __init__(
        self,
        repository_root: str | Path,
        *,
        registry: ToolRegistry | None = None,
        run_id: str = "local-run",
        event_log: InMemoryEventLog | None = None,
        capabilities: ExecutorCapabilities | None = None,
    ) -> None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root 必须是存在的目录")
        self.root = root
        self.registry = registry or build_default_registry()
        self.run_id = run_id
        self.event_log = event_log
        self.capabilities = capabilities or ExecutorCapabilities()
        self._patches: dict[str, dict[str, object]] = {}

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self.registry.get(call.name)
        if spec is None:
            return ToolResult.failure(call.id, call.name, "VALIDATION", "工具未在 MCP 目录登记")
        validation_error = self._validate_arguments(spec.input_schema, call.arguments)
        if validation_error:
            return ToolResult.failure(call.id, call.name, "VALIDATION", validation_error)
        try:
            if call.name == "search_repository":
                return self._search(call)
            if call.name == "read_file":
                return self._read_file(call)
            if call.name == "get_repository_map":
                return self._repository_map(call)
            if call.name == "get_evidence_context":
                return self._evidence_context(call)
            if call.name == "generate_patch":
                return self._generate_patch(call)
            if call.name == "validate_patch":
                return self._validate_patch(call)
            if call.name == "get_run_events":
                return self._run_events(call)
            if call.name == "get_diff":
                return self._get_diff(call)
            return self._blocked_write(call)
        except FileNotFoundError:
            return ToolResult.failure(call.id, call.name, "NOT_FOUND", "目标不存在")
        except TimeoutError:
            return ToolResult.failure(call.id, call.name, "TIMEOUT", "工具执行超时")
        except PermissionError:
            return ToolResult.failure(call.id, call.name, "PERMISSION", "执行层拒绝访问该资源")
        except (OSError, ValueError):
            # 不把绝对路径、文件内容或环境变量泄漏到 ToolResult/日志。
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "工具执行失败")

    def _search(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        question = str(args["question"])
        limit = int(args.get("limit", 5))
        hits = search_repository(self.root, question, limit=limit, retrieval_mode="bm25")
        data = {"results": [_serialize_hit(item) for item in hits], "untrusted": True}
        return ToolResult.success(call.id, call.name, data, state_fingerprint=_fingerprint(data))

    def _read_file(self, call: ToolCall) -> ToolResult:
        relative = self._safe_relative_path(str(call.arguments["path"]))
        path = self._safe_path(relative)
        lines = path.read_text(encoding="utf-8").splitlines()
        start = int(call.arguments.get("start_line", 1))
        end = int(call.arguments.get("end_line", min(len(lines), start + 199)))
        if end < start or end - start >= 1000:
            raise ValueError("读取行范围无效或过大")
        selected = lines[start - 1 : end]
        data = {
            "path": relative,
            "start_line": start,
            "end_line": min(end, len(lines)),
            "text": "\n".join(selected),
            "untrusted": True,
        }
        return ToolResult.success(
            call.id,
            call.name,
            data,
            state_fingerprint=_fingerprint({"path": relative, "lines": len(lines)}),
        )

    def _repository_map(self, call: ToolCall) -> ToolResult:
        repo_map = build_repository_map(self.root)
        data = {
            "repo_id": repo_map.repo_id,
            "index_version": repo_map.index_version,
            "symbols": list(repo_map.symbols),
            "imports": [list(item) for item in repo_map.imports],
            "file_summaries": [list(item) for item in repo_map.file_summaries],
            "untrusted": True,
            "purpose": "navigation_only",
        }
        return ToolResult.success(call.id, call.name, data, state_fingerprint=_fingerprint(data))

    def _evidence_context(self, call: ToolCall) -> ToolResult:
        result = self._search(call)
        if not result.ok:
            return result
        return ToolResult.success(
            call.id,
            call.name,
            {"evidence": result.data.get("results", []), "untrusted": True},
            state_fingerprint=result.state_fingerprint,
        )

    def _generate_patch(self, call: ToolCall) -> ToolResult:
        relative = self._safe_relative_path(str(call.arguments["path"]))
        path = self._safe_path(relative)
        old = path.read_text(encoding="utf-8")
        new = str(call.arguments["new_content"])
        patch_id = hashlib.sha256(f"{relative}\0{old}\0{new}".encode()).hexdigest()[:20]
        old_fingerprint = hashlib.sha256(old.encode("utf-8")).hexdigest()
        patch_text = _unified_diff(relative, old, new)
        payload = {
            "patch_id": patch_id,
            "path": relative,
            "diff": patch_text,
            "base_fingerprint": old_fingerprint,
            "changed": old != new,
        }
        self._patches[patch_id] = payload
        return ToolResult.success(call.id, call.name, payload, state_fingerprint=old_fingerprint)

    def _validate_patch(self, call: ToolCall) -> ToolResult:
        patch_id = str(call.arguments["patch_id"])
        patch = self._patches.get(patch_id)
        if patch is None:
            return ToolResult.failure(call.id, call.name, "NOT_FOUND", "补丁提案不存在")
        current = self._safe_path(str(patch["path"])).read_text(encoding="utf-8")
        current_fingerprint = hashlib.sha256(current.encode("utf-8")).hexdigest()
        valid = current_fingerprint == patch["base_fingerprint"] and bool(patch["changed"])
        data = {
            "patch_id": patch_id,
            "valid": valid,
            "scope": [patch["path"]],
            "diff": patch["diff"],
        }
        if not valid:
            return ToolResult.failure(
                call.id, call.name, "DB_CONFLICT", "源文件指纹已变化或补丁为空", data=data
            )
        return ToolResult.success(call.id, call.name, data, state_fingerprint=current_fingerprint)

    def _run_events(self, call: ToolCall) -> ToolResult:
        if self.event_log is None:
            return ToolResult.success(
                call.id, call.name, {"events": []}, state_fingerprint="events:0"
            )
        run_id = str(call.arguments["run_id"])
        after = int(call.arguments.get("after_sequence", 0))
        events = self.event_log.read_events(run_id, after_sequence=after)
        data = {
            "events": [
                {
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "payload": event.payload,
                }
                for event in events
            ]
        }
        return ToolResult.success(call.id, call.name, data, state_fingerprint=_fingerprint(data))

    def _get_diff(self, call: ToolCall) -> ToolResult:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "diff", "--no-ext-diff", "--", "."],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ToolResult.success(
                call.id,
                call.name,
                {"diff": "", "available": False},
                state_fingerprint="diff:unavailable",
            )
        return ToolResult.success(
            call.id,
            call.name,
            {"diff": completed.stdout[:100_000], "available": True},
            state_fingerprint=_fingerprint(completed.stdout),
        )

    def _blocked_write(self, call: ToolCall) -> ToolResult:
        missing: list[str] = []
        if not self.capabilities.approval:
            missing.append("approval")
        if not self.capabilities.checkpoint:
            missing.append("checkpoint")
        if not self.capabilities.sandbox:
            missing.append("sandbox")
        code = "PERMISSION" if call.name == "apply_patch_isolated" else "VALIDATION"
        return ToolResult.failure(
            call.id,
            call.name,
            code,
            "修改类工具未执行：缺少 " + ", ".join(missing) + "；Step 6 才提供完整能力",
            data={"status": "manual_required", "missing": missing, "applied": False},
        )

    def _safe_relative_path(self, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PermissionError("路径必须位于仓库根目录内")
        normalized = candidate.as_posix()
        if not normalized or normalized == ".":
            raise ValueError("路径不能为空")
        if any(part in {".git", ".hg", ".svn"} for part in candidate.parts):
            raise PermissionError("版本控制目录不属于工具允许范围")
        basename = candidate.name.lower()
        if (
            basename in SENSITIVE_BASENAMES
            or basename.startswith(".env.")
            or basename.endswith(SENSITIVE_SUFFIXES)
        ):
            raise PermissionError("敏感文件不允许由工具读取或修改")
        return normalized

    def _safe_path(self, relative: str) -> Path:
        path = self.root / relative
        resolved_parent = path.parent.resolve()
        try:
            resolved_parent.relative_to(self.root)
        except ValueError as error:
            raise PermissionError("符号链接或父目录越界") from error
        if path.exists() and path.resolve() != path:
            raise PermissionError("符号链接目标不在允许范围内")
        try:
            path.resolve().relative_to(self.root)
        except ValueError as error:
            raise PermissionError("路径越界") from error
        return path

    @staticmethod
    def _validate_arguments(schema: dict[str, object], arguments: dict[str, object]) -> str | None:
        if not isinstance(arguments, dict):
            return "arguments 必须是 JSON object"
        required = schema.get("required", [])
        for name in required if isinstance(required, list) else []:
            if name not in arguments:
                return f"缺少必填参数：{name}"
        if schema.get("additionalProperties") is False:
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                extra = set(arguments) - set(properties)
                if extra:
                    return f"存在未登记参数：{sorted(extra)}"
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for name, value in arguments.items():
                rule = properties.get(name)
                if isinstance(rule, dict) and rule.get("type") == "string":
                    if not isinstance(value, str):
                        return f"参数 {name} 必须是 string"
                    minimum_length = rule.get("minLength")
                    if isinstance(minimum_length, int) and len(value) < minimum_length:
                        return f"参数 {name} 不能为空"
                if isinstance(rule, dict) and rule.get("type") == "integer":
                    if not isinstance(value, int) or isinstance(value, bool):
                        return f"参数 {name} 必须是 integer"
                    minimum = rule.get("minimum")
                    maximum = rule.get("maximum")
                    if isinstance(minimum, int) and value < minimum:
                        return f"参数 {name} 小于最小值"
                    if isinstance(maximum, int) and value > maximum:
                        return f"参数 {name} 大于最大值"
        return None


def _serialize_hit(item: Any) -> dict[str, object]:
    chunk = item.chunk
    return {
        "relative_path": chunk.relative_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
        "symbol_path": chunk.symbol_path,
        "score": item.score,
        "rank": item.rank,
        "untrusted": True,
    }


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:20]


def _unified_diff(relative: str, old: str, new: str) -> str:
    import difflib

    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=relative,
            tofile=relative,
        )
    )
