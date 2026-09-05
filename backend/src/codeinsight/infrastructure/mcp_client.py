"""MCP stdio Client：让正常 Tool Loop 真正经过独立 Server 进程。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from codeinsight.agent.tool_loop import ToolCall, ToolResult
from codeinsight.infrastructure.otel import get_telemetry


class StdioMCPClient:
    def __init__(
        self,
        repository_root: str | Path,
        *,
        python_executable: str | None = None,
        server_module: str = "codeinsight.infrastructure.mcp_server",
    ) -> None:
        self.repository_root = str(Path(repository_root).resolve())
        self.python_executable = python_executable or sys.executable
        self.server_module = server_module
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._stdin: TextIO | None = None
        self._stdout: TextIO | None = None

    def __enter__(self) -> StdioMCPClient:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            [self.python_executable, "-m", self.server_module, "--root", self.repository_root],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        initialized = self._request(
            "initialize", {"clientInfo": {"name": "codeinsight-host", "version": "step5"}}
        )
        if "error" in initialized:
            raise RuntimeError("MCP initialize 失败")
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        self._stdin = None
        self._stdout = None

    def list_tools(self) -> tuple[dict[str, object], ...]:
        response = self._request("tools/list", {})
        if "error" in response:
            raise RuntimeError("MCP tools/list 失败")
        result = response.get("result", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        if not isinstance(tools, list):
            raise RuntimeError("MCP tools/list 返回格式无效")
        return tuple(item for item in tools if isinstance(item, dict))

    def call_tool(self, call: ToolCall) -> ToolResult:
        try:
            response = self._request("tools/call", {"name": call.name, "arguments": call.arguments})
        except (OSError, RuntimeError, ValueError):
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "MCP Server 不可用")
        if "error" in response:
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "MCP Server 返回协议错误")
        result = response.get("result", {})
        content = result.get("content", []) if isinstance(result, dict) else []
        text = content[0].get("text", "") if content and isinstance(content[0], dict) else ""
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "MCP ToolResult 不是有效 JSON")
        if payload.get("ok"):
            return ToolResult.success(
                call.id,
                call.name,
                payload.get("data", {}),
                state_fingerprint=payload.get("state_fingerprint"),
            )
        error = payload.get("error", {})
        return ToolResult.failure(
            call.id,
            call.name,
            str(error.get("code", "UNKNOWN")),
            str(error.get("message", "工具执行失败")),
            data=payload.get("data", {}),
            state_fingerprint=payload.get("state_fingerprint"),
        )

    def list_resources(self) -> tuple[dict[str, object], ...]:
        response = self._request("resources/list", {})
        result = response.get("result", {})
        resources = result.get("resources", []) if isinstance(result, dict) else []
        return tuple(item for item in resources if isinstance(item, dict))

    def read_resource(self, uri: str) -> str:
        response = self._request("resources/read", {"uri": uri})
        result = response.get("result", {})
        contents = result.get("contents", []) if isinstance(result, dict) else []
        if not contents or not isinstance(contents[0], dict):
            raise RuntimeError("MCP resource 返回为空")
        return str(contents[0].get("text", ""))

    def list_prompts(self) -> tuple[dict[str, object], ...]:
        response = self._request("prompts/list", {})
        result = response.get("result", {})
        prompts = result.get("prompts", []) if isinstance(result, dict) else []
        return tuple(item for item in prompts if isinstance(item, dict))

    def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> str:
        response = self._request("prompts/get", {"name": name, "arguments": arguments or {}})
        result = response.get("result", {})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages or not isinstance(messages[0], dict):
            raise RuntimeError("MCP prompt 返回为空")
        content = messages[0].get("content", {})
        return str(content.get("text", "")) if isinstance(content, dict) else str(content)

    def _notify(self, method: str, params: dict[str, object]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        with get_telemetry().span("mcp", method):
            request_id = self._next_id
            self._next_id += 1
            self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            if self._stdout is None:
                raise RuntimeError("MCP Client 未启动")
            line = self._stdout.readline()
            if not line:
                raise RuntimeError("MCP Server 已退出")
            response = json.loads(line)
            if not isinstance(response, dict):
                raise RuntimeError("MCP 响应不是 object")
            return response

    def _write(self, payload: dict[str, object]) -> None:
        if self._stdin is None:
            raise RuntimeError("MCP Client 未启动")
        self._stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._stdin.flush()
