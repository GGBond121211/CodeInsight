"""MCP stdio Client：让正常 Tool Loop 真正经过独立 Server 进程。"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
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
        timeout_seconds: float = 30.0,
    ) -> None:
        self.repository_root = str(Path(repository_root).resolve())
        self.python_executable = python_executable or sys.executable
        self.server_module = server_module
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._request_lock = threading.Lock()
        self._stdin: TextIO | None = None
        self._stdout: TextIO | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()

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
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._responses = queue.Queue()
        threading.Thread(
            target=_read_responses, args=(self._stdout, self._responses), daemon=True
        ).start()
        try:
            initialized = self._request(
                "initialize", {"clientInfo": {"name": "codeinsight-host", "version": "step5"}}
            )
            if "error" in initialized:
                raise RuntimeError("MCP initialize 失败")
            self._notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        self._process = None
        self._stdin = None
        self._stdout = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=2)
        if process.stdout:
            process.stdout.close()

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
        if not isinstance(content, list):
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "MCP content 格式无效")
        text = content[0].get("text", "") if content and isinstance(content[0], dict) else ""
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "MCP ToolResult 不是有效 JSON")
        if not isinstance(payload, dict) or not isinstance(payload.get("data", {}), dict):
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "MCP ToolResult 格式无效")
        if payload.get("ok"):
            return ToolResult.success(
                call.id,
                call.name,
                payload.get("data", {}),
                state_fingerprint=payload.get("state_fingerprint"),
            )
        error = payload.get("error", {})
        if not isinstance(error, dict):
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "MCP error 格式无效")
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
        with self._request_lock:
            with get_telemetry().span("mcp", method):
                request_id = self._next_id
                self._next_id += 1
                try:
                    self._write(
                        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
                    )
                    if self._stdout is None:
                        raise RuntimeError("MCP Client 未启动")
                    line = self._responses.get(timeout=self.timeout_seconds)
                    if not line:
                        raise RuntimeError("MCP Server 已退出")
                    response = json.loads(line)
                    if not isinstance(response, dict):
                        raise RuntimeError("MCP 响应不是 object")
                    if response.get("id") != request_id:
                        raise RuntimeError("MCP 响应 ID 与请求不匹配")
                    return response
                except queue.Empty as error:
                    self.close()
                    raise RuntimeError("MCP 响应超时，连接已关闭") from error
                except (OSError, RuntimeError, ValueError):
                    self.close()
                    raise

    def _write(self, payload: dict[str, object]) -> None:
        if self._stdin is None:
            raise RuntimeError("MCP Client 未启动")
        self._stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._stdin.flush()


def _read_responses(stream: TextIO, responses: queue.Queue[str | None]) -> None:
    """每个进程一个读取线程，不因单次请求超时不断创建线程。"""
    try:
        for line in stream:
            responses.put(line)
    except (OSError, ValueError):
        pass
    finally:
        responses.put(None)
