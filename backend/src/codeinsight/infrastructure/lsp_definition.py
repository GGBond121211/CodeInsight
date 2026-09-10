"""受固定注册表约束的只读 LSP definition 查询。"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from codeinsight.domain.code_intelligence import (
    AVAILABLE,
    INVALID_RESPONSE,
    NOT_CONFIGURED,
    TIMEOUT,
    UNSUPPORTED_LANGUAGE,
    CodeIntelligenceResult,
    CodeLocation,
)
from codeinsight.infrastructure.lsp_registry import FixedLspRegistry, language_for_path
from codeinsight.ingestion.path_policy import safe_path, safe_relative_path
from codeinsight.retrieval.repository_map import repository_fingerprint


class LspDefinitionService:
    """每次请求启停一个固定 LSP server，避免跨请求残留进程状态。"""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        registry: FixedLspRegistry | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root 必须是目录")
        if timeout_seconds <= 0:
            raise ValueError("LSP timeout_seconds 必须为正")
        self.root = root
        self.registry = registry or FixedLspRegistry()
        self.timeout_seconds = timeout_seconds

    def definition(
        self,
        *,
        path: str,
        line: int,
        column: int,
        language: str | None = None,
    ) -> CodeIntelligenceResult:
        relative = safe_relative_path(path)
        source_path = safe_path(self.root, relative)
        if not source_path.is_file():
            raise FileNotFoundError(relative)
        detected = language_for_path(relative)
        selected_language = (language or detected or "").strip().lower()
        if not selected_language:
            return CodeIntelligenceResult(
                UNSUPPORTED_LANGUAGE,
                message=f"文件类型未登记 LSP 语言：{relative.rsplit('.', 1)[-1]}",
            )
        if line < 1 or line > len(source_path.read_text(encoding="utf-8").splitlines()):
            return CodeIntelligenceResult(INVALID_RESPONSE, message="LSP 位置超出文件范围")
        resolution = self.registry.resolve(selected_language)
        if resolution.status != AVAILABLE:
            return CodeIntelligenceResult(
                resolution.status,
                message=resolution.message,
                metadata={"language": selected_language},
            )
        try:
            text = source_path.read_text(encoding="utf-8")
        except UnicodeError:
            return CodeIntelligenceResult(INVALID_RESPONSE, message="LSP 目标文件不是 UTF-8 文本")
        try:
            locations, server_version = _query_definition(
                self.root,
                relative,
                text,
                line=line,
                column=column,
                language=selected_language,
                command=resolution.command,
                timeout_seconds=self.timeout_seconds,
            )
        except _LspTimeout:
            return CodeIntelligenceResult(
                TIMEOUT,
                message="LSP definition 请求超时",
                metadata={"language": selected_language, "server": resolution.server_name},
            )
        except (FileNotFoundError, OSError):
            return CodeIntelligenceResult(
                NOT_CONFIGURED,
                message=f"固定 LSP server 无法启动：{resolution.server_name}",
                metadata={"language": selected_language},
            )
        except _LspInvalidResponse as error:
            return CodeIntelligenceResult(
                INVALID_RESPONSE,
                message=str(error),
                metadata={"language": selected_language, "server": resolution.server_name},
            )
        return CodeIntelligenceResult(
            AVAILABLE,
            locations=tuple(
                _with_identity(item, self.root, source="lsp") for item in locations
            ),
            metadata={
                "language": selected_language,
                "server": resolution.server_name,
                "server_version": server_version,
                "repo_id": _repository_id(self.root),
                "repo_fingerprint": repository_fingerprint(self.root),
            },
        )


class _LspTimeout(Exception):
    pass


class _LspInvalidResponse(Exception):
    pass


def _query_definition(
    root: Path,
    relative: str,
    text: str,
    *,
    line: int,
    column: int,
    language: str,
    command: tuple[str, ...],
    timeout_seconds: float,
) -> tuple[tuple[CodeLocation, ...], str | None]:
    process = subprocess.Popen(
        list(command),
        cwd=str(root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    messages: queue.Queue[dict[str, object] | None] = queue.Queue()
    reader = threading.Thread(
        target=_read_messages,
        args=(process.stdout, messages),
        daemon=True,
    )
    reader.start()
    try:
        _send_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": root.as_uri(),
                    "capabilities": {},
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                },
            },
        )
        initialized = _wait_for_response(messages, 1, timeout_seconds)
        if "error" in initialized or not isinstance(initialized.get("result"), dict):
            raise _LspInvalidResponse("LSP initialize 返回格式无效")
        server_info = initialized["result"].get("serverInfo")
        server_version = (
            server_info.get("version")
            if isinstance(server_info, dict) and isinstance(server_info.get("version"), str)
            else None
        )
        _send_message(
            process,
            {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        )
        document_uri = (root / relative).as_uri()
        _send_message(
            process,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": document_uri,
                        "languageId": language,
                        "version": 1,
                        "text": text,
                    }
                },
            },
        )
        _send_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "textDocument/definition",
                "params": {
                    "textDocument": {"uri": document_uri},
                    "position": {"line": line - 1, "character": column},
                },
            },
        )
        response = _wait_for_response(messages, 2, timeout_seconds)
        if "error" in response:
            raise _LspInvalidResponse("LSP definition 返回错误")
        return _parse_locations(response.get("result"), root), server_version
    finally:
        _terminate(process)


def _send_message(process: subprocess.Popen[bytes], payload: dict[str, object]) -> None:
    if process.stdin is None:
        raise _LspInvalidResponse("LSP stdin 不可用")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    process.stdin.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
    process.stdin.write(encoded)
    process.stdin.flush()


def _wait_for_response(
    messages: queue.Queue[dict[str, object] | None], request_id: int, timeout_seconds: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _LspTimeout()
        try:
            message = messages.get(timeout=remaining)
        except queue.Empty as error:
            raise _LspTimeout() from error
        if message is None:
            raise _LspInvalidResponse("LSP server 提前退出")
        if message.get("id") == request_id:
            return message


def _read_messages(stream, messages: queue.Queue[dict[str, object] | None]) -> None:
    if stream is None:
        messages.put(None)
        return
    try:
        while True:
            headers: dict[str, str] = {}
            while True:
                line = stream.readline()
                if not line:
                    messages.put(None)
                    return
                if line in {b"\r\n", b"\n"}:
                    break
                name, separator, value = line.decode("ascii", errors="strict").partition(":")
                if separator:
                    headers[name.strip().lower()] = value.strip()
            raw_length = headers.get("content-length")
            if raw_length is None:
                messages.put(None)
                return
            try:
                length = int(raw_length)
            except ValueError:
                messages.put(None)
                return
            payload = stream.read(length)
            if len(payload) != length:
                messages.put(None)
                return
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, dict):
                messages.put(None)
                return
            messages.put(value)
    except (OSError, UnicodeError, json.JSONDecodeError):
        messages.put(None)


def _parse_locations(value: object, root: Path) -> tuple[CodeLocation, ...]:
    if value is None:
        return ()
    raw_items = [value] if isinstance(value, dict) else value if isinstance(value, list) else None
    if raw_items is None:
        raise _LspInvalidResponse("LSP definition result 不是 Location 数组")
    locations: list[CodeLocation] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise _LspInvalidResponse("LSP Location 不是 object")
        uri = item.get("uri") or item.get("targetUri")
        range_value = item.get("range") or item.get("targetSelectionRange")
        if not isinstance(uri, str) or not isinstance(range_value, dict):
            raise _LspInvalidResponse("LSP Location 缺少 uri 或 range")
        relative = _uri_to_relative(uri, root)
        if relative is None:
            continue
        start = range_value.get("start")
        end = range_value.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise _LspInvalidResponse("LSP range 缺少 start/end")
        start_line = start.get("line")
        start_column = start.get("character")
        end_line = end.get("line")
        end_column = end.get("character")
        if not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in (start_line, start_column, end_line, end_column)
        ):
            raise _LspInvalidResponse("LSP range 行列号无效")
        locations.append(
            CodeLocation(
                relative,
                start_line + 1,
                start_column,
                end_line=end_line + 1,
                end_column=end_column,
            )
        )
    return tuple(locations)


def _with_identity(
    location: CodeLocation, root: Path, *, source: str
) -> CodeLocation:
    repo_id = _repository_id(root)
    return CodeLocation(
        path=location.path,
        start_line=location.start_line,
        start_column=location.start_column,
        end_line=location.end_line,
        end_column=location.end_column,
        symbol_id=location.symbol_id,
        source=source,
        repo_id=repo_id,
        repo_fingerprint=repository_fingerprint(root),
    )


def _repository_id(root: Path) -> str:
    import hashlib

    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]


def _uri_to_relative(uri: str, root: Path) -> str | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    raw_path = unquote(parsed.path)
    if os.name == "nt" and raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
        raw_path = raw_path[1:]
    candidate = Path(raw_path)
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass
