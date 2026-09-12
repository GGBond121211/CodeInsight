"""确定性 YAML 导出器（Q-012）。

只做一件事：把嵌套的 dict / list / 标量写成 block YAML 文本。它不用
PyYAML，原因是「导出物」不需要解析能力——需要解析就会出现第二套读取路径，
而我们要避免的正是「配置说有、代码没有」的第二事实源。

为什么字符串一律加双引号：不加引号时 `yes`、`no`、`1.0`、`null`
这类值会被 YAML 解释成布尔/浮点/空值，导出物与代码里的真实取值就会
产生肉眼看不出的偏差。全量加引号让同一份文本只有一种读法。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _is_safe_key(key: str) -> bool:
    if not key:
        return False
    for index, character in enumerate(key):
        allowed = character.isalnum() or character in "_-."
        if index == 0 and not (character.isalpha() or character == "_"):
            return False
        if not allowed:
            return False
    return True


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return "null"
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _key(value: str) -> str:
    return value if _is_safe_key(value) else f'"{value}"'


def _is_scalar(value: object) -> bool:
    return not isinstance(value, (Mapping, str, bytes, Sequence)) or isinstance(
        value, (str, bytes)
    )


def dumps(value: object, *, indent: int = 0) -> str:
    """把嵌套结构写成 block YAML 文本（行尾不含多余空格）。"""

    prefix = " " * indent
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = _key(str(key))
            if _is_scalar(item):
                lines.append(f"{prefix}{name}: {_scalar(item)}")
            elif isinstance(item, Mapping):
                if not item:
                    lines.append(f"{prefix}{name}: {{}}")
                else:
                    lines.append(f"{prefix}{name}:")
                    lines.append(dumps(item, indent=indent + 2))
            else:
                items = list(item)  # type: ignore[arg-type]
                if not items:
                    lines.append(f"{prefix}{name}: []")
                else:
                    lines.append(f"{prefix}{name}:")
                    lines.append(dumps(items, indent=indent + 2))
        return "\n".join(lines)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if _is_scalar(item):
                lines.append(f"{prefix}- {_scalar(item)}")
            elif isinstance(item, Mapping):
                block = dumps(item, indent=indent + 2)
                block_lines = block.split("\n")
                first = block_lines[0].strip()
                lines.append(f"{prefix}- {first}")
                lines.extend(block_lines[1:])
            else:
                lines.append(f"{prefix}-")
                lines.append(dumps(list(item), indent=indent + 2))  # type: ignore[arg-type]
        return "\n".join(lines)
    return f"{prefix}{_scalar(value)}"


def dump_document(header_lines: Sequence[str], value: object) -> str:
    """带注释头的完整文件文本；注释头由调用方提供，确保导出物可被人读懂。"""

    comments = "\n".join(line if line.startswith("#") else f"# {line}" for line in header_lines)
    return f"{comments}\n\n{dumps(value)}\n"
