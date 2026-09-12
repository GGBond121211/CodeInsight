"""Q-012：把工具目录导出成只读的 catalog.yaml。

导出物不是配置源：改这个文件不会改变任何工具行为。CI 里有一条同步测试
断言「文件内容 == 代码导出的内容」，所以漂移会直接让测试失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from codeinsight.infrastructure.tool_registry import (  # noqa: E402
    build_default_registry,
)

TARGET = ROOT / "backend" / "src" / "codeinsight" / "tools" / "catalog.yaml"


def main() -> int:
    text = build_default_registry().export_catalog_text()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    existing = TARGET.read_text(encoding="utf-8") if TARGET.exists() else None
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    status = "updated" if existing != text else "unchanged"
    print(status, TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
