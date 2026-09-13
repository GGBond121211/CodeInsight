"""Q-012：把工具目录与问题原型库导出成只读的 YAML。

导出物不是配置源：改这些文件不会改变任何工具行为。CI 里有同步测试断言
「文件内容 == 代码导出的内容」，所以漂移会直接让测试失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from codeinsight.application.archetype_library import (  # noqa: E402
    assert_library_is_read_only,
    export_library_text,
)
from codeinsight.infrastructure.tool_registry import (  # noqa: E402
    build_default_registry,
)

CATALOG = ROOT / "backend" / "src" / "codeinsight" / "tools" / "catalog.yaml"
ARCHETYPES = ROOT / "experiments" / "configs" / "archetypes_v1.yaml"


def main() -> int:
    assert_library_is_read_only()
    _write(CATALOG, build_default_registry().export_catalog_text())
    _write(ARCHETYPES, export_library_text())
    return 0


def _write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text(encoding="utf-8") if target.exists() else None
    target.write_text(text, encoding="utf-8", newline="\n")
    print("updated" if existing != text else "unchanged", target)


if __name__ == "__main__":
    raise SystemExit(main())
