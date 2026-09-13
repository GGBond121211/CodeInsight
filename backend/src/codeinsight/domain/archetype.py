"""问题原型的领域模型（Q-012 U2）。

一个原型 = 一类高频问题 + 固定检索配方 + 提问示例。它回答的是「这类问题该按
哪几步取证」，不回答「答案是什么」——答案仍然由模型基于证据写，且只能引用
应用分配的 E 编号。

为什么配方只允许只读工具：原型快路径是「换个更省的取证顺序」，不是「放开
权限」。写工具仍然只在修改路径里、经过审批与隔离校验。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# 配方步骤里可用的模板变量。限制成白名单而不是自由格式化：一个花括号求值器
# 就是一处注入面，而这里只需要「问题文本」和「上一步的首个位置」。
QUESTION_PLACEHOLDER = "{question}"
PATH_PLACEHOLDER = "{path}"
LINE_PLACEHOLDER = "{line}"
COLUMN_PLACEHOLDER = "{column}"

ALLOWED_PLACEHOLDERS: frozenset[str] = frozenset(
    {QUESTION_PLACEHOLDER, PATH_PLACEHOLDER, LINE_PLACEHOLDER, COLUMN_PLACEHOLDER}
)


@dataclass(frozen=True)
class RecipeStep:
    """一步取证：调用哪个只读工具、参数怎么写。"""

    tool: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("配方步骤必须指定工具")
        for key, value in self.arguments.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("配方参数名不能为空")
            if isinstance(value, (list, tuple)):
                if not value or not all(isinstance(item, str) for item in value):
                    raise ValueError(f"配方参数 {key} 的列表只能装字符串")
                continue
            if not isinstance(value, (str, int, bool)):
                raise ValueError(f"配方参数 {key} 只支持标量或字符串列表")
            if isinstance(value, str) and value.startswith("{"):
                if value not in ALLOWED_PLACEHOLDERS:
                    raise ValueError(f"配方参数 {key} 使用了不允许的占位符：{value}")

    @property
    def placeholders(self) -> tuple[str, ...]:
        found: list[str] = []
        for value in self.arguments.values():
            if isinstance(value, str) and value in ALLOWED_PLACEHOLDERS:
                found.append(value)
        return tuple(found)


@dataclass(frozen=True)
class QuestionArchetype:
    """一类高频问题及其固定取证顺序。"""

    name: str
    summary: str
    example_questions: tuple[str, ...]
    recipe: tuple[RecipeStep, ...]
    answer_outline: str
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.summary.strip():
            raise ValueError("原型必须有名称和说明")
        if not self.recipe:
            raise ValueError(f"{self.name} 的配方不能为空")
        if len(self.example_questions) < 5:
            raise ValueError(f"{self.name} 至少需要 5 条示例问题")
        cleaned = [item.strip() for item in self.example_questions]
        if any(not item for item in cleaned):
            raise ValueError(f"{self.name} 的示例问题不能为空")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"{self.name} 的示例问题不能重复")
        object.__setattr__(self, "example_questions", tuple(cleaned))
        if not self.answer_outline.strip():
            raise ValueError(f"{self.name} 必须说明期望的回答形状")

    def uses_only(self, *, read_only_tools: frozenset[str]) -> bool:
        return all(step.tool in read_only_tools for step in self.recipe)
