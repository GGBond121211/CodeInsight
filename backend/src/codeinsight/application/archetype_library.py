"""问题原型库（Q-012 U2）。

原型是数据，但**不放在 YAML 里加载**：配方里的工具名必须与只读白名单对齐，
这类校验只能在代码里做。YAML 是导出物，供人工浏览与漂移检查。

六个原型都来自 `docs/Q12_BASELINE.md` 的首个工具分布与 Q-008 的真实轮次：
真实运行里模型几乎总是先 `get_repository_map` 或 `search_repository`，然后
再重复一次检索；「定义在哪」「谁调用它」这类问题本可以一次到位。
"""

from __future__ import annotations

from codeinsight.agent.code_understanding_tool_loop import CODE_UNDERSTANDING_TOOLS
from codeinsight.domain.archetype import (
    LINE_PLACEHOLDER,
    PATH_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    QuestionArchetype,
    RecipeStep,
)
from codeinsight.infrastructure.yaml_export import dump_document

LIBRARY_VERSION = "q012-archetypes-v1"

ARCHETYPES_V1: tuple[QuestionArchetype, ...] = (
    QuestionArchetype(
        name="definition_lookup",
        summary="某个符号或函数定义在哪里",
        example_questions=(
            "send 方法在哪里定义？",
            "Client 类定义在哪个文件？",
            "where is parse_model_answer defined?",
            "这个函数定义在哪个模块？",
            "resolve_route_budget 是怎么实现的？",
            "哪个文件声明了这个常量？",
        ),
        recipe=(
            RecipeStep("search_repository", {"question": QUESTION_PLACEHOLDER}),
            RecipeStep(
                "read_file",
                {"path": PATH_PLACEHOLDER, "start_line": LINE_PLACEHOLDER},
            ),
        ),
        answer_outline="先给定义所在文件与行号，再简述它做什么；行号必须来自证据。",
    ),
    QuestionArchetype(
        name="call_sites",
        summary="谁调用了某个符号、它在哪些地方被引用",
        example_questions=(
            "谁调用了 ConversationService？",
            "这个函数在哪些地方被引用？",
            "find all call sites of parse_model_answer",
            "哪些模块用到了 EvidenceLedger？",
            "这个类还有别的地方在用吗？",
        ),
        recipe=(
            RecipeStep("search_repository", {"question": QUESTION_PLACEHOLDER}),
            RecipeStep(
                "read_file",
                {"path": PATH_PLACEHOLDER, "start_line": LINE_PLACEHOLDER},
            ),
        ),
        answer_outline="逐条列出引用位置（路径:行号）；索引不可用时必须说明「只覆盖已读到的证据」。",
    ),
    QuestionArchetype(
        name="module_inventory",
        summary="仓库或某个目录下有哪些模块、符号",
        example_questions=(
            "这个仓库的顶层模块有哪些？",
            "src 目录下都有哪些符号？",
            "what modules live under backend/src?",
            "整个项目的目录结构是什么？",
            "这个仓库有哪些入口文件？",
        ),
        recipe=(
            # 地图只做导航、不进证据台账，所以单靠它 + 一次 read_file 永远凑不满 2 条证据
            # （= 必然回退，实测 3/3）。改成先检索再读命中位置：检索本身会带回多条可引用
            # 证据，读一次给出确定行号。代价是答案只能覆盖「检索到的模块」，所以
            # answer_outline 明确要求不许声称列全。
            RecipeStep("search_repository", {"question": QUESTION_PLACEHOLDER}),
            RecipeStep("read_file", {"path": PATH_PLACEHOLDER}),
        ),
        answer_outline=(
            "只列出证据里真实出现的文件与符号，并说明这是检索命中的结果、不是完整清单；"
            "没有证据的目录不要提。"
        ),
    ),
    QuestionArchetype(
        name="config_source",
        summary="某个配置项从哪里读取、默认值是什么",
        example_questions=(
            "read_timeout 配置从哪里读取？",
            "这个超时默认值是多少？",
            "where is CODEINSIGHT_MODEL read?",
            "环境变量是在哪个文件解析的？",
            "这个开关默认打开还是关闭？",
        ),
        recipe=(
            RecipeStep("search_repository", {"question": QUESTION_PLACEHOLDER}),
            RecipeStep(
                "read_file",
                {"path": PATH_PLACEHOLDER, "start_line": LINE_PLACEHOLDER},
            ),
        ),
        answer_outline="给出读取位置与默认值，并区分「代码默认」与「部署时覆盖」。",
    ),
    QuestionArchetype(
        name="flow_trace",
        summary="一个请求从哪进入系统、按什么顺序流动",
        example_questions=(
            "这次请求从哪里进入系统？",
            "一次对话的完整链路是什么？",
            "how does a turn flow through the service?",
            "改了代码之后哪个进程会执行？",
            "从 HTTP 到数据库中间经过了哪些层？",
        ),
        recipe=(
            RecipeStep("search_repository", {"question": QUESTION_PLACEHOLDER}),
            RecipeStep("get_repository_map", {"symbol_query": "router", "max_symbols": 40}),
            RecipeStep(
                "read_file",
                {"path": PATH_PLACEHOLDER, "start_line": LINE_PLACEHOLDER},
            ),
        ),
        answer_outline="按调用顺序编号列出经过的模块与函数；顺序必须有证据支持，不靠常识补全。",
    ),
    QuestionArchetype(
        name="test_coverage_lookup",
        summary="某个行为有没有测试覆盖、测试在哪里",
        example_questions=(
            "这个函数的测试在哪里？",
            "哪些测试覆盖了审批流程？",
            "where are the tool loop tests?",
            "这个模块有没有集成测试？",
            "改这个文件要跑哪些测试？",
        ),
        recipe=(
            RecipeStep("search_repository", {"question": QUESTION_PLACEHOLDER}),
            RecipeStep(
                "read_file",
                {"path": PATH_PLACEHOLDER, "start_line": LINE_PLACEHOLDER},
            ),
        ),
        answer_outline="列出测试文件与用例名；没有找到就说没有找到，不用推理代替证据。",
    ),
)

LIBRARY_HEADER: tuple[str, ...] = (
    "CodeInsight 问题原型库（导出物，不是配置源）。",
    "",
    "这个文件由 codeinsight.application.archetype_library 生成，运行时不会读它。",
    "原型要改就改代码，然后重新生成：",
    "    python experiments/export_tool_catalog.py",
    "",
    "配方只允许只读工具；写工具永远不进入快路径。",
)


def export_library_text() -> str:
    document = {
        "library_version": LIBRARY_VERSION,
        "read_only_tools": sorted(CODE_UNDERSTANDING_TOOLS),
        "archetypes": [
            {
                "name": archetype.name,
                "version": archetype.version,
                "summary": archetype.summary,
                "answer_outline": archetype.answer_outline,
                "example_questions": list(archetype.example_questions),
                "recipe": [
                    {"tool": step.tool, "arguments": dict(step.arguments)}
                    for step in archetype.recipe
                ],
            }
            for archetype in ARCHETYPES_V1
        ],
    }
    return dump_document(LIBRARY_HEADER, document)


def assert_library_is_read_only() -> None:
    """启动期自检：任何原型引用了非只读工具就直接失败。"""

    allowed = frozenset(CODE_UNDERSTANDING_TOOLS)
    for archetype in ARCHETYPES_V1:
        if not archetype.uses_only(read_only_tools=allowed):
            offending = [step.tool for step in archetype.recipe if step.tool not in allowed]
            raise ValueError(f"{archetype.name} 的配方含非只读工具：{offending}")
