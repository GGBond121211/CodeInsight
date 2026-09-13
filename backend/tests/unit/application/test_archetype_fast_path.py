"""Q-012 U2：问题原型路由与快路径的契约测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codeinsight.agent.tool_loop import (
    ToolCall,
    ToolModelResponse,
    ToolResult,
)
from codeinsight.application.archetype_fast_path import (
    FAST_PATH_ENV,
    fast_path_enabled,
    run_archetype_fast_path,
)
from codeinsight.application.archetype_library import (
    ARCHETYPES_V1,
    assert_library_is_read_only,
    export_library_text,
)
from codeinsight.application.archetype_router import (
    ArchetypeMatch,
    ArchetypeRouter,
    ArchetypeRouterConfig,
)
from codeinsight.domain.archetype import QuestionArchetype, RecipeStep
from codeinsight.infrastructure.tool_registry import build_default_registry


class FakeEmbedder:
    """只回放一张表；表里没有的文本直接失败，避免测试悄悄走别的路径。"""

    def __init__(self, table: dict[str, tuple[float, ...]]) -> None:
        self.table = table
        self.calls = 0
        self.texts = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return SimpleNamespace(vectors=tuple(self.table[text] for text in texts))


def _archetype(name: str, examples: tuple[str, ...]) -> QuestionArchetype:
    return QuestionArchetype(
        name=name,
        summary=f"{name} 的说明",
        example_questions=examples,
        recipe=(RecipeStep("search_repository", {"question": "{question}"}),),
        answer_outline="给出位置并说明依据。",
    )


LIBRARY = (
    _archetype("alpha", ("a1", "a2", "a3", "a4", "a5")),
    _archetype("beta", ("b1", "b2", "b3", "b4", "b5")),
)


def _router_table(**extra: tuple[float, ...]) -> dict[str, tuple[float, ...]]:
    table: dict[str, tuple[float, ...]] = {}
    for index, question in enumerate(("a1", "a2", "a3", "a4", "a5")):
        table[question] = (1.0, 0.0, index * 0.01)
    for index, question in enumerate(("b1", "b2", "b3", "b4", "b5")):
        table[question] = (0.0, 1.0, index * 0.01)
    table.update(extra)
    return table


def test_question_close_to_one_archetype_is_routed():
    embedder = FakeEmbedder(_router_table(q=(1.0, 0.0, 0.0)))
    router = ArchetypeRouter(embed=embedder, library=LIBRARY)

    match = router.route("q")

    assert isinstance(match, ArchetypeMatch)
    assert match.archetype.name == "alpha"
    assert match.score > 0.99
    assert match.margin > 0.0


def test_unrelated_question_falls_back_instead_of_guessing():
    embedder = FakeEmbedder(_router_table(q=(0.0, 0.0, 1.0)))
    router = ArchetypeRouter(embed=embedder, library=LIBRARY)

    assert router.route("q") is None


def test_ambiguous_question_falls_back_when_two_archetypes_tie():
    table = _router_table(q=(1.0, 1.0, 0.0))
    router = ArchetypeRouter(embed=FakeEmbedder(table), library=LIBRARY)

    # 两个原型的得分几乎一样时宁可回退：错命中比未命中更贵。
    assert router.route("q") is None


def test_example_vectors_are_cached_across_questions():
    embedder = FakeEmbedder(_router_table(q=(1.0, 0.0, 0.0), q2=(0.99, 0.0, 0.0)))
    router = ArchetypeRouter(embed=embedder, library=LIBRARY)

    assert router.route("q") is not None
    assert router.route("q2") is not None

    # 两次路由只算一遍示例问题向量：3 次调用 = 1 次示例 + 2 次问题。
    assert embedder.calls == 3
    assert embedder.texts == 10 + 2


def test_router_config_is_validated():
    with pytest.raises(ValueError):
        ArchetypeRouterConfig(threshold=0.0)
    with pytest.raises(ValueError):
        ArchetypeRouterConfig(margin=-0.1)


class FakeHost:
    def __init__(self, results: dict[str, ToolResult]) -> None:
        self.tools = build_default_registry().list_tools()
        self.results = results
        self.calls: list[ToolCall] = []

    def list_tools(self):
        return self.tools

    def call_tool(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return self.results.get(call.name, ToolResult.failure(
            call.id, call.name, "NOT_FOUND", "测试未准备该工具",
        ))


class ContractModel:
    """只回一次字：快路径允许它做的唯一一件事。"""

    def __init__(self, payload: str | None, *, with_tool_call: bool = False) -> None:
        self.payload = payload
        self.with_tool_call = with_tool_call
        self.calls: list[tuple[object, ...]] = []
        self.messages: list[tuple[dict[str, object], ...]] = []

    def complete_with_tools(self, messages, tools):
        self.calls.append(tuple(tools))
        self.messages.append(tuple(messages))
        if self.payload is None:
            return ToolModelResponse(None, (), "fake", 10, 0)
        if self.with_tool_call:
            return ToolModelResponse(
                None,
                (ToolCall("c1", "read_file", {"path": "src/a.py"}),),
                "fake",
                10,
                0,
            )
        return ToolModelResponse(self.payload, (), "fake", 120, 30)


def _search_and_read_results() -> dict[str, ToolResult]:
    return {
        "search_repository": ToolResult.success(
            "c1",
            "search_repository",
            {
                "results": [
                    {
                        "relative_path": "src/app.py",
                        "start_line": 10,
                        "end_line": 20,
                        "text": "def send(): pass",
                    }
                ]
            },
        ),
        "read_file": ToolResult.success(
            "c2",
            "read_file",
            {
                "path": "src/app.py",
                "start_line": 10,
                # 行区间必须与检索命中不同：同一 (path, start, end) 出现两个
                # 不同指纹会被评估器判成「索引与工作区不同步」（矛盾证据），
                # 快路径据此回退——那是对的，但不是这条用例要测的东西。
                # 换成一个更宽的范围，才会真的多出一条可引用证据。
                "end_line": 25,
                "text": "def send(self):\n    return self._send()",
            },
        ),
    }


def test_fast_path_returns_the_same_contract_as_the_tool_loop():
    from codeinsight.application.archetype_library import ARCHETYPES_V1 as real_library

    archetype = real_library[0]
    host = FakeHost(_search_and_read_results())
    model = ContractModel(
        '{"outcome": "answered", "answer": "send 定义在 src/app.py:10。",'
        ' "citations": ["E1", "E2"]}'
    )

    result = run_archetype_fast_path(
        archetype=archetype,
        question="send 方法在哪里定义？",
        mcp_client=host,
        model=model,
    )

    assert result is not None
    assert result.fast_path is True
    assert result.archetype == archetype.name
    assert result.status == "ANSWERED"
    assert [item.evidence_id for item in result.citations] == ["E1", "E2"]
    assert result.citations[0].relative_path == "src/app.py"
    assert result.structured_evidence and result.structured_evidence[0].path == "src/app.py"
    # 快路径只允许一次模型调用，并且不向模型提供任何工具。
    assert model.calls == [()]


def test_fast_path_falls_back_when_evidence_is_missing():
    host = FakeHost({})  # 每个工具都失败
    model = ContractModel('{"outcome": "answered", "answer": "x", "citations": ["E1"]}')

    result = run_archetype_fast_path(
        archetype=ARCHETYPES_V1[0],
        question="send 方法在哪里定义？",
        mcp_client=host,
        model=model,
    )

    assert result is None
    assert model.calls == []  # 证据不足时根本不调模型


def test_fast_path_rejects_citations_it_cannot_map():
    host = FakeHost(_search_and_read_results())
    model = ContractModel(
        '{"outcome": "answered", "answer": "x", "citations": ["E99"]}'
    )

    result = run_archetype_fast_path(
        archetype=ARCHETYPES_V1[0],
        question="send 方法在哪里定义？",
        mcp_client=host,
        model=model,
    )

    assert result is None




def test_fast_path_falls_back_when_the_answer_is_not_json():
    """模型返回的正文不是合法答案契约时回退，不能把解析异常抛给调用方。"""

    host = FakeHost(_search_and_read_results())
    model = ContractModel("这不是 JSON")
    events: list[dict[str, str]] = []

    result = run_archetype_fast_path(
        archetype=ARCHETYPES_V1[0],
        question="send 方法在哪里定义？",
        mcp_client=host,
        model=model,
        emit=lambda kind, payload: events.append(dict(payload)),
    )

    assert result is None
    assert events[-1]["reason"] == "invalid_answer"
    # 默认允许一次纠正：两次都不合法才回退，事件里保留重试理由便于统计。
    assert model.calls == [(), ()]
    assert [event["reason"] for event in events] == ["retry:invalid_answer", "invalid_answer"]
def test_fast_path_gives_up_when_the_model_wants_more_tools():
    host = FakeHost(_search_and_read_results())
    model = ContractModel(None, with_tool_call=True)

    result = run_archetype_fast_path(
        archetype=ARCHETYPES_V1[0],
        question="send 方法在哪里定义？",
        mcp_client=host,
        model=model,
    )

    assert result is None


def test_fast_path_is_off_by_default():
    assert fast_path_enabled({}) is False
    assert fast_path_enabled({FAST_PATH_ENV: ""}) is False
    assert fast_path_enabled({FAST_PATH_ENV: "off"}) is False
    assert fast_path_enabled({FAST_PATH_ENV: "on"}) is True
    assert fast_path_enabled({FAST_PATH_ENV: "TRUE"}) is True


def test_built_in_library_is_read_only_and_unique():
    assert_library_is_read_only()
    names = [item.name for item in ARCHETYPES_V1]
    assert len(names) == len(set(names))
    assert all(len(item.example_questions) >= 5 for item in ARCHETYPES_V1)


def test_recipe_placeholder_whitelist_is_enforced():
    with pytest.raises(ValueError):
        RecipeStep("read_file", {"path": "{not_allowed}"})
    with pytest.raises(ValueError):
        RecipeStep("", {})
    with pytest.raises(ValueError):
        QuestionArchetype(
            name="x",
            summary="s",
            example_questions=("one",),
            recipe=(RecipeStep("read_file", {}),),
            answer_outline="o",
        )


def test_exported_library_matches_the_code():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / ".."
        / "experiments"
        / "configs"
        / "archetypes_v1.yaml"
    ).resolve()
    assert path.exists(), f"缺少导出物：{path}"
    assert path.read_text(encoding="utf-8") == export_library_text()


def test_fast_path_shows_the_model_the_evidence_text_not_only_ids():
    """快路径必须把证据正文片段交给模型：只给编号时内容问题只能靠记忆作答。"""

    host = FakeHost(_search_and_read_results())
    model = ContractModel(
        '{"outcome": "answered", "answer": "send 定义在 src/app.py:10。",'
        ' "citations": ["E1"]}'
    )

    result = run_archetype_fast_path(
        archetype=ARCHETYPES_V1[0],
        question="send 方法在哪里定义？",
        mcp_client=host,
        model=model,
    )

    assert result is not None
    joined = "\n".join(str(item["content"]) for item in model.messages[-1])
    assert "def send(): pass" in joined  # 检索命中的正文片段
    assert "def send(self):" in joined  # read_file 的正文片段
    assert "citations 只能使用这些编号" in joined


class ScriptedModel:
    """按脚本依次返回多份正文；用来测「纠正一次之后成功」。"""

    def __init__(self, payloads: tuple[str, ...]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[object, ...]] = []
        self.messages: list[tuple[dict[str, object], ...]] = []

    def complete_with_tools(self, messages, tools):
        self.calls.append(tuple(tools))
        self.messages.append(tuple(messages))
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return ToolModelResponse(self.payloads[index], (), "fake", 120, 30)


def _two_path_results() -> dict[str, ToolResult]:
    """两条不同路径的检索结果：用来构造「正文说 A、引用给 B」的不一致。"""

    return {
        "search_repository": ToolResult.success(
            "c1",
            "search_repository",
            {
                "results": [
                    {
                        "relative_path": "src/app.py",
                        "start_line": 10,
                        "end_line": 20,
                        "text": "def send(): pass",
                    },
                    {
                        "relative_path": "src/other.py",
                        "start_line": 30,
                        "end_line": 40,
                        "text": "class Client: pass",
                    },
                ]
            },
        ),
    }


def _search_only_archetype() -> QuestionArchetype:
    """只有一步检索的配方：两条路径的用例不需要 read_file 参与。"""

    return _archetype("alpha", ("a1", "a2", "a3", "a4", "a5"))


def test_fast_path_retries_once_when_the_answer_is_not_json() -> None:
    host = FakeHost(_search_and_read_results())
    model = ScriptedModel((
        "这不是 JSON",
        '{"outcome": "answered", "answer": "send 定义在 src/app.py:10。", "citations": ["E1"]}',
    ))
    events: list[dict[str, str]] = []

    result = run_archetype_fast_path(
        archetype=ARCHETYPES_V1[0],
        question="send 方法在哪里定义？",
        mcp_client=host,
        model=model,
        emit=lambda kind, payload: events.append(dict(payload)),
    )

    assert result is not None
    assert len(model.calls) == 2
    assert [event["reason"] for event in events] == ["retry:invalid_answer", "hit"]
    # 纠正指令是追加的一条 user 消息，不是改写原始问题。
    assert len(model.messages[1]) == len(model.messages[0]) + 1


def test_fast_path_retries_when_the_answer_names_a_file_it_did_not_cite() -> None:
    """复测发现的真实缺陷：正文写对了文件，引用却挂在另一条证据上。"""

    host = FakeHost(_two_path_results())
    model = ScriptedModel((
        '{"outcome": "answered", "answer": "Client 定义在 src/other.py。", "citations": ["E1"]}',
        '{"outcome": "answered", "answer": "Client 定义在 src/other.py。", "citations": ["E2"]}',
    ))
    events: list[dict[str, str]] = []

    result = run_archetype_fast_path(
        archetype=_search_only_archetype(),
        question="Client 类定义在哪个文件？",
        mcp_client=host,
        model=model,
        emit=lambda kind, payload: events.append(dict(payload)),
    )

    assert result is not None
    assert [item.evidence_id for item in result.citations] == ["E2"]
    assert [event["reason"] for event in events] == ["retry:citation_path_mismatch", "hit"]
    assert "src/other.py" in str(model.messages[1][-1]["content"])


def test_fast_path_falls_back_when_the_citation_never_matches() -> None:
    host = FakeHost(_two_path_results())
    model = ScriptedModel(
        ('{"outcome": "answered", "answer": "Client 定义在 src/other.py。", "citations": ["E1"]}',)
    )
    events: list[dict[str, str]] = []

    result = run_archetype_fast_path(
        archetype=_search_only_archetype(),
        question="Client 类定义在哪个文件？",
        mcp_client=host,
        model=model,
        emit=lambda kind, payload: events.append(dict(payload)),
    )

    # 改不对就回退 Tool Loop：宁可多花一次检索，也不能给「看起来有引用」的答案。
    assert result is None
    assert [event["reason"] for event in events] == [
        "retry:citation_path_mismatch",
        "citation_path_mismatch",
    ]


def test_fast_path_attempt_budget_can_be_turned_off(monkeypatch) -> None:
    from codeinsight.application.archetype_fast_path import (
        FAST_PATH_ATTEMPTS_ENV,
        fast_path_answer_attempts,
    )

    monkeypatch.setenv(FAST_PATH_ATTEMPTS_ENV, "1")
    assert fast_path_answer_attempts() == 1
    monkeypatch.setenv(FAST_PATH_ATTEMPTS_ENV, "99")
    assert fast_path_answer_attempts() == 4
    monkeypatch.setenv(FAST_PATH_ATTEMPTS_ENV, "abc")
    assert fast_path_answer_attempts() == 2
    monkeypatch.setenv(FAST_PATH_ATTEMPTS_ENV, "1")

    host = FakeHost(_search_and_read_results())
    model = ScriptedModel(("这不是 JSON",))

    assert (
        run_archetype_fast_path(
            archetype=ARCHETYPES_V1[0],
            question="send 方法在哪里定义？",
            mcp_client=host,
            model=model,
        )
        is None
    )
    assert len(model.calls) == 1
