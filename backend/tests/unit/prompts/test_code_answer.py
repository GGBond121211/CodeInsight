from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.source import SourceChunk
from codeinsight.prompts.code_answer import PROMPT_VERSION, build_answer_prompt


def test_prompt_labels_evidence_with_path_and_lines() -> None:
    results = (RankedChunk(SourceChunk("src/app.py", 3, 5, "def run():\n    return 1"), 4.2, 1),)

    system_prompt, user_prompt = build_answer_prompt("Where is run defined?", results)

    assert PROMPT_VERSION == "code-answer-v2"
    assert '"citations":["E1"]' in system_prompt
    assert "非空的解释性句子" in system_prompt
    assert "包含支撑结论所需的全部实现代码块" in system_prompt
    assert "简要说明无法支持的部分" in system_prompt
    assert "[E1] src/app.py:3-5" in user_prompt
    assert "Where is run defined?" in user_prompt


def test_prompt_preserves_result_order() -> None:
    results = (
        RankedChunk(SourceChunk("b.py", 1, 1, "B = 1"), 2.0, 1),
        RankedChunk(SourceChunk("a.py", 4, 4, "A = 1"), 1.0, 2),
    )

    _, user_prompt = build_answer_prompt("question", results)

    assert user_prompt.index("[E1] b.py:1-1") < user_prompt.index("[E2] a.py:4-4")
