import runpy
from pathlib import Path

from codeinsight.domain.answer import AnswerCitation, RepositoryAnswer

RUNNER_PATH = Path(__file__).with_name("run_answer_eval.py")


def test_answer_evaluation_payload_shape(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    case = {
        "id": "case-1",
        "input": {"question": "Where is run?"},
        "expected": {
            "outcome": "answered",
            "evidence": [{"path": "src/app.py", "start_line": 1, "end_line": 2}],
        },
        "required_terms": ["run"],
    }
    result = RepositoryAnswer(
        outcome="answered",
        answer="run is defined in app.py.",
        citations=(AnswerCitation("E1", "src/app.py", 1, 2),),
        retrieval_mode="bm25",
        model="fake-model",
        prompt_version="code-answer-v1",
        input_tokens=10,
        output_tokens=4,
    )
    build_payload = runpy.run_path(str(RUNNER_PATH))["build_evaluation_payload"]

    payload = build_payload(
        cases=[case],
        results={"case-1": result},
        errors={},
        elapsed_milliseconds={"case-1": 12},
        fixture_root=tmp_path,
        model="fake-model",
        base_url_host="api.example.test",
        prompt_version="code-answer-v1",
    )

    assert set(payload) == {
        "prompt_version",
        "retrieval_mode",
        "model",
        "base_url_host",
        "case_count",
        "metrics",
        "usage",
        "cases",
    }
    assert payload["metrics"]["outcome_accuracy"] == 1.0
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 4}
    assert payload["cases"][0]["failure_types"] == []
    assert payload["cases"][0]["citations"][0]["relative_path"] == "src/app.py"
