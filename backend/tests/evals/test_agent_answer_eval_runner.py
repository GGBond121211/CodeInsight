import runpy
from pathlib import Path

from codeinsight.domain.agent import AgentEvent, AgentRepositoryAnswer
from codeinsight.domain.answer import AnswerCitation, RepositoryAnswer

RUNNER_PATH = Path(__file__).with_name("run_agent_answer_eval.py")


def test_agent_payload_records_metrics_events_usage_and_revisions(tmp_path: Path) -> None:
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
    answer = RepositoryAnswer(
        "answered",
        "run is defined here.",
        (AnswerCitation("E1", "src/app.py", 1, 2),),
        "hybrid",
        "fake-model",
        "citation-agent-v3",
        30,
        8,
    )
    agent_result = AgentRepositoryAnswer(
        answer,
        1,
        (AgentEvent(1, "retrieve", "Retrieved evidence."),),
        30,
        8,
    )
    build_payload = runpy.run_path(str(RUNNER_PATH))["build_agent_evaluation_payload"]

    payload = build_payload(
        cases=[case],
        results={"case-1": agent_result},
        errors={},
        elapsed_milliseconds={"case-1": 20},
        fixture_root=tmp_path,
        model="fake-model",
        base_url_host="api.example.test",
    )

    assert payload["metrics"]["citation_precision"] == 1.0
    assert payload["usage"] == {"input_tokens": 30, "output_tokens": 8}
    assert payload["total_revisions"] == 1
    assert payload["cases"][0]["events"][0]["step"] == "retrieve"
