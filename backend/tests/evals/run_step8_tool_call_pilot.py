"""Run five bounded native Tool Calling probes through the Model Gateway."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from codeinsight.infrastructure.model_gateway import GatewayChatModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "STEP8-MINIMUM-RELEASE"
TOOL_SCHEMA = (
    {
        "type": "function",
        "function": {
            "name": "search_repository",
            "description": "Search a code repository and return evidence references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["question", "limit"],
                "additionalProperties": False,
            },
        },
    },
)
PROMPTS = (
    "Find the implementation path for the checkout request and return evidence.",
    "Locate where the payment provider is validated and return evidence.",
    "Trace the response currency field to its source and return evidence.",
    "Find the inventory reservation boundary and return evidence.",
    "Find the function that handles a missing payment provider and return evidence.",
)


def run() -> dict[str, object]:
    model = GatewayChatModel.from_environment()
    rows: list[dict[str, object]] = []
    for index, prompt in enumerate(PROMPTS, start=1):
        try:
            response = model.complete_with_tools(
                (
                    {
                        "role": "system",
                        "content": "Use the provided tool when repository evidence is needed.",
                    },
                    {"role": "user", "content": prompt},
                ),
                TOOL_SCHEMA,
            )
            rows.append(
                {
                    "probe": index,
                    "status": "ok",
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "tool_call_count": len(response.tool_calls),
                    "tool_names": [call.name for call in response.tool_calls],
                }
            )
        except Exception as error:  # noqa: BLE001 - record probe failure without secret text
            rows.append(
                {
                    "probe": index,
                    "status": "error",
                    "error_type": type(error).__name__,
                }
            )
    output = {
        "schema_version": 1,
        "experiment_id": "STEP8-TOOL-CALL-PILOT",
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "probe_count": len(rows),
        "rows": rows,
        "interpretation": [
            "这是原生 tool_calls 能力探针，不是完整 Tool Loop 成功率或 Patch 质量实验。",
            "探针只记录工具调用元数据，不执行模型返回的工具。",
        ],
    }
    run_dir = sorted(OUTPUT_ROOT.glob("*"))[-1]
    path = run_dir / "tool_call_pilot.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))
    return output


if __name__ == "__main__":
    run()
