"""Q-012 A2：用真实模型补一批 explain 样本（小批量，不跑全量）。

只回答一个问题：一次 explain 到底要几次工具调用、花多少 token。
样本很少（默认 4 题），因此结论只用于判断「值不值得做快路径」，不构成质量结论。

用法：
    backend/.venv/Scripts/python.exe experiments/q12_explain_probe.py --questions 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from codeinsight.application.code_understanding_route import (  # noqa: E402
    run_code_understanding_answer,
)
from codeinsight.infrastructure.model_gateway import GatewayChatModel  # noqa: E402

DEFAULT_QUESTIONS: tuple[str, ...] = (
    "send 方法在哪里定义？",
    "Client 类定义在哪个文件？",
    "这个仓库的顶层模块有哪些？",
    "哪里实现了重定向跟随？",
    "read_timeout 配置从哪里读取？",
    "哪些模块引用了 Request 类？",
)


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Q-012 explain 真实取样")
    parser.add_argument("--repository", type=Path, default=ROOT / "work" / "benchmarks" / "httpx")
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "q12_explain_probe.json")
    args = parser.parse_args()

    _load_env()
    if not args.repository.is_dir():
        print(f"仓库不存在：{args.repository}")
        return 2
    model = GatewayChatModel.from_environment()
    questions = DEFAULT_QUESTIONS[: max(1, args.questions)]
    records: list[dict[str, object]] = []
    total_tokens = 0
    for index, question in enumerate(questions, start=1):
        started = time.monotonic()
        result = run_code_understanding_answer(
            args.repository,
            question,
            model=model,
            run_id=f"q12-probe-{index}",
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        tools = [str(event.payload.get("tool_name", "")) for event in result.events
                 if event.event_type == "tool_result_committed"]
        record = {
            "question": question,
            "status": result.status,
            "outcome": result.outcome,
            "tool_calls": result.tool_calls,
            "first_tool": tools[0] if tools else "",
            "tools": tools,
            "steps": result.steps,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "token_budget": result.token_budget,
            "budget_exhausted": result.budget_exhausted,
            "elapsed_ms": elapsed_ms,
            "structured_rows": len(result.structured_evidence),
        }
        records.append(record)
        total_tokens += result.input_tokens + result.output_tokens
        print(json.dumps(record, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": getattr(model, "model", ""),
                "provider_host": os.environ.get("CODEINSIGHT_BASE_URL", ""),
                "repository": str(args.repository),
                "records": records,
                "total_tokens": total_tokens,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"总 token：{total_tokens}；已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
