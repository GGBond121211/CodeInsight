"""Q-012 C2.3 / C2.4 / C2.5：快路径与 Tool Loop 的同数据对照（真实模型）。

两条路线跑同一批题面、同一个模型、同一份仓库，只换取证方式：

  arm=fast  按命中原型的固定配方取证，只让模型写一次措辞（问题原型命中才跑）；
  arm=loop  现有开放只读 Tool Loop（对照组，生产路径）。

为什么是「同数据对照」而不是分开跑两份报告：快路径省的 token 只有在答对的前提下
才算收益。分开跑就容易只报省下的钱、不报答错的题。

口径（写进 docs/Q12_C2_EVALUATION.md，不在这里偷偷改）：
  - 命中判据是 expected_paths 的子串匹配，来自 rg 在同一个 checkout 上查出的位置；
  - 快路径没命中原型、或配方取证不足而回退，都记成 not_hit / fallback，不算答对；
  - 整个对照共用一条 MCP 会话，开跑前先预热一次索引；Evidence Ledger 与 Tool Loop
    状态仍然每轮新建，预热耗时单独登记、不计入任何一臂。

用法：
    backend/.venv/Scripts/python.exe experiments/q12_fast_path_ab.py --trials 3
    加 --cases 2 --trials 1 可以先跑小额冒烟。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from codeinsight.agent.code_understanding_tool_loop import (  # noqa: E402
    CodeUnderstandingToolLoop,
)
from codeinsight.agent.tool_loop import ToolCall  # noqa: E402
from codeinsight.application.archetype_fast_path import (  # noqa: E402
    FAST_PATH_ENV,
    run_archetype_fast_path,
)
from codeinsight.application.archetype_router import ArchetypeRouter  # noqa: E402
from codeinsight.infrastructure.mcp_client import StdioMCPClient  # noqa: E402

CASES = ROOT / "experiments" / "configs" / "q12_explain_cases.yaml"
EVIDENCE_TOOLS = frozenset({"search_repository", "get_evidence_context", "read_file"})
MCP_TIMEOUT_SECONDS = 300.0
_CLIENT: StdioMCPClient | None = None
_RESTARTS = 0


def _shared_client(repository: Path) -> StdioMCPClient:
    """整个对照共用一条 MCP 会话；冷启动的整仓库索引只建一次。"""

    global _CLIENT
    if _CLIENT is None:
        _CLIENT = StdioMCPClient(str(repository), timeout_seconds=MCP_TIMEOUT_SECONDS)
        _CLIENT.start()
    return _CLIENT


def _healthy_client(repository: Path) -> StdioMCPClient:
    """跑每一轮之前确认会话还活着；死了就重开一条并计数。

    为什么要有这一层：MCP 子进程一旦退出，后面每一轮都会在 0 毫秒内 FAILED，
    整场对照就变成噪音。重开的代价是那一轮之后要重建一次索引（约两分钟），
    所以重启次数单独登记，不藏进任何一臂的耗时里。
    """

    global _CLIENT, _RESTARTS
    client = _shared_client(repository)
    try:
        client.list_tools()
        return client
    except Exception:  # noqa: BLE001 - 会话不可用只意味着「重开一条」
        _RESTARTS += 1
        print(f"MCP 会话不可用，第 {_RESTARTS} 次重开（下一轮要重新建索引）")
        client.close()
        _CLIENT = None
        return _shared_client(repository)


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


def _load_cases(limit: int) -> tuple[Path, list[dict[str, object]]]:
    document = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    repository = ROOT / str(document["repository"])
    cases = [dict(item) for item in document["cases"]]
    return repository, cases[:limit] if limit > 0 else cases


def _cited_paths(result) -> list[str]:
    return [str(citation.relative_path) for citation in result.citations]


def _path_metrics(result, expected: list[str]) -> tuple[bool, float]:
    cited = _cited_paths(result)
    matched = [pattern for pattern in expected if any(pattern in path for path in cited)]
    return bool(matched), len(matched) / len(expected) if expected else 0.0


def _run_fast(*, case, repository: Path, router, model) -> dict[str, object]:
    question = str(case["question"])
    started = time.monotonic()
    match = router.route(question)
    if match is None:
        return {
            "arm": "fast",
            "question": question,
            "case_id": case["id"],
            "route": "not_hit",
            "fast_path": False,
            "status": None,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls": 0,
            "tools": [],
            "first_tool": "",
            "citations": [],
        }
    # 回退原因要留在记录里：只说「回退」看不出是没命中、工具报错还是证据不足。
    report: list[dict[str, str]] = []
    result = run_archetype_fast_path(
        archetype=match.archetype,
        question=question,
        mcp_client=_healthy_client(repository),
        model=model,
        run_id=f"q12-ab-fast-{case['id']}",
        emit=lambda event_type, payload: report.append(dict(payload)),
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if result is None:
        return {
            "arm": "fast",
            "question": question,
            "case_id": case["id"],
            "route": "fallback",
            "reasons": report,
            "archetype": match.archetype.name,
            "archetype_score": round(match.score, 4),
            "fast_path": False,
            "status": None,
            "elapsed_ms": elapsed_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls": len(match.archetype.recipe),
            "tools": [step.tool for step in match.archetype.recipe],
            "first_tool": match.archetype.recipe[0].tool,
            "citations": [],
        }
    matched, recall = _path_metrics(result, list(case["expected_paths"]))
    return {
        "arm": "fast",
        "question": question,
        "case_id": case["id"],
        "route": "hit",
        "reasons": report,
        "archetype": match.archetype.name,
        "archetype_score": round(match.score, 4),
        "fast_path": True,
        "status": result.status,
        "outcome": result.outcome,
        "answer": result.answer,
        "citation_hit": matched,
        "citation_recall": recall,
        "elapsed_ms": elapsed_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "tool_calls": result.tool_calls,
        "tools": [step.tool for step in match.archetype.recipe],
        "first_tool": match.archetype.recipe[0].tool,
        "citations": _cited_paths(result),
    }


def _run_loop(*, case, repository: Path, model) -> dict[str, object]:
    question = str(case["question"])
    started = time.monotonic()
    loop = CodeUnderstandingToolLoop(
        model,
        _healthy_client(repository),
        run_id=f"q12-ab-loop-{case['id']}",
        repo_root=str(repository),
    )
    result = loop.run(question)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    tools = [
        str(event.payload.get("tool_name", ""))
        for event in result.events
        if event.event_type == "tool_result_committed"
    ]
    matched, recall = _path_metrics(result, list(case["expected_paths"]))
    return {
        "arm": "loop",
        "question": question,
        "case_id": case["id"],
        "route": "loop",
        "fast_path": False,
        "status": result.status,
        "outcome": result.outcome,
        "answer": result.answer,
        "citation_hit": matched,
        "citation_recall": recall,
        "elapsed_ms": elapsed_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "tool_calls": result.tool_calls,
        "tools": tools,
        "first_tool": tools[0] if tools else "",
        "budget_exhausted": result.budget_exhausted,
        "termination_reason": result.termination_reason,
        "citations": _cited_paths(result),
    }


def _guarded(run, *, case, **kwargs) -> dict[str, object]:
    """单轮异常不许带走整场对照：记成 error 行，继续跑后面的轮次。"""

    try:
        return run(case=case, **kwargs)
    except Exception as error:  # noqa: BLE001 - 已经跑完的轮次要保住
        return {
            "arm": "fast" if run is _run_fast else "loop",
            "question": str(case["question"]),
            "case_id": case["id"],
            "route": "error",
            "status": None,
            "elapsed_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls": 0,
            "tools": [],
            "first_tool": "",
            "citations": [],
            "error": f"{type(error).__name__}: {error}"[:200],
        }


def _consecutive_failures(rows: list[dict[str, object]]) -> int:
    """末尾连续多少轮属于「根本没跑起来」：会话或 Provider 级故障，不是模型行为。"""

    count = 0
    for row in reversed(rows):
        failed = row.get("route") == "error" or (
            row.get("status") == "FAILED" and int(row.get("tool_calls") or 0) == 0
        )
        if not failed:
            break
        count += 1
    return count


def _tokens(row: dict[str, object]) -> int:
    """一轮的 input+output；缺字段按 0 算（没跑模型的那几种终止也走这里）。"""

    return int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)


def _describe(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"runs": 0}
    tokens = [_tokens(row) for row in rows]
    elapsed = [int(row["elapsed_ms"]) for row in rows]
    answered = [row for row in rows if row.get("status") == "ANSWERED"]
    hits = [row for row in rows if row.get("citation_hit")]
    recall = [float(row.get("citation_recall") or 0.0) for row in rows]
    first_tools = [str(row.get("first_tool") or "") for row in rows]
    return {
        "runs": len(rows),
        "answered": len(answered),
        "answered_rate": len(answered) / len(rows),
        "citation_hit": len(hits),
        "citation_hit_rate": len(hits) / len(rows),
        "citation_recall_mean": mean(recall),
        "tokens_total": sum(tokens),
        "tokens_mean": mean(tokens),
        "tokens_median": median(tokens),
        "elapsed_mean_ms": mean(elapsed),
        "tool_calls_mean": mean([int(row.get("tool_calls") or 0) for row in rows]),
        "first_tool_evidence_rate": (
            sum(1 for name in first_tools if name in EVIDENCE_TOOLS) / len(first_tools)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Q-012 C2.4 快路径 vs Tool Loop")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cases", type=int, default=0, help="0 表示全部题面")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "q12_fast_path_ab.json")
    args = parser.parse_args()

    _load_env()
    # arm=loop 必须走开放 Tool Loop：这里显式关掉快路径开关，不依赖 shell 环境。
    os.environ[FAST_PATH_ENV] = ""

    from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
    from codeinsight.infrastructure.model_gateway import GatewayChatModel

    model = GatewayChatModel.from_environment()
    router = ArchetypeRouter(embed=OpenAIEmbeddingModel.from_environment().embed)
    repository, cases = _load_cases(args.cases)
    if not repository.is_dir():
        print(f"仓库不存在：{repository}")
        return 2

    print(f"模型 {getattr(model, 'model', '')}｜题面 {len(cases)} 道｜Trial {args.trials}")
    warmup_started = time.monotonic()
    warmup = _shared_client(repository).call_tool(
        ToolCall("warmup", "search_repository", {"question": str(cases[0]["question"])})
    )
    warmup_ms = int((time.monotonic() - warmup_started) * 1000)
    print(f"索引预热 ok={warmup.ok} err={warmup.error_code} {warmup_ms}ms")
    rows: list[dict[str, object]] = []
    aborted = False
    for trial in range(1, args.trials + 1):
        for case in cases:
            # 两条路线都在同一 process 里顺序跑：先快路径后对照，方便中断时看进度。
            for record in (
                _guarded(
                    _run_fast,
                    case=case,
                    repository=repository,
                    router=router,
                    model=model,
                ),
                _guarded(
                    _run_loop, case=case, repository=repository, model=model
                ),
            ):
                record["trial"] = trial
                rows.append(record)
                print(
                    f"T{trial} {record['case_id']:22s} {record['arm']:4s} "
                    f"{str(record.get('route')):8s} status={record.get('status')} "
                    f"cites={len(record.get('citations') or [])} "
                    f"hit={record.get('citation_hit')} "
                    f"tok={_tokens(record)} "
                    f"{record['elapsed_ms']}ms"
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if _consecutive_failures(rows) >= 3:
                    # 3 轮连失败基本不是模型行为，是环境故障：再跑下去只是烧钱。
                    print("连续 3 轮未跑起来，判定环境故障，停止本次对照（已跑的轮次保留）")
                    aborted = True
                    break
            if aborted:
                break
        if aborted:
            break

    fast_rows = [row for row in rows if row["arm"] == "fast"]
    loop_rows = [row for row in rows if row["arm"] == "loop"]
    fast_hits = [row for row in fast_rows if row["route"] == "hit"]
    summary = {
        "version": "q012-fast-path-ab-v1",
        "trials": args.trials,
        "cases": len(cases),
        "model": getattr(model, "model", ""),
        "shared_mcp_session": True,
        "warmup_ms": warmup_ms,
        "router_threshold": router.config.threshold,
        "router_margin": router.config.margin,
        "fast_all": _describe(fast_rows),
        "fast_hit_only": _describe(fast_hits),
        "loop": _describe(loop_rows),
        "fast_routes": {
            "hit": len(fast_hits),
            "error": len([row for row in rows if row["route"] == "error"]),
            "not_hit": len([row for row in fast_rows if row["route"] == "not_hit"]),
            "fallback": len([row for row in fast_rows if row["route"] == "fallback"]),
        },
        "mcp_restarts": _RESTARTS,
        "aborted_after_failures": aborted,
    }
    if fast_hits and loop_rows:
        fast_tokens = mean(
            [int(row["input_tokens"]) + int(row["output_tokens"]) for row in fast_hits]
        )
        loop_tokens = mean(
            [int(row["input_tokens"]) + int(row["output_tokens"]) for row in loop_rows]
        )
        summary["token_ratio_fast_over_loop"] = fast_tokens / loop_tokens if loop_tokens else None
    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已写入 {args.output}")
    if _CLIENT is not None:
        _CLIENT.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
