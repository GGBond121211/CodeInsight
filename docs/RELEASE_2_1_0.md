# CodeInsight 2.1.0

2.1.0 大版本更新。相对 2.0.3 共 45 个提交、247 个文件变更（+31865 / −6143）。

## 主要变化

- **检索**：Dense/Sparse + Qdrant 取代 BM25/lexical 运行时检索；索引发布支持 staging 与回滚。
- **代码理解**：只读 Tool Loop、Evidence Ledger、证据评估与 Repair Loop；explain 请求收口到新路径。
- **异步 Agent Run**：Agent Run Worker 与队列、跨进程执行、产出事实层与读取投影、事件 SSE 重放。
- **模型与上下文**：Gateway 缓存与降级、上下文预算、长上下文生命周期策略。
- **前端**：工作台界面重做，覆盖等待隔离校验、结果不明、需要人工确认等新状态。

## 验证

- 后端 `tests/unit`：720 passed / 1 skipped；`ruff` 通过。
- 前端：`tsc -b`、`eslint`、`vitest` 4 files / 12 tests 通过。
- 跨进程链路有真实取证：API 与 Worker 分属两个进程 + 真 MySQL + 真 Redis + 真模型跑通一轮，
  换一个新进程 HTTP 读回结果一致。

## 已知边界

- 参数状态为 `implemented_unoptimized`，尚未做 A/B 对照，答案质量无结论。
- 多进程容量实验只到 2–4 进程，长跑稳定性未测。
- 详见 `docs/RELEASE_2_1_0_Q1_Q5.md`、`docs/RELEASE_2_1_0_Q8_Q9.md`、`docs/CAPABILITIES.md`。
