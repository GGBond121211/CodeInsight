# 当前能力边界（2026-09-07，2.0.3 开发分支）

截至 2026-09-07，本项目可由代码、离线测试和最小真实 Provider smoke 证明；GitHub 最新正式 Release 仍为 v2.0.2，以下新增内容属于未发布的 2.0.3 开发分支：

- Auto Answer：本地仓库扫描、问题路由、BM25/语义候选、证据引用映射。
- Unified Chat：同一 Session 内的多轮上下文、`general_chat`/`clarify`/解释/变更任务路由、SSE 阶段事件和调试 reasoning 展示。
- Unified Chat 回答契约：`assistant_message` 是唯一自然语言正文；`result` 只保存路线、引用、模型、Router、缓存、usage 和 Change 详情。
- 普通聊天边界：已绑定 Session 后不重新扫描仓库、不调用 Query Router/Embedding/Rerank/代码检索和修改工具；文本 Gateway 请求不发送 JSON response format。
- Usage Dashboard：调用次数、输入/输出、cache read/write、命中率、成本、模型和降级信息。
- Tool Loop：8 个只读/提案工具通过 MCP stdio 执行；写工具不向模型暴露。
- Change API：preview → approval → isolated workspace → fixed validation → result/rollback。
- 变更状态在本地状态目录持久化；进程重启时 RUNNING 会转为 UNKNOWN，禁止自动重放。
- 结果可通过 `GET /api/v2/change/{run_id}/patches/{patch_id}` 查询；并发重复 apply 在单进程内只执行一次。

以下不属于当前已证明能力：真实模型质量基准、生产级高可用、多写者并发、真实 Kubernetes rollout/undo、完整 SWE-bench resolve，以及所有 Provider 都经过同一 Gateway。

本轮 2.0.3 开发工作树的验证证据：后端 Ruff 通过，完整 pytest 为 `630 passed, 46 skipped`；前端 lint、4 个测试文件共 9 个测试和 production build 均通过。另完成一次真实 DeepSeek `general_chat` FastAPI smoke：返回 `COMPLETED`、任务类型为 `general_chat`，实际模型为 `deepseek-v4-flash`，返回非空自然语言，事件包含 `model_called`/`debug_reasoning` 且不包含 `retrieval_started`，`result` 不含第二份 `answer` 字段。这不是模型质量基准或大规模并发结论。
