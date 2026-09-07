# 当前能力边界（2026-09-07，2.0.3 开发分支）

截至 2026-09-07，本项目可由代码、离线测试和最小真实 Provider smoke 证明；GitHub 最新正式 Release 为 v2.0.3：

- Auto Answer：本地仓库扫描、问题路由、BM25/语义候选、证据引用映射。
- Unified Chat：同一 Session 内的多轮上下文、`general_chat`/`scope_redirect`/`clarify`/解释/变更任务路由、SSE 阶段事件和调试 reasoning 展示。
- 业务范围引导：业务外闲聊会自然承接并引导回代码任务；该路径不调用模型、不访问仓库，但保留 Session 审计记录。
- Unified Chat 回答契约：`assistant_message` 是唯一自然语言正文；`result` 只保存路线、引用、模型、Router、缓存、usage 和 Change 详情。
- 普通聊天边界：已绑定 Session 后不重新扫描仓库、不调用 Query Router/Embedding/Rerank/代码检索和修改工具；文本 Gateway 请求不发送 JSON response format。
- Usage Dashboard：调用次数、输入/输出、cache read/write、命中率、成本、模型和降级信息。
- Tool Loop：8 个只读/提案工具通过 MCP stdio 执行；写工具不向模型暴露。
- Change API：preview → approval → isolated workspace → fixed validation → result/rollback；本地开发模式可显式自动确认并跳过 Docker 校验，但结果会标记为未执行，生产环境强制关闭。
- MCP 路径兼容：工具仍拒绝越界、敏感文件和版本控制目录；对当前仓库根目录内的绝对路径或重复仓库前缀做安全归一化，减少真实模型探索时的路径误判。
- 变更状态在本地状态目录持久化；进程重启时 RUNNING 会转为 UNKNOWN，禁止自动重放。
- 结果可通过 `GET /api/v2/change/{run_id}/patches/{patch_id}` 查询；并发重复 apply 在单进程内只执行一次。

以下不属于当前已证明能力：真实模型质量基准、生产级高可用、多写者并发、真实 Kubernetes rollout/undo、完整 SWE-bench resolve，以及所有 Provider 都经过同一 Gateway。

2.0.3 发布工作树的验证证据：后端 Ruff 通过，完整 pytest 为 `644 passed, 46 skipped`；前端 lint、4 个测试文件共 10 个测试和 production build 均通过。已完成一次真实 DeepSeek `general_chat` FastAPI smoke，以及一次真实 DeepSeek `change` smoke：后者在开发模式下返回 `COMPLETED`/`change_result`、自动审批、明确标记 `validation_skipped=true`，且源 fixture 未改变；本轮还对 `scope_redirect` 做了当前 API smoke，确认 `model_called=false`。真实调用只证明当前链路可运行，不是模型质量基准或大规模并发结论。
