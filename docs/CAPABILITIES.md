# 当前能力边界（2026-09-07 快照：2.0.3；2.1.0 开发分支见文末）

截至 2026-09-07，本项目可由代码、离线测试和最小真实 Provider smoke 证明；GitHub 最新正式 Release 为 v2.0.3：

- Auto Answer：本地仓库扫描、问题路由、Provider Dense/Sparse 候选、证据引用映射。
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

## 2.1.0 开发分支（2026-09-11，未发布）

`codex/2.1.0-development` 分支在 2.0.3 之上实现了异步 Agent Run。以下是当前能由代码、
契约测试和真实中间件实验证明的部分：

- 统一 Chat 的 Agent Run 由后台 Worker 执行：`POST /api/v2/chat/turns` 先把这一轮写成
  可查询的 Run 事实，再投递任务，接口返回 202；`AgentRunWorker` 按租约领取，重建上下文
  并跑完整 Model → Tool → Model 循环。
- ValidationWorker 独立执行固定 Sandbox 检查：apply 之后不再同步跑沙箱，而是登记一条校验
  任务（幂等键 `run_id:patch_id:profile`），由校验 Worker 领取，结论再回投给 Agent Run。
- SSE 从持久事件流实时/断线续传：先回放已发生的 sequence 再订阅增量，重复事件按 sequence
  去重；终态与 `UNKNOWN` / `MANUAL_REQUIRED` 都会停流。
- 状态与队列事实可恢复：Run/Task/Event 落到 `agent_runs` 等事实表（配了 MySQL 时），
  进程重启后仍可回放；恢复不了的状态显式标记，不自动重放有副作用的操作。
- 队列背压有明确错误类别：`CODEINSIGHT_AGENT_RUN_QUEUE_LIMIT` 到上限拒绝并留
  `QUEUE_FULL` 事实；broker 不可达时只读任务 `FAILED/DISPATCH_FAILED`、续跑
  `MANUAL_REQUIRED/DISPATCH_UNKNOWN`，两者都不会停在 `QUEUED`。

仍然不属于已证明能力：3 个以上 Worker 进程的规模与长跑稳定性、Redis 抖动（断开再恢复）、
broker 自动重投（多进程实验里的接管靠显式重投，未 ack 的消息要等 Celery 的 visibility
timeout）、队列等待超时（本版本未实现）、真实模型在高并发下的吞吐。跨进程执行与读取、
两个真实 Worker 进程的租约竞争与逐进程资源归因、真实杀进程后的接管都有证据，见
`docs/AGENT_RUN_WORKER_EVALUATION.md`；参数状态停在 `implemented_unoptimized`。
