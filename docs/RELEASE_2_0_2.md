# CodeInsight v2.0.2

发布日期：2026-09-07

这是正式 Release，不是 pre-release。发布基线为 GitHub 上的 v2.0.1，内容汇总了本地尚未推送的项目更新。

## 本版本内容

- 接入统一 Conversation API，支持同一 Session 的多轮上下文。
- 增加解释、代码变更和继续处理的任务路由。
- 增加 SSE 阶段事件、运行状态、当前模型/降级模型和调试 reasoning 展示。
- 增加 Gateway 用量汇总、调用明细、缓存读写、命中率、成本和延迟观测。
- 完善 Change API 的 Sandbox preflight、隔离工作区、校验、指纹保护、有限续作和回滚闭环。
- 扩充后端回归、前端契约测试和 CI 的前端 lint/test/build 检查。
- 更新 Docker Compose、Kubernetes manifest、部署脚本和应用版本标记为 2.0.2。

## 发布前验证

- 后端 Ruff：通过。
- 后端 pytest：617 passed, 46 skipped。
- 前端：4 个测试文件、8 个测试通过；lint 和 production build 通过。
- Docker Compose config：通过。
- Docker 发布候选镜像构建：通过。
- 最小真实 Provider smoke：通过；未进行大规模调用或质量基准测试。

## 边界

本版本不宣称真实大并发、完整 SWE-bench resolve、生产级 Kubernetes rollout/undo 或多区域高可用。真实 Provider 的原始响应、密钥和模型隐藏推理不写入仓库。
