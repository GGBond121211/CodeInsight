# CodeInsight v2.0.3

发布日期：2026-09-07

这是正式 Release，不是 pre-release。发布基线为 GitHub 上的 v2.0.2。

## 本版本内容

- 完善统一 Conversation 的普通聊天、澄清、代码理解和代码修改路由。
- 增加 `scope_redirect` 业务范围引导：对篮球、天气等无关话题自然承接并引导回代码任务，不调用模型、不访问仓库工具。
- 保留同一 Session 的多轮上下文、代码目标和修改闭环；范围引导不会创建或替换代码目标。
- 增加开发调试策略：本地调试可自动确认修改预览、跳过不可用的固定校验，但结果会明确标记校验未执行。
- 保持统一单回答契约，前端只展示一份 `assistant_message`；模型、路由、缓存、usage、reasoning 和范围引导状态进入运行详情。
- 同步更新后端、前端、Compose、Kubernetes 和部署脚本的版本标记为 2.0.3。

## 发布前验证

- 后端 Ruff：通过。
- 后端 pytest：644 passed, 46 skipped。
- 前端：4 个测试文件、10 个测试通过；lint 和 production build 通过。
- 当前 API scope redirect smoke：返回 `COMPLETED`、`route=scope_redirect`、`model_called=false`。
- 真实 DeepSeek general chat 与 development change smoke 已在本开发周期完成；真实调用只证明链路可运行，不是模型质量基准或大规模并发结论。

## 边界

本版本不宣称真实大并发、完整 SWE-bench resolve、生产级 Kubernetes rollout/undo、多写者并发或多区域高可用。真实 Provider 的原始响应、密钥和模型隐藏推理不写入仓库。
