# CodeInsight 2.0.1

`2.0.1` 是正式 Release，不是 pre-release。它在 2.0 的 Gateway、Tool Loop、Session/Memory 和本地部署基础上，补齐一次可验证的用量、缓存和长上下文切片。

## 本次交付

- Provider usage 同时识别 DeepSeek 原生的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，并兼容 OpenAI-compatible 的 `prompt_tokens_details.cached_tokens`。
- Gateway 为完整请求和稳定前缀生成不可逆 SHA-256 指纹。语义缓存键纳入 model、prompt version、消息、tools、response format 和输出预留额度；含 workspace state 的请求继续禁止语义缓存。
- Gateway 提供 `/v1/usage/summary` 与 `/v1/usage/calls`。返回模型、路由、Token、缓存读/未命中、成本、延迟、错误和指纹，不返回原始 prompt、工具参数、模型正文或隐藏推理。
- Tool Loop 将 `tool_call_requested`、`tool_call_validated`、`tool_dispatched`、`tool_result_committed` 和 `tool_aborted` 写入同一个 Run 事件流；只读工具可并行执行，但结果按模型提交顺序重新归并；取消或未知工具会得到明确的失败结果。
- Session 自动保留最近 24 轮；超出后把旧轮次写成不调用模型的恢复线索摘要，并保留全局轮次边界。模型调用前，tool results、session summary、repository map 等低优先级分区支持头尾确定性压缩。
- 后端、前端、镜像和 Kubernetes 本地部署契约统一到 `2.0.1` / `2.0.1-local`。

## 明确边界

本版本提供监控数据接口和低基数 Prometheus 指标，尚未把图表型 Dashboard UI 作为本次发布的验收项。真实 Provider 的质量、成本与缓存收益仍需脱敏的线上样本或获批实验测量；本地 Fake Provider smoke 不能替代该结论。CD 会构建并推送版本镜像到 GHCR，但不会在没有集群和环境审批时自动部署 Kubernetes。

## 合并前验证

```text
uv run --project backend --locked ruff check backend/src backend/tests
uv run --project backend --locked python -m pytest --basetemp=work/pytest-2-0-1-full -q backend/tests
npm.cmd ci
npm.cmd run lint
npm.cmd test -- --run
npm.cmd run build
docker compose config
docker build --file Dockerfile --tag codeinsight-api:2.0.1-local .
kubectl kustomize k8s/base
```
