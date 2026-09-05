# Step 8 最小发布实验预算记录

日期：2026-09-05
授权口径：用户明确允许预计总消耗低于 10,000,000 Token 的实验直接执行；预计达到 9,000,000 Token 前停止并重新确认。
配置：`experiments/configs/step8_minimum_release.yaml`

## 本轮允许执行的真实调用

| 实验 | 案例数 | 每案上限/假设 | 预计 Token | 状态 |
| --- | ---: | --- | ---: | --- |
| 阿里云 `qwen3.7-text-rerank` pilot | 30 | 最多 100 个候选，按实际 usage 记录 | 5,000,000 | 已授权，执行前继续记录实际 usage |
| Tool Loop/主链路 smoke | 5~10 | 单 Trial，失败不自动扩大样本 | 250,000 | 已授权，按阶段执行 |
| Retry/fallback 预留 | - | 仅真实请求产生时计入 | 1,000,000 | 预算保留，不主动消耗 |
| Embedding 四模型 A/B | - | 不在本轮执行 | 0 | 延期到 2.1 |
| Chat Provider 完整 A/B | - | 不在本轮执行 | 0 | 延期到 2.1 |

## 预算判断

- 当前保守估计合计：`6,250,000 Token`。
- 硬停止线：累计实际 `usage.total_tokens >= 9,000,000`，立即停止后续真实调用。
- 用户上限：`10,000,000 Token`；本轮不接近该上限运行。
- Fake Provider、确定性检索、契约测试、Gateway 故障注入和并发测试不消耗上游模型 Token。

## 记录要求

每一次真实请求必须记录 request ID、model、case ID、候选数量、输入/输出/总 Token、耗时和错误类别；不得保存 API Key、Authorization 或原始敏感响应。

本预算只授权小规模 baseline pilot，不授权 Master-200 全量真实重跑、四模型 Embedding A/B、完整 Provider A/B、300/450 真实并发或 ×10 规模实验。
