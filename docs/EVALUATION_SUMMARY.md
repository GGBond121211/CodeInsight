# 评测摘要

截至 2026-09-07，以下证据可以公开说明：

| 范围 | 证据 | 结论 |
| --- | --- | --- |
| 离线工程回归 | 发布装配树运行 Ruff；后端 `617 passed, 46 skipped` | 代码与关键失败路径可回归 |
| 前端契约 | lint、4 个 Vitest 测试文件共 8 个测试、production build | 本地演示页面可构建 |
| 真实 Provider smoke | 最小 Auto Answer 与统一 Chat 单轮调用成功 | 证明配置和请求链可用，不代表模型质量基准 |
| Step 8 pilot | 30 条小样本真实 pilot，约 1,052,664 tokens | 证明评测链路可运行，不代表全量质量 |
| L2 dev localization | 20 条 frozen dev，BM25 + fixed-80/0 | 只代表定位基线，不代表 resolve rate |
| Step 9 | Compose、Fake Gateway、Worker 和 Kustomize render | 本地部署契约，不代表 Kubernetes 生产部署 |

历史 Master-200 指标仍保留在私有评测文档和 README 的历史章节中。它们来自旧流水线，不能解释当前 qwen Rerank 接入后的收益；本轮没有重复运行 Master-200、holdout、完整 SWE-bench 或大规模真实并发。

当前最小可复现命令：

```powershell
cd backend
uv run python -m pytest --basetemp=../work/pytest-current
uv run ruff check src tests
cd ..\frontend
npm.cmd run lint
npm.cmd test -- --run
npm.cmd run build
```

发布装配树使用项目锁文件完成 `uv` 环境准备；CI 仍会在 Ubuntu 上执行独立的 `uv sync --locked`、Ruff 和全量 pytest。
