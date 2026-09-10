# CodeInsight 2.1.0 Q1–Q5 升级切片

本文件记录 `codex/2.1.0-development` 工作树中 Q1–Q5 的实现边界。它是本地升级
切片的事实入口，不把契约测试、进程内 Qdrant 或未配置的外部后端写成生产质量证据。

## 已实现的运行时变化

| 计划 | 本次实现 | 当前证据边界 |
| --- | --- | --- |
| Q1 Provider Dense/Sparse 检索 | EmbeddingBatch 可表达 Provider Dense + Sparse；Sparse 坐标/权重、批次顺序和响应格式受校验；Qdrant 使用 `dense`/`sparse` Named Vectors；正常 `hybrid` 路径用两路召回、RRF 和 qwen Rerank；没有 Sparse 时受控失败 | Fake Provider 契约和本地 Qdrant 契约已测；目标 Provider 是否真的返回 Sparse 仍需真实最小请求确认，不能由模型名推断 |
| Q2 Qdrant-only 向量存储 | Qdrant collection 支持 staging、point/payload/fingerprint/schema 校验，以及 active/previous alias 的 publish 和 rollback；运行时不写本地向量 JSON | 进程内 Qdrant 只用于开发/测试；生产需显式 `CODEINSIGHT_QDRANT_URL`，重启恢复、真实集群迁移和旧数据迁移尚未作为本切片的生产验收 |
| Q3 RRF / Top-K | `rrf_k=60`；Dense 粗召回 40；Sparse 粗召回 40；融合候选上限 100；默认最终 Top-K 10 | 参数已进入代码和登记配置；尚无质量 A/B，不声称 10 是最优 |
| Q4 RepositoryMap MCP | 保留 `get_repository_map`，增加目录/符号筛选、files/symbols/imports 选择、上限、序列化预算和 cursor；cursor 绑定 repo、源码指纹、地图版本和筛选条件；Executor 生命周期内缓存并在源码变化后失效 | 只提供轻量 Python AST 文件/符号/import 导航；地图标记 `untrusted=true`、`purpose=navigation_only`，必须回到 `read_file` 或搜索形成 Evidence |
| Q5 LSP / SCIP MCP | 新增 `lsp_definition` 与 `scip_references` 两个只读工具；LSP 使用固定 server registry 和单次进程超时；SCIP 读取固定 `.codeinsight/scip/index.json` 导出并校验仓库指纹、版本和分页；能力缺失返回结构化状态 | 当前机器是否安装 Pyright/TypeScript server、是否有有效 SCIP 导出由运行环境决定；未配置时返回 `NOT_CONFIGURED`/`INDEX_NOT_FOUND`，不宣称已支持 |

## 关键边界

- Q1 已移除 BM25/lexical 检索实现、路由值、测试入口和评测运行器；正常 Router、Auto
  Answer 和 MCP 搜索只走 Dense/Sparse。历史结果文件仍仅作为不可执行的演进证据保留。
- Qdrant payload 保存 `repoId`、`indexVersion`、`path`、行号、source fingerprint、
  chunk version、embedding model 和 vector schema。查询同时过滤当前仓库、当前索引版本和
  `visibility=active`；源码正文仍从绑定仓库读取。
- LSP/SCIP 返回的位置只是导航线索。它们不改变工具目录、权限或系统 Prompt，也不会执行
  被分析仓库代码；Agent 需要继续 `read_file` 并进行应用层校验。
- 生产环境必须配置 `CODEINSIGHT_QDRANT_URL`。开发环境没有 URL 时的 Qdrant in-process
  模式没有持久化/重启保证，只服务于本地契约验证。

## 已执行的窄验证

使用 `backend\.venv\Scripts\python.exe` 和工作树内的 `PYTHONPATH=backend/src` 语义，已通过：

- Q4/Q5 RepositoryMap、ToolExecutor、LSP/SCIP 状态：13 项；
- Q1 Embedding、Dense/Sparse semantic、索引 pipeline、历史 exact 对照：28 项；
- Qdrant Named Vector、Qdrant staging/active/rollback、Q5 状态：10 项；
- 搜索、Auto Answer、Agent、回答映射回归：22 项；
- API、聊天、MCP stdio 和 mainline Tool Loop：20 项；
- `ruff` 受影响源码检查通过，源码 `compileall` 通过。

临时目录权限曾导致 pytest 默认临时目录初始化失败，随后改用工作树内的 pytest basetemp
完成上述测试；这不是产品测试失败。

## 未在本切片中宣称完成的事项

1. 目标 Embedding Provider 的真实 Sparse 返回、usage 和端到端质量尚未重新取证。
2. Qdrant 真实服务重启、旧数据迁移、生产 alias/控制表和 rollback 演练尚未形成发布级证据。
3. 本机 LSP server 安装/版本与 SCIP index 生成器未登记为已验证能力。
4. 2.1.0 的检索质量、延迟、Token 和成本没有用本切片的契约测试替代正式评测。

## 2026-09-10 收尾复核

- **数据集准入恢复 `PASS`**：Q-001 删除 BM25 后，`temporary_30` 清单的 `primary_sparse_mode`
  由 `ast-bm25`/`graph-bm25` 改为 `sparse`；清单内容已用
  `backend/tests/evals/build_temporary_30_ablation_manifest.py` 重建比对（字节一致），随后在
  `baseline_2_1_0.yaml` 与 `dataset_quality_policy_2_1_0.yaml` 中重新登记 `sha256`。
  这次是带原因、带日期的事实登记更新，不是放宽门禁：路径、数量、证据区间和结构校验不变。
- **基线口径修正**：`final_top_k=10` 已按用户 Q-003 决定成为运行时默认
  （`search_repository.py:38`），基线文件把它从“用户候选”改登记为 `runtime_actual`，
  并保留 `5 vs 10` 为未完成的 A/B；旧值 5 改写为“2026-09-10 之前的运行时默认”。
- **本机环境事实（只读检查，不代表能力结论）**：
  - `CODEINSIGHT_EMBEDDING_MODEL` 与 `CODEINSIGHT_EMBEDDING_BASE_URL` 为空，本机没有可用
    Embedding Provider，Q1 的“真实 Provider 返回 Sparse”闸门仍未闭合；
  - `127.0.0.1:6335` 无 Qdrant 响应，Docker 守护进程当前不可访问，Q2 的真实服务行为仍未验证；
  - 本机未安装 `pyright`、`typescript-language-server`、`scip-python`、`scip-typescript`，
    Q5 的两个工具在未配置环境只返回结构化状态，没有本机 `AVAILABLE` 取证。
- **本次复核运行的测试**：后端全量回归 `632 passed, 49 skipped`；
  `ruff check backend/src backend/tests experiments` 通过；`experiments/test_dataset_quality.py`
  与 `backend/tests/evals` 合计 `90 passed, 1 skipped`（含数据集准入回归）。
- 上述三项环境缺口都没有在本次收尾中关闭，不能因为准入恢复 `PASS` 就扩大 Q1/Q2/Q5 的声明。
