# CodeInsight 2.1.0 Q1–Q5 升级切片

本文件记录 `codex/2.1.0-development` 工作树中 Q1–Q5 的实现边界。它是本地升级
切片的事实入口，不把契约测试、进程内 Qdrant 或未配置的外部后端写成生产质量证据。

## 已实现的运行时变化

| 计划 | 本次实现 | 当前证据边界 |
| --- | --- | --- |
| Q1 Dense/Sparse 替换 BM25 | EmbeddingBatch 可表达 Provider Dense + Sparse；Sparse 坐标/权重、批次顺序和响应格式受校验；Qdrant 使用 `dense`/`sparse` Named Vectors；正常 `hybrid` 路径用两路召回、RRF 和 qwen Rerank；没有 Sparse 时受控失败 | Fake Provider 契约和本地 Qdrant 契约已测；目标 Provider 是否真的返回 Sparse 仍需真实最小请求确认，不能由模型名推断 |
| Q2 Qdrant-only 向量存储 | 默认应用组装不再调用 LocalJsonVectorStore 或 persistent semantic JSON；Qdrant collection 支持 staging、point/payload/fingerprint/schema 校验、active 小型 manifest、publish 和 rollback；Local JSON 仅作为历史/离线对照 | 进程内 Qdrant 只用于开发/测试；生产需显式 `CODEINSIGHT_QDRANT_URL`，重启恢复、真实集群迁移和旧缓存 backfill 尚未作为本切片的生产验收 |
| Q3 RRF / Top-K | `rrf_k=60`；Dense 粗召回 40；Sparse 粗召回 40；融合候选上限 100；默认最终 Top-K 10 | 参数已进入代码和登记配置；尚无质量 A/B，不声称 10 是最优 |
| Q4 RepositoryMap MCP | 保留 `get_repository_map`，增加目录/符号筛选、files/symbols/imports 选择、上限、序列化预算和 cursor；cursor 绑定 repo、源码指纹、地图版本和筛选条件；Executor 生命周期内缓存并在源码变化后失效 | 只提供轻量 Python AST 文件/符号/import 导航；地图标记 `untrusted=true`、`purpose=navigation_only`，必须回到 `read_file` 或搜索形成 Evidence |
| Q5 LSP / SCIP MCP | 新增 `lsp_definition` 与 `scip_references` 两个只读工具；LSP 使用固定 server registry 和单次进程超时；SCIP 读取固定 `.codeinsight/scip/index.json` 导出并校验仓库指纹、版本和分页；能力缺失返回结构化状态 | 当前机器是否安装 Pyright/TypeScript server、是否有有效 SCIP 导出由运行环境决定；未配置时返回 `NOT_CONFIGURED`/`INDEX_NOT_FOUND`，不宣称已支持 |

## 关键边界

- Q1 删除的是默认运行路径中的 BM25 依赖，不删除 `bm25.py`、历史评测数据或历史 exact
  对照。`lexical`/`bm25` 只能由显式历史调用使用，正常 Router、Auto Answer 和 MCP 搜索
  走 Dense/Sparse。
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
2. Qdrant 真实服务重启、旧 local JSON backfill、生产 alias/控制表和 rollback 演练尚未形成
   发布级证据。
3. 本机 LSP server 安装/版本与 SCIP index 生成器未登记为已验证能力。
4. 2.1.0 的检索质量、延迟、Token 和成本没有用本切片的契约测试替代正式评测。
