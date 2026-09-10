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

1. 2.1.0 的检索质量、延迟、Token 和成本没有用本切片的契约测试替代正式评测。
2. Qdrant 旧数据迁移、生产多节点集群和发布级 rollback 演练尚未形成发布级证据。
3. Python 侧 SCIP 生成器在本机不可用（见下文），Python SCIP 引用只有契约测试，
   没有本机导出取证。
4. 仓库指纹只覆盖受支持文本扩展名（`.py/.pyi/.md/.txt/.toml/.yaml/.yml/.json`）；
   `.ts` 不在其中，纯 TypeScript 源码改动不会把 SCIP 索引导向 `INDEX_STALE`。

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

## 2026-09-10 真实环境取证（关闭上述三项环境缺口）

本节关闭上一节记录的三个环境缺口。所有真实调用都在 `codex/2.1.0-development` 工作树内进行，
带请求级输出上限和 `max_retries=0`；没有写入密钥、原始供应商响应、用户隐私或模型隐藏推理。

### Q1：真实 Embedding Provider 返回 Sparse

- 真实端点只有显式声明输出类型时才同时返回 Dense 和 Sparse；不发声明默认只回 Dense。
  适配器因此新增 `CODEINSIGHT_EMBEDDING_OUTPUT_TYPE`，设为 `dense&sparse` 才拿到稀疏坐标。
- 供应商按权重降序返回稀疏项，而 `SparseEmbedding` 契约要求索引升序。适配器新增
  `_sparse_pairs()` 归一化，兼容 `{indices, values}` 和 `[{index, value, token}]` 两种
  真实存在的格式，并在出口统一排序。
- 取证结果：`has_sparse=True`、schema `dense-sparse-v1`、`dense_dim=1024`、
  单请求稀疏项 256/303 个、索引升序、`require_sparse_batch` 通过。
- 边界：这只证明目标 Provider 会真实返回 Sparse，不构成检索质量证据。

### Q2：真实 Qdrant 全流程与重启持久化

- 真实服务为 `127.0.0.1:6335`（Qdrant 1.15.5，容器 `codeinsight-qdrant`，挂载持久化卷）。
- 已取证：staging 写入 → publish → active alias 切换；dense 检索 5 hits；sparse 检索 5 hits；
  改源码产生新 `index_version` 后 rollback 回上一版本成功。
- 已取证：容器 restart 后 collection、alias、point 数量和 green 状态全部保留。
- 演练使用独立 drill 前缀并已清理（残留 collection 0、alias 0），没有污染既有集合。

### Q5：本机 LSP / SCIP

| 工具 | 本机版本 | 状态 |
| --- | --- | --- |
| `pyright` | 1.1.414 | 可用 |
| `pyright-langserver` | 1.1.414 | 可用（Python LSP 的正确入口） |
| `typescript-language-server` | 6.0.0 | 可用 |
| `typescript` | 5.9.3 | 可用 |
| `scip-typescript` | 0.4.0 | 可用 |
| `scip-python` | 0.6.6 | **Windows 不可用** |

- 修复的真实缺陷：`lsp_registry` 原本把 Python 指向 `pyright --langserver --stdio`，
  但 Pyright 1.1.4xx 已移除 `--langserver`，正确入口是 `pyright-langserver --stdio`；
  照原样调用只会拿到 `Unexpected option`。注册表已改到正确入口并加回归测试。
- 真实 LSP 取证：Python 里 `reserve` 的定义正确定位到 `src/shop/inventory.py`，
  `status=AVAILABLE`、`server=pyright-langserver`；TypeScript 的本文件符号与跨文件符号都能定位。
- 真实 SCIP 取证：`scip-typescript` 生成二进制索引，再走受控两步导出
  （`node` 只做反序列化，`python` 落成 `.codeinsight/scip/index.json` 并复用项目自身的
  `repository_fingerprint`），最后经 `ToolExecutor` 调用 `scip_references`。
  跨文件符号 `add()` 返回 3 个位置（`math.ts` 定义、`use.ts` import、`use.ts` 调用）；
  同文件符号返回 2 个位置；按 path/line/column 定位、cursor 分页、`INDEX_STALE`
  （版本不匹配和指纹不匹配两条）与 `INDEX_NOT_FOUND` 都按契约返回。
- **`scip-python` 在 Windows 直接崩溃**：`new RegExp(path.sep)` 没有转义反斜杠，
  生成非法正则 `/\/g/`（Node 报 `Invalid regular expression: /\/g/: \ at end of pattern`）。
  本机因此没有可用的 Python SCIP 生成器，
  Python SCIP 引用保持“契约测试已过、无本机导出取证”的边界。
- **指纹覆盖边界**：`TEXT_EXTENSIONS` 不含 `.ts`，所以只改 TypeScript 源码不会让索引失效。
  对照实验里改动 `tsconfig.json`（受支持扩展名）会正确触发 `INDEX_STALE`，
  但只要编辑落在 `.ts` 上守卫就不响。当前 SCIP 过期守卫对纯 TypeScript 仓库是失效的，
  在扩大 TS 支持声明前必须处理。

### 本节执行的验证

- 受影响窄验证：`test_code_intelligence.py`、`test_embeddings.py`、`tests/unit/retrieval`
  合计 `45 passed`。
- 后端全量回归：`639 passed, 46 skipped`（上一节为 `632 passed, 49 skipped`）。
  差额来自环境缺口关闭后不再跳过的用例：真实 Qdrant 集成测试（`test_qdrant_store.py`）
  和 Docker 可用时的 compose 契约测试转为真实执行。
- 剩余跳过项只有三类，都与本次改动无关：未配置 MySQL（44 项）、
  外部 benchmark 工作树不随公开仓库分发（1 项）、当前 Windows 账户无创建符号链接权限（1 项）。
- 未执行：2.1.0 检索质量 A/B、Token 成本评测，以及 Qdrant 生产集群/迁移演练。
