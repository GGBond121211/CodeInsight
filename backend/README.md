# CodeInsight 后端

后端对外提供 `auto-answer` 和统一 `chat` 两个业务入口，并提供用量观测与变更闭环接口。一次问答先经过 Router，由它整理并拆分问题，再选择内部 linear 或 LangGraph Agent 引擎；统一 chat 在同一 Session 内保留多轮上下文，并按任务类型进入解释或变更路径。原来的 `search`、`answer` 和 `agent-answer` 仍供 Auto Answer 内部复用，但对应的 HTTP 路由和 CLI 命令已经停用。

检索链是 `BM25 + semantic → RRF/代码感知 reranker`。AST-BM25、graph-BM25 以及相关的 AST 分块、静态调用图和运行脚本已经删除。语义索引缓存在本地 JSON 中，没改过的文件可以直接复用向量。一次请求只加载或构建一个 SemanticIndex，供所有子问题使用；候选、重排结果和证据覆盖仍按子问题分开保存。

## 配置

```powershell
$env:CODEINSIGHT_API_KEY="<chat-key>"
$env:CODEINSIGHT_MODEL="<chat-model>"
$env:CODEINSIGHT_BASE_URL="<chat-compatible-base-url>"
# reasoning 与最终结构化回答共用输出额度，默认 40960；可按 Provider 调整
# $env:CODEINSIGHT_MAX_OUTPUT_TOKENS="40960"
# 可选：当前 BASE_URL 确实支持的备用模型；不填则关闭自动降级
# $env:CODEINSIGHT_FALLBACK_MODELS="<same-provider-model>,<same-provider-model-2>"
$env:CODEINSIGHT_EMBEDDING_API_KEY="<embedding-key>"
$env:CODEINSIGHT_EMBEDDING_MODEL="<embedding-model>"
$env:CODEINSIGHT_EMBEDDING_BASE_URL="<embedding-compatible-base-url>"
# 可选：覆盖操作系统默认的本地语义索引目录
# $env:CODEINSIGHT_SEMANTIC_CACHE_DIR="<cache-directory>"
```

Gateway 的模型注册表包含能力和价格目录，不代表同一个兼容端点支持全部模型。
运行时默认只调用主模型；只有显式设置 `CODEINSIGHT_FALLBACK_MODELS` 才启用降级，
而且这些模型必须由同一个 `CODEINSIGHT_BASE_URL` 承载。启动时可访问 Gateway
`/health` 查看实际生效的主模型和降级链。

变更对话在生成补丁前会执行 Sandbox preflight，先检查 Docker daemon 和固定校验镜像；
如果 Docker 不可用，前端会显示具体阻塞原因，模型不会先生成一个无法验收的补丁。
如果补丁已经应用但校验因 Sandbox 基础设施失败进入 `REVIEW_REQUIRED`，同一 Session
输入“继续”会先重新校验现有隔离 workspace，不会重复生成或重复审批同一补丁；只有代码
本身导致的固定检查失败，才会把脱敏失败摘要交给模型生成新的修复补丁。

## 当前命令

```powershell
uv sync --locked
uv run codeinsight auto-answer --repo tests/fixtures/sample_repo "How does checkout validate input?"
uv run uvicorn codeinsight.api.app:app --host 127.0.0.1 --port 8001
```

HTTP 当前注册：

- `GET /api/v1/health`
- `POST /api/v1/auto/answer`
- `GET /api/v1/usage/summary`
- `GET /api/v1/usage/calls`
- `/api/v2/chat/*` 多轮对话、SSE 事件和审批接口
- `/api/v2/change/*` 变更预览、审批、应用、校验、回滚和事件接口

## 验证

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run ruff check src tests
uv run pytest --basetemp=.pytest-release-basetemp
```

历史检索和回答实验结果保留在 `docs/` 与 `outputs/`，它们是演进证据，不代表当前公开入口或当前运行时检索器。
