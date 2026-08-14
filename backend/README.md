# CodeInsight 后端

后端对外只有 `auto-answer` 一个业务入口。一次请求先经过 Router，由它整理并拆分问题，再选择内部 linear 或 LangGraph Agent 引擎。原来的 `search`、`answer` 和 `agent-answer` 仍供 Auto Answer 内部复用，但对应的 HTTP 路由和 CLI 命令已经停用。

检索链是 `BM25 + semantic → RRF/代码感知 reranker`。AST-BM25、graph-BM25 以及相关的 AST 分块、静态调用图和运行脚本已经删除。语义索引缓存在本地 JSON 中，没改过的文件可以直接复用向量。一次请求只加载或构建一个 SemanticIndex，供所有子问题使用；候选、重排结果和证据覆盖仍按子问题分开保存。

## 配置

```powershell
$env:CODEINSIGHT_API_KEY="<chat-key>"
$env:CODEINSIGHT_MODEL="<chat-model>"
$env:CODEINSIGHT_BASE_URL="<chat-compatible-base-url>"
$env:CODEINSIGHT_EMBEDDING_API_KEY="<embedding-key>"
$env:CODEINSIGHT_EMBEDDING_MODEL="<embedding-model>"
$env:CODEINSIGHT_EMBEDDING_BASE_URL="<embedding-compatible-base-url>"
# 可选：覆盖操作系统默认的本地语义索引目录
# $env:CODEINSIGHT_SEMANTIC_CACHE_DIR="<cache-directory>"
```

## 当前命令

```powershell
uv sync --locked
uv run codeinsight auto-answer --repo tests/fixtures/sample_repo "How does checkout validate input?"
uv run uvicorn codeinsight.api.app:app --host 127.0.0.1 --port 8001
```

HTTP 当前只注册：

- `GET /api/v1/health`
- `POST /api/v1/auto/answer`

## 验证

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run ruff check src tests
uv run pytest --basetemp=.pytest-release-basetemp
```

历史检索和回答实验结果保留在 `docs/` 与 `outputs/`，它们是演进证据，不代表当前公开入口或当前运行时检索器。
