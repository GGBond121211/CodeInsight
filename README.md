# CodeInsight

> 一个能把回答落到源码文件和行号上的本地代码仓库问答工具。

CodeInsight 用来回答陌生代码仓库里的具体问题。你可以用中文、英文或中英混合提问，它会先拆解问题，再去源码里找证据，最后给出带文件路径和行号的回答。

我不希望模型靠印象解释代码，所以把整个过程拆成了一条可以检查的证据链：

```text
用户问题 → QueryPlan → 代码检索 → 独立证据 → 回答生成 → 文件与行号引用
```
目前版本为 `1.0.0`。现在对外只保留 Auto Answer 一个入口。

我目前专注于项目的后端开发，重点是 Python、FastAPI、RAG、检索评测和 LangGraph 工作流。React 前端只用于把后端能力做成一个可以操作的本地演示页面，主要由 Codex 辅助完成；我负责前后端 HTTP API 的边界、接口联调和整体运行流程，不把这个项目当作前端能力展示。

![CodeInsight Smart Answer](docs/assets/codeinsight-smart-answer.png)

## 为什么做这个项目

当你看陌生仓库时，关键词搜索很难处理中文改写、错别字和跨文件调用。把一大堆代码直接塞给模型也不理想：上下文会很乱，模型还可能编造路径，或者漏掉真正需要的文件。

CodeInsight 把这些步骤拆开处理：

- Router 负责整理问题，并把混合问题拆成最多 8 个可以单独回答的子问题。
- 每个子问题都有自己的候选、Top-K 证据、回答、outcome 和引用，不共用一个大证据池。
- BM25 找代码标识符和精确词面，semantic retrieval 补充语义改写。
- RRF/agreement 合并候选，确定性的代码感知 reranker 再做一次排序。
- linear 路径直接逐题回答；Agent 路径多一层 Critic-Reviser，最多修改 5 次。
- 模型只能引用 `E1/E2/...` 这类证据编号。文件路径和行号由后端映射，模型不能自己编。

## 当前流程

```mermaid
flowchart TD
    UI["React Smart Answer"] -->|"POST /api/v1/auto/answer"| API["FastAPI"]
    API --> ROUTER["LLM Router / QueryPlan"]
    ROUTER --> ROUTE{"选择执行路线"}
    ROUTE -->|"insufficient"| STOP["证据条件不足，停止"]
    ROUTE -->|"linear"| LINEAR["逐子问题直接回答"]
    ROUTE -->|"agent"| AGENT["LangGraph Agent"]

    LINEAR --> RETRIEVE["每个子问题独立检索"]
    AGENT --> RETRIEVE
    RETRIEVE --> BM25["BM25 候选"]
    RETRIEVE --> SEM["语义检索候选"]
    BM25 --> RERANK["RRF / agreement + 代码感知重排"]
    SEM --> RERANK
    RERANK --> EVIDENCE["各子问题自己的 Top-K 证据"]
    EVIDENCE --> DRAFT["基于证据生成草稿"]
    DRAFT -->|"linear"| MERGE["按顺序确定性汇总"]
    DRAFT -->|"agent"| CRITIC["Critic → Reviser，最多 5 次"]
    CRITIC --> MERGE
    MERGE --> CITATIONS["经过校验的文件与行号引用"]
```

语义索引以 JSON 缓存在操作系统的本地目录里，不会写进被分析的仓库。没改过的文件可以复用已有向量；Embedding 模型或切块配置改变时，系统会使用新的缓存身份。

## 目前能做什么

- 只读扫描本地仓库，跳过依赖、构建产物和缓存目录。
- 按固定行数切分源码，保留仓库相对路径和从 1 开始的行号。
- 整理多语言问题、识别意图并拆分子问题。
- 使用 BM25 和 semantic 做混合检索。
- 用本地 JSON 持久化语义索引，只重建发生变化的文件，不依赖 MySQL 或向量数据库。
- 使用确定性的 RRF/agreement 融合和代码感知重排。
- 每个子问题单独维护证据，最后按 QueryPlan 顺序汇总。
- 同时保留 linear RAG 和有次数上限的 LangGraph Critic-Reviser 路径。
- 约束模型返回结构化结果，并由应用代码映射引用。
- 提供 CLI、FastAPI 和 React 页面。
- 提供离线单元测试、集成测试和可复现的评测资产。

## 本地运行

### 环境要求

- Python `3.12` or `3.13`
- [uv](https://docs.astral.sh/uv/)
- Node.js `22`
- 一个兼容 OpenAI 接口的聊天模型
- 一个兼容 OpenAI 接口的 Embedding 模型

### 1. 配置模型

参考 `.env.example` 设置环境变量。PowerShell 示例：

```powershell
$env:CODEINSIGHT_API_KEY="<chat-key>"
$env:CODEINSIGHT_MODEL="<chat-model>"
$env:CODEINSIGHT_BASE_URL="<openai-compatible-chat-url>"

$env:CODEINSIGHT_EMBEDDING_API_KEY="<embedding-key>"
$env:CODEINSIGHT_EMBEDDING_MODEL="<embedding-model>"
$env:CODEINSIGHT_EMBEDDING_BASE_URL="<openai-compatible-embedding-url>"
```

密钥只保存在后端进程的环境变量中，不会传给 React 前端，也不会写入仓库。

### 2. 启动后端

```powershell
cd backend
uv sync --locked
uv run uvicorn codeinsight.api.app:app --host 127.0.0.1 --port 8001
```

健康检查地址：`http://127.0.0.1:8001/api/v1/health`

### 3. 启动前端

再打开一个终端：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

浏览器访问 `http://127.0.0.1:5173`。

### 也可以使用 CLI

```powershell
cd backend
uv run codeinsight auto-answer `
  --repo tests/fixtures/sample_repo `
  "用户结账时怎样校验商品编号和数量，之后库存和金额按什么顺序处理？"
```

## HTTP 接口

目前公开两个接口：

- `GET /api/v1/health`
- `POST /api/v1/auto/answer`

请求示例：

```json
{
  "repository_root": "tests/fixtures/sample_repo",
  "question": "用户结账时怎样校验商品编号和数量，之后库存和金额按什么顺序处理？",
  "limit": 5
}
```

响应里会返回 `QueryPlan`、执行路线、逐子问题答案、经过校验的引用、token 用量、Embedding 用量和公开工作流事件，不会返回模型的隐藏推理过程。

## 评测结果

Master-200 一共有 200 个高难问题，覆盖项目 fixture 以及固定版本的 HTTPX、Click 和 Requests。问题里包含中文、英文、中英混合、歧义、错别字和多意图表达。

把长距离证据标注修正为符合产品 80 行切块边界后，结果如下：

| 指标 | Master-200 |
| --- | ---: |
| Outcome 准确率 | 91.50% |
| 引用有效率 | 100.00% |
| 引用精度 | 66.79% |
| 证据召回率 | 86.19% |
| 完整证据覆盖率 | 76.50% |
| 必需术语通过率 | 92.00% |
| 子问题数量准确率 | 90.50% |
| 自动证据约束通过率 | 34.50% |

其中 100 个真实仓库案例的检索漏斗是：

```text
原始候选 98.92%
    ↓
RRF Top-20 95.17%
    ↓
重排后 Top-5 86.42%
    ↓
最终引用 81.17%
```

这些都是自动计算的证据约束指标，不能理解为“人工确认了 91.5% 的回答语义完全正确”。从漏斗看，当前最需要继续改的是最后的证据筛选。

在 Click 8.4.1 上验证持久化索引时，冷启动需要处理 101,866 个仓库代码 Embedding token，热加载时降为 0。每次用户查询本身仍需要调用一次 Embedding。

## 目录结构

```text
backend/               Python 应用、检索、Agent 和测试
frontend/              React + TypeScript 本地页面
docs/assets/           README 使用的产品截图
work/                  本地外部评测仓库，不进入 Git
```

前后端只通过 HTTP JSON 通信。前端不会导入后端源码，后端也不依赖 React 状态。

## 运行检查

后端：

```powershell
cd backend
uv run ruff check src tests
uv run pytest --basetemp=.pytest-release-basetemp
```

前端：

```powershell
cd frontend
npm.cmd run lint
npm.cmd test -- --run
npm.cmd run build
```

自动化测试使用确定性的 fake model 和 fake embedding adapter，不会消耗付费模型额度。



## 后续想做

- 在证据不足时改写查询并重新检索，也就是 Evidence Retrieval Repair Loop。
- 对比 learned reranker 和基于模型的 reranker。
- 本地 JSON 索引遇到实际规模瓶颈后，再评估独立向量库。
- 增加更多编程语言和 monorepo 场景。
- 如果以后支持远程仓库，再单独设计安全边界。


## 开源许可

CodeInsight 使用 [MIT License](LICENSE)。简单说就是：你可以使用、修改和分发代码，但需要保留版权和许可声明，软件本身不提供担保。
