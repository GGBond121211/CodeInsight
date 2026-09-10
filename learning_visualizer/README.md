# CodeInsight 学习可视化

这是一个完全独立的学习辅助工具，用来把 CodeInsight 的真实请求链画成可点击、可分步播放的节点图。

这是一份公开的静态学习演示，不依赖私有 handoff 文档或运行时服务。

## 边界

- 只用于学习和理解，不执行 CodeInsight 请求。
- 不修改原有 `backend/` 和 `frontend/` 的业务代码。
- 不接入 ComfyUI、Ollama 或任何额外模型。
- 本目录可以在四天学习计划结束后整体删除。

## 运行

可以直接双击 `index.html`，也可以在本目录启动一个静态文件服务：

```powershell
cd learning_visualizer
python -m http.server 4173
```

然后打开：

```text
http://127.0.0.1:4173
```

## 当前内容

- 一次请求主链：从 `App.submit()`、`autoAnswerRepository()`、`postJson()`、`fetch()`，一直展示到后端 `auto_answer()`、Router、检索、回答和引用校验。
- 检索放大：展示 Provider `search_chunks_dense()`、`search_chunks_sparse()`、Qdrant、RRF 和 `rerank_ranked_chunks()` 的调用关系。
- Agent 循环：展示 `run_citation_agent()`、`retrieve()`、`draft()`、`review()`、`route_review()`、`revise()` 和 `finalize_revised()` 的调用与回环。
- 节点第一行使用真实函数名；点击节点可以查看仓库相对路径、内部调用、下一步函数、输入、输出和失败方式。
- 实线箭头表示函数调用；虚线箭头表示 HTTP 边界、返回值或数据传递；橙色虚线表示 Agent 回环。
- 使用“上一步 / 下一步 / 播放”按学习顺序观察节点变化。

## 后续扩展边界

如果以后接入真实运行轨迹，应只增加脱敏的 trace 数据，不让可视化页面重新实现业务流程。建议记录事件顺序、函数名、文件路径、耗时、候选数量、证据 ID 和错误摘要，不记录密钥或隐藏推理。
