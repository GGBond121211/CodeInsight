# CodeInsight 前端

这是一个用 React、TypeScript 和 Vite 写的本地演示页面，目前只保留 Smart Answer。前端每次提交问题只调用 `POST /api/v1/auto/answer`。以前的 Grounded Answer、Verified Agent 和 Inspect Retrieval 页面已经关闭，对应的后端引擎仍由 Auto Answer 内部调用。模型密钥只配置在 FastAPI 后端，浏览器不会保存或发送密钥。

这个前端主要用于展示后端能力，由 Codex 辅助完成。项目的开发重点仍是 Python 后端、检索、RAG 和 LangGraph 工作流。

## 本地开发

```powershell
npm.cmd install
npm.cmd run dev
```

浏览器打开 `http://127.0.0.1:5173`。

Vite 将 `/api` 代理到 `http://127.0.0.1:8001`。先在 `backend/` 启动 FastAPI：

```powershell
uv run uvicorn codeinsight.api.app:app --host 127.0.0.1 --port 8001
```

## 验证

```powershell
npm.cmd run lint
npm.cmd test -- --run
npm.cmd run build
```
