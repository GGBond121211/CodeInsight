# CodeInsight 前端

这是一个用 React、TypeScript 和 Vite 写的本地演示页面，提供统一多轮 Conversation 入口，并展示实时阶段事件、当前模型/降级模型、Gateway 用量和调用明细。前端通过 `/api/v2/chat/*` 维持同一 Session 的连续对话，可交错普通聊天、代码理解和代码修改；一轮只展示 `assistant_message`，模型、Router、引用、缓存和 usage 进入可折叠详情，不再重复显示同一回答。也保留 `/api/v1/auto/answer` 的 Smart Answer 入口。模型密钥只配置在 FastAPI 后端，浏览器不会保存或发送密钥。

这个前端主要用于展示后端能力，由 Codex 辅助完成。项目的开发重点仍是 Python 后端、检索、RAG 和 LangGraph 工作流。

## 本地开发

```powershell
npm.cmd install
npm.cmd run dev
```

浏览器打开 `http://127.0.0.1:5173`。

Vite 默认将 `/api` 代理到 `http://127.0.0.1:8001`。如果后端运行在其他端口，
先设置 `VITE_API_BACKEND_URL`，避免页面请求落到错误的服务实例：

```powershell
$env:VITE_API_BACKEND_URL="http://127.0.0.1:8002"
npm.cmd run dev -- --port 5174
```

先在 `backend/` 启动 FastAPI：

```powershell
uv run uvicorn codeinsight.api.app:app --host 127.0.0.1 --port 8001
```

## 验证

```powershell
npm.cmd run lint
npm.cmd test -- --run
npm.cmd run build
```
