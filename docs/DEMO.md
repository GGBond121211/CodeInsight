# 最小演示

在 `backend/` 下运行：

```powershell
uv run codeinsight change-demo --work-dir ..\work
```

该命令创建固定的临时样例，打印证据和 diff，要求用户输入 `APPLY` 后才调用固定 Sandbox；输入 `ROLLBACK` 可恢复隔离 workspace。源样例不会被修改。Sandbox 不可用时应保留可查询的失败状态，不代表真实模型修复质量。

## Conversation 四轮验收

先启动后端和前端，仓库路径填写 `tests/fixtures/sample_repo`，在同一个 Conversation Session 依次输入：

1. `请解释 checkout 如何校验输入？` —— 进入 `explain`，返回一份带引用的代码回答。
2. `那这个地方是不是有问题？` —— 继承 active code goal，继续进入只读分析，不会误进普通聊天。
3. `那你把这个 bug 修改一下。` —— 进入 `change`，生成 diff 后等待用户审批；未审批前不写源仓库。
4. `那你看看现在还有没有 bug。` —— 基于已批准的隔离 workspace 重新检查。



可穿插验证：在任意代码轮次之间输入 `你好` 或 `你能做什么？`。它应进入 `general_chat`，返回自然语言；页面只显示一份回答，模型、缓存和 usage 位于“普通对话 · 查看模型与用量”详情中。输入没有明确对象的 `这个东西怎么样` 时，应进入 `clarify`，不扫描仓库也不调用修改工具。

## 异步 Agent Run 验收（2.1.0 开发分支）

同一套对话入口，现在每一步都应先看到「受理」再看「跑完」：

1. 提交一条代码问题。接口立刻返回 202 和 `turn_id`；页面阶段事件随后出现
   `task_queued` → `worker_claimed` → `retrieval_started` → `answer_ready`，最后到终态。
   中途刷新页面或断网再连，事件应从断点继续而不是从头再来。
2. 提交一条修改请求。停在 `WAITING_APPROVAL` 且源文件未变；点批准后进入
   `WAITING_VALIDATION`，校验 Worker 跑完才出现 `COMPLETED`。
3. 验收「不会假装成功」：把校验环境停掉再批准，Run 应停在 `MANUAL_REQUIRED`，
   变更结果仍是 `WAITING_VALIDATION`。
4. 验收「不会重复执行」：对同一轮重复点批准，第二次必须被拒绝，不会出现第二份补丁。
