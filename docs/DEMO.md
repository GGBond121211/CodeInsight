# 最小演示

在 `backend/` 下运行：

```powershell
uv run codeinsight change-demo --work-dir ..\work
```

该命令创建固定的临时样例，打印证据和 diff，要求用户输入 `APPLY` 后才调用固定 Sandbox；输入 `ROLLBACK` 可恢复隔离 workspace。源样例不会被修改。Sandbox 不可用时应保留可查询的失败状态，不代表真实模型修复质量。
