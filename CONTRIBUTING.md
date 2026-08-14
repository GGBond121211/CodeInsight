# 参与 CodeInsight 开发

CodeInsight 目前只做一件事：根据本地仓库里的真实证据回答代码问题。准备提交改动前，请先确认它确实能改善 Auto Answer、证据质量、评测可信度或开发体验。

## 开发环境

后端：

```powershell
cd backend
uv sync --locked
uv run ruff check src tests
uv run pytest --basetemp=.pytest-release-basetemp
```

前端：

```powershell
cd frontend
npm.cmd install
npm.cmd run lint
npm.cmd test -- --run
npm.cmd run build
```

自动化测试使用 fake chat 和 fake embedding adapter，不需要付费模型密钥。

## 提交 Pull Request

1. 如果改动会改变公开行为或增加依赖，请先开 Issue 说明原因。
2. 一个 Pull Request 只解决一个问题，别顺手重构无关代码。
3. 为改动补上最直接的测试，不用为了覆盖率堆测试。
4. 运行受影响的后端或前端检查。
5. 用户能看到的行为发生变化时，同步更新文档。
6. 文档里要分清当前实现、历史实验和后续计划。

## 不在当前范围内的改动

- 不要执行或修改被分析仓库的代码。
- 不要提交复制来的第三方评测仓库，它们应该放在已忽略的 `work/` 目录中。
- 没有明确产品需求时，不增加生产基础设施、登录系统或多租户抽象。
- 没有冻结案例和可复现数据时，不要宣称指标有所提升。

## 密钥与评测数据

不要提交 API Key、Token、`.env`、私人接口地址、请求头或模型隐藏推理。配置项名称和占位值统一写在 `.env.example`。

真实模型评测可能包含用户问题和模型回答。提交前请先检查内容，只保留复现或核对已有结论所需的产物。
