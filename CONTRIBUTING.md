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

## 分支与工作树生命周期

项目采用短生命周期版本分支流程。`main` 是长期保留、始终指向已合并成果的稳定主线；它不是可以随意删除的临时工作树。

每个版本按下面的顺序开发：

```text
main
  -> release/2.0.3（从最新 main 创建）
  -> 本地开发、提交、测试和修复
  -> Pull Request 合并回 main
  -> 创建 v2.0.3 Git tag 和正式 Release
  -> CI/CD 全部通过
  -> 删除 release 分支和对应的临时工作树
```

- 版本分支建议使用 `release/2.0.x` 命名；自动化开发环境可以使用 `codex/2.0.x-development`，但仍遵循同一生命周期。
- 本地提交不等于推送。没有明确发布指令时，只提交当前版本分支，不推送 GitHub。
- 合并前必须经过受影响测试；合并或 tag 触发的 CI/CD 未全部通过时，保留版本分支用于定位问题。
- `v2.0.x` tag 和 GitHub Release 是版本的永久锚点；删除版本分支不会删除已经合并的提交或 tag。
- Git 分支和 Git worktree 是两件事：分支是提交历史引用，worktree 是某个分支在磁盘上的检出目录。删除 worktree 不等于删除分支。
- 根目录是仓库的 primary worktree。它可能当前检出的是历史开发分支，也可能有未提交修改；删除它会直接丢失本地工作，未经核对不得删除、重置或清理。
- 删除任何旧 worktree 前，先执行 `git worktree list` 和该目录的 `git status --short`。只清理已合入、工作区干净且没有被进程占用的临时版本工作树；含未提交内容的工作树必须先迁移、提交或经确认放弃。

当前单人串行开发只保留 `main` 与正在开发的一个版本分支。需要并行实验时才额外创建 worktree；实验结束后先确认成果归属，再清理目录和已结束的版本分支。

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
