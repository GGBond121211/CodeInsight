# 第三方评测仓库说明

CodeInsight 使用以下开源项目的固定版本测试仓库理解能力。源码只会克隆到本地已忽略的 `work/benchmarks/` 目录，供只读分析使用，**不会随本仓库分发**。

仓库中提交的评测资产只包含 CodeInsight 的问题、预期文件与行号、仓库版本信息和汇总结果。记录上游项目名称和源码范围，是为了让评测可以复现。

## HTTPX

- 上游仓库：<https://github.com/encode/httpx>
- 版本：`0.28.1`
- Commit：`26d48e0634e6ee9cdc0533996db289ce4b430177`
- 评测范围：`httpx/`
- 上游许可证：BSD-3-Clause

## Click

- 上游仓库：<https://github.com/pallets/click>
- 版本：`8.4.1`
- Commit：`6eeb50e948ea136db145280f6f5dd52eca3fa7e5`
- 评测范围：`src/click/`
- 上游许可证：BSD-3-Clause

## Requests

- 上游仓库：<https://github.com/psf/requests>
- 版本：`2.34.2`
- Commit：`6e83187b8feb273ed4c6cdab5efd8d54901dfab3`
- 评测范围：`src/requests/`
- 上游许可证：Apache-2.0

如果需要重新分发这些项目的源码，请先阅读各上游仓库的完整版权和许可证文本。CodeInsight 自身代码单独使用本仓库中的 MIT License。
