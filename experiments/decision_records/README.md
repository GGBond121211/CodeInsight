# 决策记录目录

每个重要选型一份 Markdown，回答同一组问题：

1. **问题**：在选什么？
2. **候选**：A、B、（C）分别是什么？
3. **实验**：ExperimentSpec ID、数据集、Trial 数、判据
4. **结果**：主指标 + **代价**（延迟 / token / 成本 / 复杂度）
5. **结论**：PROVEN / DIRECTIONAL / NO_EFFECT / REGRESSION
6. **何时切换**：什么条件下该换成 B
7. **如何回滚**：具体步骤

## 与 docs/DECISIONS.md 的分工

- `docs/DECISIONS.md` —— 项目级决策，含非实验类（边界、范围、流程）
- 本目录 —— **实验驱动**的技术选型，每份对应一个或一组 ExperimentSpec

两者交叉引用：DEC 记录指向本目录的详细报告，本目录回指 DEC 编号。

完成后同步更新 `docs/DECISION_QA_MATRIX.md` 对应行的状态与证据链接。
