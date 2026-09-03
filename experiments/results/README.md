# 实验结果目录

每次实验一个子目录，命名为 `<experimentId>/`，内含：

| 文件 | 内容 |
| --- | --- |
| `spec.json` | 运行时使用的 ExperimentSpec 快照（不是引用，是拷贝） |
| `results.jsonl` | 逐 case 逐 trial 的原始结果，**不做任何过滤** |
| `summary.json` | 聚合指标、CI、结论分级 |
| `trace/` | 完整 Trace（如适用） |

## 纪律

- **原始 JSONL 永不删除。** 包括失败、超时、STUCK 的记录。DEC-0033 要求失败案例单独统计而非静默丢弃。
- 重跑不覆盖，新建 run-id 子目录。禁止重跑多次后只保留好看的那次。
- `summary.json` 中的结论必须是 `PROVEN` / `DIRECTIONAL` / `NO_EFFECT` / `REGRESSION` 之一。

本目录内容已 gitignore（体积原因），与 `outputs/` 同一惯例。
配置、runner 与 decision_records 进 Git，因为它们才是可复现性的来源。
