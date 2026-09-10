# CodeInsight 2.1.0 数据集准入报告

> 本报告只描述数据准入和证据审查状态，不把模型 pilot 写成最终质量结论。

- 自动准入：`PASS`
- 策略：`experiments\configs\dataset_quality_policy_2_1_0.yaml`
- 基线：`experiments\configs\baseline_2_1_0.yaml`
- 模型质量数据集：`conversation_2_1_0`
- 真实模型限制：每次连续验收周期不超过 10,000,000 tokens；请求级输出预算和 `max_retries=0` 由运行命令控制

## 登记资产

| 数据集 | 数量 | 状态 | SHA-256 / 说明 |
|---|---:|---|---|
| `master_200` | 200 | `READY_FOR_DETERMINISTIC` | `8774ec409c4071ec32d07364316962c77e3be3015d9c1cfd11477aa539cf5935` |
| `change_tasks_l1` | 34 | `STRUCTURALLY_VALID` | `16f8ec7dd6eeb5db76f969487ca2ca680def0ee34d372355fe999262c1fc157a` |
| `swebench_l2` | 80 | `READY_FOR_DETERMINISTIC` | `aee52d5251b6a8f0d5f1e728df5757c040dfdc926dad9744ce4571c0b2b96368` |
| `multilingual_100` | 100 | `STRUCTURALLY_VALID` | `c181ab8dbf00d8fc309f014f895dba5857178bb7109711552ea1cafb96e603ed` |
| `fusion_131` | 131 | `STRUCTURALLY_VALID` | `5442519c227923a4469adcfd59997343bf8b625e823c2ad7ef556ac2ffa8a3a4` |
| `temporary_30` | 30 | `STRUCTURALLY_VALID` | `f9657cc6e91b07bd9dd046a0d8aea754c85efff1849e22effcb73dd7e843cd80` |
| `conversation_2_1_0` | 24 | `READY_FOR_MODEL_PILOT` | `be0ac800909ce2cf3a5ec2fcef42187af2497658499cd0e5d57337872eb0e0f2；turns=123` |

## 多轮数据集结论

- 24 个独立 conversation case、123 个用户轮次；dev/regression/golden/holdout 各 6 条，中文 18 条、中英混合 6 条。
- Evidence 路径、1-based 行区间、case/turn ID、split 成员、跨 case 问题重复和禁存模型内容门禁通过。
- 24/24 条 case 已完成 `codex-agent-source-audit` 源码审查，覆盖率 100%；这不是独立人工双盲审查。
- 当前状态是 `READY_FOR_MODEL_PILOT`，允许小规模真实 Provider pilot；没有独立人工复核的最终语义结论仍保持阻断。
- change 轮只验证“生成预览并等待审批、不得直接改 fixture”；未审批不计入修改成功。

## 现有缺口仍然保留

- Master-200、multilingual 和 fusion 仍是单轮数据，不能代替多轮上下文数据，也没有独立 comprehension holdout。
- L1 change manifest 的部分任务仍是 `described_not_written`，不能进入 patch success 分母。
- L2 frozen-80 只支持文件/行定位结论，不等同完整 SWE-bench resolve。
- 真实 pilot 只报告 route、事实/证据命中、目标连续性、上下文压缩、token、缓存和延迟；不使用 Fake Provider 代表模型质量。

## 停止条件

1. 自动准入出现 `DATASET_BLOCKED` 时停止实验，不自动刷新 hash 或跳过案例。
2. pilot 使用 dev/regression/golden；holdout 只在参数和 Prompt 冻结后使用。
3. 只有独立审查和 holdout 都满足，状态才可升级为 `READY_FOR_FINAL`。
