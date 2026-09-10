# CodeInsight 2.1.0 Q-008 / Q-009 升级切片

本文件记录 `codex/2.1.0-development` 工作树中 Q-008（代码理解迁移到 MCP Tool Loop）
与 Q-009（Evidence Retrieval Repair Loop）的实现边界。它是本地升级切片的事实入口，
不把契约测试当成效果证据。

## 已实现的运行时变化

| 计划 | 本次实现 | 当前证据边界 |
| --- | --- | --- |
| Q-008 Evidence Ledger | `application/evidence_ledger.py`：工具结果按 `path + 行区间 + 指纹` 去重并分配 `E1/E2`；只有 search/get_evidence_context/read_file 能成为证据 | 契约测试覆盖；未做质量对照 |
| Q-008 只读 Tool Loop | `agent/code_understanding_tool_loop.py`：MCP discovery 结果先按只读白名单收窄，写工具在视图层直接拒绝 | 契约测试覆盖写工具暴露为 0 |
| Q-008 答案契约 | 复用既有 `{outcome, answer, citations}`，`citations` 只能取应用回填过的编号 | 未知编号会被拒绝；两次纠正后失败关闭 |
| Q-008 路由接入 | `explain` 路线新增 `CODEINSIGHT_CODE_UNDERSTANDING_TOOL_LOOP` 开关 | **默认关闭**，见下文「未完成」 |
| Q-009 确定性评估 | `application/evidence_assessor.py`：sufficient / insufficient / contradictory / invalid | 契约测试覆盖；不做文本相似度判定 |
| Q-009 Repair 控制器 | `application/evidence_repair.py`：四种动作、预算分账、动作不重复 | 契约测试覆盖 |
| Q-009 控制层接缝 | `agent/tool_loop.py` 新增 `ToolOutcomeController`，在 ToolResult 之后、模型收尾之前介入 | 不传控制层时行为与之前完全一致 |
| Q-009 公开事件 | `evidence_assessed`、`evidence_repair_started/finished` | 只记状态、计数和动作 |

## 关键边界

- 代码理解路线**只读**：`get_repository_map`、`search_repository`、`read_file`、
  `get_evidence_context`、`lsp_definition`、`scip_references`。
  `generate_patch`、`validate_patch`、`apply_patch_isolated`、
  `run_allowlisted_checks`、`rollback_workspace` 不进目录，调用也会被视图层拒绝。
  代码修改仍走独立的受控路径，approval / 隔离 workspace / Sandbox / rollback 不变。
- `get_repository_map`、`lsp_definition`、`scip_references` 只提供导航线索，
  不构成可引用证据；必须回到 `read_file`。
- 模型不能自报引用：`citations` 只接受应用回填过的 `E` 编号，未知编号被拒绝。
- `invalid`（位置越界或畸形）既不进入 Repair 也不放行，直接按安全边界终止。
- Repair 的轮数、动作与 Tool Loop 的步数、工具调用数、deadline 分别计数，
  不与 Patch Repair 混算。

## 与计划的两处偏离（有意）

1. **答案字段沿用 `citations` 而不是新增 `evidence_ids`。**
   项目已有 `parse_model_answer` 契约和整套调用方；并行引入第二套字段会让
   「模型只能返回 E 编号」这条硬门槛出现两个实现。语义相同，字段名沿用既有契约。
2. **`explain` 默认仍走原路径。** 计划 Task 5/6 要求先做同数据、同模型对照，
   通过后才切默认。对照尚未运行（需要真实模型与固定集），因此开关默认关闭。

## 已执行的窄验证

使用 `backend\.venv\Scripts\python.exe` 与工作树内 `PYTHONPATH=backend/src`：

- `test_evidence_ledger.py`：9 项（去重、越界拒绝、额度上限、编号校验）
- `test_evidence_assessor.py`：9 项（四状态、锚点、冲突、invalid 优先）
- `test_evidence_repair.py`：15 项（动作选择、不重复、预算耗尽、序号校验）
- `test_code_understanding_tool_loop.py`：14 项（只读边界、证据映射、
  未知编号、降级、Repair 接入、取消、超时与步数上限）
- `test_code_understanding_route.py`：4 项（开关默认关闭、映射与 payload）
- `test_evidence_repair_events.py`：3 项（事件字段与顺序）
- 后端全量回归：`702 passed, 46 skipped`（Q-006/Q-007 之后为 `639 passed`），
  `ruff check backend/src backend/tests` 通过。

Windows 上 `time.monotonic()` 分辨率约 15.6ms，亚毫秒 deadline 会让超时断言变成
随机结果；超时测试改用真实耗时（工具内 sleep）触发。

## 未完成 / 未宣称

1. **没有质量对照。** LangGraph / linear / Tool Loop / Tool Loop + Repair 的同数据
   比较（计划 Task 5、Q-009 Task 6）尚未运行，因此不能声称新路线更好或更差。
2. **`explain` 默认未切换。** 开关默认关闭，生产行为与 2.1.0 之前一致。
3. **Repair 轮数没有调优。** 默认 `max_repair_rounds=1` 是工程基线，
   0/1/2 的对照未做，不声称 1 是最优值。
4. **无真实模型端到端取证。** 现有证据全部是 Fake Provider 契约测试；
   真实 Provider 下的工具选择行为、Token 成本和延迟未测。
5. **Embedding 用量不回流。** MCP Server 在子进程内做 Embedding，
   工具循环路径的 `embedding_input_tokens` 记 0，成本统计不完整。
