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
| Q-008 路由接入 | 对话 / HTTP / CLI 三条入口收口到 `application/code_understanding_route.py`，`explain` 一律走只读 Tool Loop，不再有开关 | 契约测试覆盖三条入口不再引用旧路线；无真实模型取样 |
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

## 老路封闭（2026-09-10）

用户决定：旧的 LangGraph 代码理解路线**弃用**，以后全部走只读 Tool Loop；
老路代码暂时不删，但测试和正式运行都不再使用它。

- **代码保留，入口清零。** `agent/workflow.py:run_citation_agent` 与
  `application/agent_answer_repository.py` 原样保留在仓库里作为历史实现，
  但没有任何运行入口再调用它们。原先三份各写一遍的
  `execution_route == "agent"` 分支已收成一个
  `run_code_understanding_answer()`，不存在「某处还没切过来」的暗门。
- **配置开关删除。** `CODEINSIGHT_CODE_UNDERSTANDING_TOOL_LOOP` 及其读取函数
  已从代码和 `.env.example` 移除。没有开关就没有「生产还在走老路」这种状态，
  也不会出现开关打开但某条入口忘了检查的静默回退。
- **老路的测试退役。** `tests/unit/agent/test_workflow.py`、
  `tests/unit/agent/test_query_plan_workflow.py`、
  `tests/integration/test_agent_answer_repository.py` 标为 `legacy`，
  由 `backend/pyproject.toml` 的 `addopts = "-m 'not legacy'"` 排除在默认回归之外；
  需要复核历史实现时显式执行 `pytest -m legacy`。退役而不是删除，是因为冻结代码
  仍可能作为回退参照，但在默认回归里继续跑它就等于让弃用路线保持「活着」的假象。
- **守卫测试替代人工纪律。** `test_code_understanding_route.py` 断言三条入口
  文件里既没有 `from codeinsight.agent.workflow import` 也没有 `run_citation_agent(`，
  同时断言 `agent/workflow.py` 里 `def run_citation_agent` 仍然存在（冻结而非删除）。
- **老路评测脚本一并退役。** `tests/evals/run_agent_answer_eval.py`、
  `run_fusion_answer_eval.py`、`run_multilingual_answer_eval.py`、
  `run_master_200_eval.py` 仍然能跑（它们直接调用冻结实现，不经过入口），
  但不作为 2.1.0 的结论来源，2.1.0 的取证只认新路。仍有几个
  `tests/evals/test_*_answer_eval_runner.py` 在跑：它们只测 payload 与指标
  这类纯函数，不调用老路，因此没有退役。

仍未做：老路的**质量对照**不再有可比对象——它在封闭时也没有同数据、同模型的
真实取样，因此本文不声称新路比老路更好或更差。

## 与计划的两处偏离（有意）

1. **答案字段沿用 `citations` 而不是新增 `evidence_ids`。**
   项目已有 `parse_model_answer` 契约和整套调用方；并行引入第二套字段会让
   「模型只能返回 E 编号」这条硬门槛出现两个实现。语义相同，字段名沿用既有契约。
2. **`explain` 直接弃用老路，而不是「对照通过后切默认」。** 计划 Task 5/6 要求
   先做同数据、同模型对照，通过后才切默认。用户 2026-09-10 直接决定老路弃用，
   因此没有「切默认」这一步：`explain` 只有只读 Tool Loop 一条路。
   代价必须写明——这次切换是**产品决策**，不是实验结果；不能用「对照通过」
   来支撑它，本文也不声称新路质量优于老路。

## 已执行的窄验证

使用 `backend\.venv\Scripts\python.exe` 与工作树内 `PYTHONPATH=backend/src`：

- `test_evidence_ledger.py`：9 项（去重、越界拒绝、额度上限、编号校验）
- `test_evidence_assessor.py`：9 项（四状态、锚点、冲突、invalid 优先）
- `test_evidence_repair.py`：15 项（动作选择、不重复、预算耗尽、序号校验）
- `test_code_understanding_tool_loop.py`：14 项（只读边界、证据映射、
  未知编号、降级、Repair 接入、取消、超时与步数上限）
- `test_code_understanding_route.py`：7 项（映射与 payload、非 ANSWERED 状态
  一律收敛为证据不足、公开事件只带计数、三条入口的旧路线封闭守卫、冻结实现仍在）
- `test_evidence_repair_events.py`：3 项（事件字段与顺序）
- 后端全量回归：`697 passed, 46 skipped, 8 deselected`
  （封闭前为 `702 passed, 46 skipped`；差额来自老路用例退役，
  其余 5 项差额由路线测试从 4 项扩到 7 项抵消），
  `ruff check backend/src backend/tests` 通过。
  受影响窄验证：`test_auto_answer_api.py` + `test_code_understanding_route.py`
  `10 passed`。

Windows 上 `time.monotonic()` 分辨率约 15.6ms，亚毫秒 deadline 会让超时断言变成
随机结果；超时测试改用真实耗时（工具内 sleep）触发。

## 未完成 / 未宣称

1. **没有质量对照。** linear / Tool Loop / Tool Loop + Repair 的同数据
   比较（计划 Task 5、Q-009 Task 6）尚未运行，因此不能声称新路线更好或更差。
2. **新路没有真实模型取样。** 老路已封闭，但新路目前也只有 Fake Provider
   契约测试；真实 Provider 下的工具选择、严格答案成功率和 Token 成本未测。
3. **Repair 轮数没有调优。** 默认 `max_repair_rounds=1` 是工程基线，
   0/1/2 的对照未做，不声称 1 是最优值。
4. **Embedding 用量不回流。** MCP Server 在子进程内做 Embedding，
  工具循环路径的 `embedding_input_tokens` 记 0，成本统计不完整。
