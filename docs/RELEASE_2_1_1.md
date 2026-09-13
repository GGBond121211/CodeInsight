# CodeInsight 2.1.1

2.1.1 是 Q-012 收口后的版本：把开发模式「第一次检索必失败」、正文与引用不一致、
快路径白烧一次模型调用、module_inventory 配方必然回退这四类问题修掉，
并闭合上一轮留下的三个未闭合项（Provider 长生成被判超时、两条题面不被原型命中、事件序号竞态）。

## 主要变化

- **检索**：持久化 Qdrant + 按索引身份复用 active 索引。开发模式下第二个进程的首次检索
  从 129,453 ms 降到 7,014 ms；不配 CODEINSIGHT_QDRANT_URL 时行为不变。
- **答案可核验**：证据片段锚定到问题标识符所在行；提示词要求正文点名的文件必须出现在
  citations 里；新增确定性检查与一次纠正重试（CODEINSIGHT_ARCHETYPE_FAST_PATH_ATTEMPTS，
  默认 2、上限 4）。
- **原型路由**：每个原型除示例问题外增加**摘要锚点**；call_sites 摘要改写后，
  「哪些模块引用了 X」这类问法可以命中（0.696 vs 第二名 0.579），而 60 条 holdout +
  12 条负例上的命中、判对、判错、负例错命中四项指标全部不变。
- **module_inventory 配方**：由「先读地图」改为「先检索再读」，不再因为证据条数凑不满而必然回退。
- **模型超时**：聊天请求墙钟上限 60s → 180s，可用 CODEINSIGHT_CHAT_TIMEOUT_SECONDS 覆盖。
  依据是一次 explain 的生成在慢时段实测需要 124 秒；原值会把「生成得慢」记成「调用失败」，
  再触发熔断把同一轮的另一条路线一起打掉。
- **事件流**：事件序号冲突时重读再写（有界重试），父进程与 Worker 并发写同一个 run
  不再以 HTTP 500 收场；真的连续冲突仍然显式失败。
- **前端**：step_started 的 tool_exploration / validation_retry / apply 有固定文案，
  未知阶段保留通用文案。

## 验证

- 后端 tests/unit：790 passed / 1 skipped；ruff 通过。
- 前端：tsc -b、eslint、vitest 18 passed。
- 真实模型小批量：read-timeout-source 与 request-references 的同数据两臂对照都返回
  ANSWERED 且引用命中；会话数据集 regression split 6 case / 30 轮 0 失败。

## 已知边界

- 参数状态仍是 implemented_unoptimized / measured_unoptimized，答案质量没有 A/B 结论。
- redirect-implementation 一类问句仍在阈值下回退（实测近并列：flow_trace 0.4393 vs
  definition_lookup 0.4375），这是「拿不准就回退」的设计行为，由只读 Tool Loop 兜底。
- 多进程容量实验只到 2–4 进程，长跑稳定性未测。
- 详见 docs/CAPABILITIES.md。
