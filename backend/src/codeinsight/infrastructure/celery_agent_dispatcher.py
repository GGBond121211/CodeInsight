"""Celery 侧的 Agent Run 投递实现。

broker 只负责把消息送到 Worker；Run 的状态事实在 Store 里。这个方向不能反过来：
Celery 的 result backend 会按过期策略清理记录，把它当成业务状态来源，重启之后就
再也说不出「那一轮到底跑没跑」。

任务名与队列名集中在这里，Worker 与 API 引用同一份常量——两边各写一个字符串，
改名时就只会改一边。
"""

from __future__ import annotations

from celery import Celery

from codeinsight.domain.agent_run import AgentRunTask
from codeinsight.infrastructure.task_queue import TaskEnvelope

AGENT_RUN_TASK_NAME = "codeinsight.run_agent"
# 计划里写的是 codeinsight.run_validation，线上 Worker 注册的实际名字是下面这个
# （compose 与 k8s 都按它启动，集成测试也断言它）。改名要连部署一起改，
# 所以这里记录现状，不制造第二个名字。
VALIDATION_TASK_NAME = "codeinsight.run_registered_validation"

# 队列暂不拆分：compose 与 k8s 启动的是默认队列上的 Worker，把消息发到没人
# 监听的队列，等于「已排队但永远不会执行」——正是这一层要避免的状态。按职责
# 分队列（agent_run / validation）要连启动命令一起改成 celery -Q，那一步属于
# 部署，不在代码里假装已经生效。


class CeleryAgentRunTransport:
    """用 send_task 投递：application 层不必 import Worker 模块。"""

    def __init__(self, app: Celery) -> None:
        self._app = app

    def publish(self, task: AgentRunTask) -> str:
        result = self._app.send_task(AGENT_RUN_TASK_NAME, kwargs=task.as_payload())
        return str(result.id)


class CeleryValidationTransport:
    """用 send_task 投递一条固定校验：消息里只有 task_id。

    要跑哪个 profile、校验哪个隔离 workspace，都留在登记过的队列事实里。把命令写进
    消息，等于让投递方（以及任何能写消息的东西）决定执行什么。
    """

    def __init__(self, app: Celery) -> None:
        self._app = app

    def publish(self, task: TaskEnvelope) -> str:
        result = self._app.send_task(VALIDATION_TASK_NAME, kwargs={"task_id": task.task_id})
        return str(result.id)
