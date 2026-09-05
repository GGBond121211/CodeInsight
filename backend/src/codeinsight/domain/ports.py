"""持久化契约（Ports）。

放在 domain 层而不是 application 层的理由：

    项目的依赖方向是 ``cli/api → application → ingestion/retrieval/domain``，
    以及 ``infrastructure → domain``。如果契约定义在 application 里，
    infrastructure 的实现就得反向 import application，破坏单向依赖。
    定义在 domain 里，两边都只依赖 domain，方向保持干净。

用 ``Protocol`` 而不是抽象基类：结构化类型不要求实现类去继承，测试用的
内存实现可以完全独立存在。这也是项目已有的风格（1.0 用 Callable 类型别名
做同样的事）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from codeinsight.domain.change import (
    ChangeApproval,
    CodeGoal,
    ConversationSession,
    RunSnapshot,
)
from codeinsight.domain.memory import MemoryRecord
from codeinsight.domain.prompt_models import (
    PromptRelease,
    PromptRun,
    PromptTemplate,
    PromptVersion,
)
from codeinsight.domain.trace import AuditRecord, IdempotencyKey, RunEvent


@runtime_checkable
class SessionStore(Protocol):
    """Session 与 Goal 的持久化。"""

    def get_session(self, session_id: str) -> ConversationSession | None: ...

    def save_session(self, session: ConversationSession) -> None: ...

    def get_goal(self, goal_id: str) -> CodeGoal | None: ...

    def save_goal(self, goal: CodeGoal) -> None: ...

    def list_active_goals(self, session_id: str) -> tuple[CodeGoal, ...]: ...


@runtime_checkable
class RunStore(Protocol):
    """Run 快照的持久化。

    ``save_run`` 必须以 CAS（compare-and-set）方式写入：只有当库里的
    ``state_version`` 等于传入快照的前一版本时才更新，否则抛
    ``StateVersionConflictError``。

    这一条是整个 Step 2 的关键。没有它，两个并发的 resume 请求会各自读到
    同一份状态、各自推进一步、后写的覆盖先写的——于是补丁被应用两次。
    普通的「读-改-写」做不到这件事，必须由存储层用带条件的 UPDATE 保证。
    """

    def get_run(self, run_id: str) -> RunSnapshot | None: ...

    def create_run(self, run: RunSnapshot) -> None: ...

    def save_run(self, run: RunSnapshot, *, expected_version: int) -> None: ...

    def list_runs_for_goal(self, goal_id: str) -> tuple[RunSnapshot, ...]: ...


@runtime_checkable
class EventLog(Protocol):
    """事件日志的追加与回放。

    ``append`` 必须保证同一个 run 内 ``sequence`` 连续且唯一。序号有洞
    或重复时回放结果不可信，SSE 的断线续传也会错位。
    """

    def append(self, event: RunEvent) -> None: ...

    def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[RunEvent, ...]: ...

    def next_sequence(self, run_id: str) -> int: ...


@runtime_checkable
class AuditLog(Protocol):
    """独立审计流。永不采样、不与事件日志共享保留策略（增补 R-5）。"""

    def record(self, entry: AuditRecord) -> None: ...

    def read_records(self, run_id: str) -> tuple[AuditRecord, ...]: ...


@runtime_checkable
class ApprovalStore(Protocol):
    """审批令牌的存储与一次性消费。

    ``consume`` 必须是原子的：并发消费同一个令牌，只能有一个成功。
    否则「一次性」就不成立，同一次批准可以被用来应用两次补丁。
    """

    def issue(self, approval: ChangeApproval) -> None: ...

    def get(self, token: str) -> ChangeApproval | None: ...

    def consume(self, token: str, *, now_epoch_ms: int) -> ChangeApproval: ...


@runtime_checkable
class IdempotencyStore(Protocol):
    """幂等键的登记与查询。

    ``register`` 返回 True 表示这是首次登记（调用方应当执行该动作），
    返回 False 表示已经登记过（调用方应当跳过执行并返回既有结果）。
    """

    def register(self, key: IdempotencyKey, *, result_ref: str) -> bool: ...

    def lookup(self, key: IdempotencyKey) -> str | None: ...


@runtime_checkable
class MemoryStore(Protocol):
    """三层 Memory 记录的持久化契约。"""

    def save(self, record: MemoryRecord) -> None: ...

    def get(
        self,
        layer: str,
        owner_id: str,
        record_id: str,
        *,
        tenant_id: str,
        user_id: str,
        repo_id: str,
    ) -> MemoryRecord | None: ...

    def list(
        self,
        layer: str,
        owner_id: str,
        *,
        tenant_id: str,
        user_id: str,
        repo_id: str,
    ) -> tuple[MemoryRecord, ...]: ...

    def delete(
        self,
        layer: str,
        owner_id: str,
        record_id: str,
        *,
        tenant_id: str,
        user_id: str,
        repo_id: str,
    ) -> None: ...


@runtime_checkable
class PromptStore(Protocol):
    """Prompt 模板、版本、Release 和实际运行记录的契约。"""

    def save_template(self, template: PromptTemplate) -> None: ...

    def save_version(self, version: PromptVersion) -> None: ...

    def save_release(self, release: PromptRelease) -> None: ...

    def resolve(
        self,
        prompt_name: str,
        *,
        environment: str,
        tenant_id: str,
    ) -> tuple[PromptVersion, PromptRelease]: ...

    def record_run(self, run: PromptRun) -> None: ...


@runtime_checkable
class CacheStore(Protocol):
    """热点缓存契约。缓存失效不能替代事实 Store。"""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...
