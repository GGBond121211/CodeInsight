"""ORM 表定义的静态校验。**不需要 MySQL 就能跑。**

这组测试的价值在于：schema 里那几处「配错了不会报错、只会静默失去保证」
的设置，必须有测试盯着。具体是：

    - ``version_id_col`` 没配 → CAS 静默失效，并发写入互相覆盖
    - ``version_id_generator`` 没设 False → ORM 自己编版本号，
      领域层的 state_version 变成摆设
    - 表没显式 InnoDB → 在默认引擎是 MyISAM 的环境下没有行锁
    - 字符集不是 utf8mb4 → 中文与 emoji 存不下
    - 缺 UNIQUE(run_id, sequence) → 并发写入产生重复序号

这些都不会在开发时报错，只会在上线后偶发丢数据。
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from codeinsight.infrastructure.db.engine import (
    DEFAULT_PORT,
    ENV_DATABASE,
    ENV_HOST,
    ENV_PASSWORD,
    ENV_PORT,
    ENV_USER,
    MySqlConfig,
)
from codeinsight.infrastructure.db.schema import (
    ApprovalRow,
    AuditRecordRow,
    Base,
    GatewayCostRow,
    GoalRow,
    IdempotencyRow,
    MemoryRecordRow,
    RunEventRow,
    RunRow,
    SessionRow,
)


def _mysql_ddl(model: type) -> str:
    """把某张表的建表语句按 MySQL 方言渲染成文本。"""
    return str(CreateTable(model.__table__).compile(dialect=mysql.dialect()))


# ---------------------------------------------------------------------------
# 乐观锁配置
# ---------------------------------------------------------------------------


def test_run_row_has_version_id_col() -> None:
    """没配这一项，CAS 会静默失效——UPDATE 不带版本条件，谁后写谁赢。"""
    mapper = RunRow.__mapper__
    assert mapper.version_id_col is not None
    assert mapper.version_id_col.name == "state_version"


def test_version_id_generator_is_disabled() -> None:
    """必须为 False，否则 ORM 自己 +1，领域层的 state_version 成为摆设。"""
    assert RunRow.__mapper__.version_id_generator is False


def test_no_other_table_claims_optimistic_locking() -> None:
    """只有 runs 需要乐观锁。别处误配会让普通写入变得随机失败。"""
    for mapper in Base.registry.mappers:
        if mapper.class_ is RunRow:
            continue
        assert mapper.version_id_col is None, mapper.class_.__name__


# ---------------------------------------------------------------------------
# 引擎与字符集
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        SessionRow,
        GoalRow,
        RunRow,
        RunEventRow,
        AuditRecordRow,
        ApprovalRow,
        IdempotencyRow,
        MemoryRecordRow,
    ],
)
def test_every_table_is_innodb(model: type) -> None:
    """MyISAM 没有行锁与事务，CAS 在它上面不成立。"""
    assert "ENGINE=InnoDB" in _mysql_ddl(model)


@pytest.mark.parametrize(
    "model",
    [
        SessionRow,
        GoalRow,
        RunRow,
        RunEventRow,
        AuditRecordRow,
        ApprovalRow,
        IdempotencyRow,
        MemoryRecordRow,
    ],
)
def test_every_table_is_utf8mb4(model: type) -> None:
    """MySQL 的 "utf8" 是每字符最多 3 字节的残缺实现，存不了 emoji。"""
    ddl = _mysql_ddl(model)
    assert "CHARSET=utf8mb4" in ddl


def test_no_table_uses_legacy_utf8() -> None:
    for table in Base.metadata.sorted_tables:
        charset = table.kwargs.get("mysql_charset")
        assert charset == "utf8mb4", f"{table.name} 的字符集是 {charset}"


# ---------------------------------------------------------------------------
# 约束与索引
# ---------------------------------------------------------------------------


def test_event_sequence_is_unique_per_run() -> None:
    """应用层检查挡不住并发；唯一约束是最后一道防线。"""
    ddl = _mysql_ddl(RunEventRow)
    assert "UNIQUE" in ddl
    assert "uq_run_events_run_sequence" in ddl


def test_idempotency_primary_key_is_composite() -> None:
    """三种粒度的键空间必须隔离，否则 request 级会误命中 action 级。"""
    key_columns: list[str] = []
    for column in IdempotencyRow.__table__.primary_key.columns:
        key_columns.append(column.name)
    assert sorted(key_columns) == ["key_value", "scope"]


def test_approval_consumed_column_is_nullable() -> None:
    """NULL 表示未消费。这是「带条件 UPDATE」那条语句的判定依据。"""
    assert ApprovalRow.__table__.c.consumed_at_epoch_ms.nullable is True


def test_audit_and_event_are_separate_tables() -> None:
    """分表而不是加 is_audit 列——避免清理旧事件时误删审计记录（增补 R-5）。"""
    assert AuditRecordRow.__tablename__ != RunEventRow.__tablename__
    assert "is_audit" not in RunEventRow.__table__.c


def test_gateway_cost_has_price_version_and_attempt_primary_key() -> None:
    assert GatewayCostRow.__table__.c.attempt_id.primary_key is True
    assert GatewayCostRow.__table__.c.price_version.nullable is False
    assert GatewayCostRow.__table__.c.total_stars.nullable is False


def test_tenant_columns_exist_on_business_tables() -> None:
    """增补 R-3：维度先进数据模型，将来加租户不必改表结构。"""
    for model in [SessionRow, GoalRow, RunRow]:
        assert "tenant_id" in model.__table__.c
        assert "user_id" in model.__table__.c


def test_epoch_columns_are_bigint() -> None:
    """毫秒时间戳超出 INT 范围（2038 问题的变体），必须 BIGINT。"""
    assert "BIGINT" in _mysql_ddl(RunEventRow)
    assert "BIGINT" in _mysql_ddl(ApprovalRow)


def test_all_tables_compile_under_mysql_dialect() -> None:
    """整套 DDL 能在 MySQL 方言下渲染——不需要连库就能发现类型错误。"""
    for table in Base.metadata.sorted_tables:
        rendered = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert "CREATE TABLE" in rendered


# ---------------------------------------------------------------------------
# 连接配置
# ---------------------------------------------------------------------------


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [ENV_HOST, ENV_PORT, ENV_USER, ENV_PASSWORD, ENV_DATABASE]:
        monkeypatch.delenv(name, raising=False)


def test_from_env_returns_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """返回 None 而不是抛异常——否则没装 MySQL 的机器整个套件都跑不起来。"""
    _clear_env(monkeypatch)
    assert MySqlConfig.from_env() is None


def test_from_env_reads_all_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(ENV_HOST, "127.0.0.1")
    monkeypatch.setenv(ENV_PORT, "3307")
    monkeypatch.setenv(ENV_USER, "codeinsight")
    monkeypatch.setenv(ENV_PASSWORD, "secret")
    monkeypatch.setenv(ENV_DATABASE, "codeinsight_v2")
    config = MySqlConfig.from_env()
    assert config is not None
    assert config.port == 3307
    assert config.database == "codeinsight_v2"


def test_port_defaults_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(ENV_HOST, "127.0.0.1")
    monkeypatch.setenv(ENV_USER, "codeinsight")
    monkeypatch.setenv(ENV_DATABASE, "codeinsight_v2")
    config = MySqlConfig.from_env()
    assert config is not None
    assert config.port == DEFAULT_PORT


def test_partial_config_is_treated_as_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """少了 database 就不算配好——半配置连上去会建错库。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv(ENV_HOST, "127.0.0.1")
    monkeypatch.setenv(ENV_USER, "codeinsight")
    assert MySqlConfig.from_env() is None


def test_empty_password_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """空密码是合法配置（本地开发常见），不该被当作未配置。"""
    _clear_env(monkeypatch)
    monkeypatch.setenv(ENV_HOST, "127.0.0.1")
    monkeypatch.setenv(ENV_USER, "root")
    monkeypatch.setenv(ENV_DATABASE, "codeinsight_v2")
    config = MySqlConfig.from_env()
    assert config is not None
    assert config.password == ""


def test_url_declares_utf8mb4() -> None:
    """PyMySQL 的默认字符集不是 utf8mb4，少了这段中文会以 latin1 往返。"""
    config = MySqlConfig(
        host="127.0.0.1", port=3306, user="u", password="p", database="d"
    )
    assert "charset=utf8mb4" in config.url


def test_safe_url_masks_the_password() -> None:
    """写日志只能用这个。密码进日志等于泄露。"""
    config = MySqlConfig(
        host="127.0.0.1", port=3306, user="u", password="hunter2", database="d"
    )
    assert "hunter2" not in config.safe_url
    assert "***" in config.safe_url


@pytest.mark.parametrize(
    ("host", "port", "user", "database"),
    [
        ("", 3306, "u", "d"),
        ("h", 0, "u", "d"),
        ("h", 70000, "u", "d"),
        ("h", 3306, "  ", "d"),
        ("h", 3306, "u", ""),
    ],
)
def test_invalid_config_is_refused(host: str, port: int, user: str, database: str) -> None:
    with pytest.raises(ValueError):
        MySqlConfig(host=host, port=port, user=user, password="", database=database)
