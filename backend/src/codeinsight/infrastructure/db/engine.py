"""数据库连接配置与引擎构造。

配置从环境变量读取，**未配置时返回 None 而不是抛异常**。理由是集成测试
要能在没有 MySQL 的机器上自动跳过；如果这里抛异常，测试收集阶段就会失败，
整个套件都跑不起来。

连接串刻意不落任何默认密码。``.env.example`` 里只写变量名不写值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

ENV_HOST = "CODEINSIGHT_MYSQL_HOST"
ENV_PORT = "CODEINSIGHT_MYSQL_PORT"
ENV_USER = "CODEINSIGHT_MYSQL_USER"
ENV_PASSWORD = "CODEINSIGHT_MYSQL_PASSWORD"
ENV_DATABASE = "CODEINSIGHT_MYSQL_DATABASE"

DEFAULT_PORT = 3306


@dataclass(frozen=True)
class MySqlConfig:
    """MySQL 连接配置。"""

    host: str
    port: int
    user: str
    password: str
    database: str

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("MySQL host 不能为空")
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"端口不合法：{self.port}")
        if not self.user.strip():
            raise ValueError("MySQL user 不能为空")
        if not self.database.strip():
            raise ValueError("MySQL database 不能为空")

    @property
    def url(self) -> str:
        """SQLAlchemy 连接串。

        ``charset=utf8mb4`` 必须显式写上——PyMySQL 的默认字符集不是它，
        少了这一段中文会以 latin1 往返，读回来是乱码。
        """
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )

    @property
    def safe_url(self) -> str:
        """可安全写进日志的连接串——密码替换为固定占位符。"""
        return (
            f"mysql+pymysql://{self.user}:***"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )

    @classmethod
    def from_env(cls) -> MySqlConfig | None:
        """从环境变量读取配置。任一必需项缺失即返回 None。

        返回 None 表示「本机没有配 MySQL」，调用方应跳过而不是失败。
        """
        host = os.environ.get(ENV_HOST, "").strip()
        user = os.environ.get(ENV_USER, "").strip()
        database = os.environ.get(ENV_DATABASE, "").strip()
        if not host or not user or not database:
            return None
        raw_port = os.environ.get(ENV_PORT, "").strip()
        if raw_port:
            try:
                port = int(raw_port)
            except ValueError:
                return None
        else:
            port = DEFAULT_PORT
        return cls(
            host=host,
            port=port,
            user=user,
            password=os.environ.get(ENV_PASSWORD, ""),
            database=database,
        )


def create_db_engine(config: MySqlConfig, *, echo: bool = False) -> Engine:
    """构造引擎。

    ``pool_pre_ping=True``：MySQL 默认 8 小时后断开空闲连接，连接池里的
    死连接会在下一次使用时报 "server has gone away"。pre_ping 在取出连接前
    先探一次，代价是每次取连接多一个轻量往返，换来的是不会因为闲置而报错。

    ``pool_recycle=3600``：主动在 1 小时后回收连接，比 MySQL 的 8 小时
    超时更早，双保险。
    """
    return create_engine(
        config.url,
        echo=echo,
        pool_pre_ping=True,
        pool_recycle=3600,
        # future 风格下 autoflush 仍默认开启，这里显式关掉：
        # 自动 flush 会让「查询」意外触发写入，CAS 的时序就不可控了。
        future=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """构造 Session 工厂。

    ``expire_on_commit=False``：默认情况下 commit 之后所有对象属性会被
    标记为过期，下次访问会重新查库。我们的 Store 在 commit 后要把 ORM 行
    转换成领域对象，属性过期会引发额外查询甚至 DetachedInstanceError。
    """
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def create_all_tables(engine: Engine) -> None:
    """建表。

    刻意用 ``create_all`` 而不引入 Alembic：Step 2 只有一次建表，没有
    迁移历史需要管理。等到真的需要改已上线的表结构时再引入迁移工具——
    那时才有它要解决的问题。
    """
    from codeinsight.infrastructure.db.schema import Base

    Base.metadata.create_all(engine)
    _apply_additive_compatibility_migrations(engine)


def _apply_additive_compatibility_migrations(engine: Engine) -> None:
    """补充 create_all 无法处理的安全加列；不删除、不改写既有数据。

    只加列、且新列都允许为空或带默认值，因此对既有数据是原地兼容的，
    已经存在的库不必重建，也不会丢数据。
    """

    additions: tuple[tuple[str, str, str], ...] = (
        (
            "gateway_cost_records",
            "ttft_milliseconds",
            "ALTER TABLE gateway_cost_records "
            "ADD COLUMN ttft_milliseconds FLOAT NULL AFTER latency_milliseconds",
        ),
        (
            "sessions",
            "repo_root",
            "ALTER TABLE sessions ADD COLUMN repo_root VARCHAR(1024) NULL",
        ),
        (
            "agent_runs",
            "validation_profile",
            "ALTER TABLE agent_runs ADD COLUMN validation_profile "
"VARCHAR(64) NOT NULL DEFAULT 'python_compile'",
        ),
        (
            "agent_runs",
            "result_limit",
            "ALTER TABLE agent_runs ADD COLUMN result_limit INT NOT NULL DEFAULT 5",
        ),
        (
            "agent_runs",
            "show_debug_reasoning",
            "ALTER TABLE agent_runs ADD COLUMN show_debug_reasoning "
            "TINYINT(1) NOT NULL DEFAULT 0",
        ),
        (
            "agent_runs",
            "patch_id",
            "ALTER TABLE agent_runs ADD COLUMN patch_id VARCHAR(64) NULL",
        ),
        (
            "agent_runs",
            "approval_token",
            "ALTER TABLE agent_runs ADD COLUMN approval_token VARCHAR(255) NULL",
        ),
        (
            "agent_run_outputs",
            "worker_id",
            "ALTER TABLE agent_run_outputs ADD COLUMN worker_id VARCHAR(128) NULL",
        ),
    )
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table, column, statement in additions:
        if table not in existing_tables:
            continue
        columns = {item["name"] for item in inspector.get_columns(table)}
        if column in columns:
            continue
        with engine.begin() as connection:
            connection.execute(text(statement))


def drop_all_tables(engine: Engine) -> None:
    """删表。**仅供测试使用。**

    放在这里而不是测试文件里，是为了让它出现在代码搜索结果中——
    任何人搜 "drop" 都能立刻看到它存在，以及它只该被测试调用。
    """
    from codeinsight.infrastructure.db.schema import Base

    Base.metadata.drop_all(engine)
