"""MySQL 集成测试的公共装置。

**安全设计：测试只连一个专用的测试库，绝不连应用库。**

这些测试会 ``TRUNCATE`` 全部表。如果它们复用 ``CODEINSIGHT_MYSQL_DATABASE``
（应用运行时用的库），某天有人在配好开发环境的机器上跑一次全量测试，
开发数据就没了。所以测试读的是另一个变量::

    CODEINSIGHT_MYSQL_TEST_DATABASE=codeinsight_v2_test

未设置就整体跳过。这不是「多一个变量的麻烦」，而是让「清空数据」这个
动作在配置层面就无法指向真实数据——比写一行注释提醒可靠得多。

额外再加一道：库名必须含 ``test``。两个人都会犯的错是把测试变量指向
应用库，库名检查能把这种情况挡下来。
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from codeinsight.infrastructure.db.engine import (
    DEFAULT_PORT,
    ENV_HOST,
    ENV_PASSWORD,
    ENV_PORT,
    ENV_USER,
    MySqlConfig,
    create_all_tables,
    create_db_engine,
)
from codeinsight.infrastructure.db.stores import truncate_all_tables

ENV_TEST_DATABASE = "CODEINSIGHT_MYSQL_TEST_DATABASE"


def _test_config() -> MySqlConfig | None:
    """读取测试库配置。任一必需项缺失即返回 None（测试跳过）。"""
    host = os.environ.get(ENV_HOST, "").strip()
    user = os.environ.get(ENV_USER, "").strip()
    database = os.environ.get(ENV_TEST_DATABASE, "").strip()
    if not host or not user or not database:
        return None
    if "test" not in database.lower():
        pytest.fail(
            f"{ENV_TEST_DATABASE} 指向了 {database!r}，库名里没有 'test'。"
            "集成测试会清空全部表——拒绝对一个看起来不是测试库的目标执行。"
        )
    raw_port = os.environ.get(ENV_PORT, "").strip()
    if raw_port:
        port = int(raw_port)
    else:
        port = DEFAULT_PORT
    return MySqlConfig(
        host=host,
        port=port,
        user=user,
        password=os.environ.get(ENV_PASSWORD, ""),
        database=database,
    )


@pytest.fixture(scope="session")
def mysql_engine() -> Engine:
    """整个测试会话共享一个引擎；建表一次。

    连不上时 skip 而不是 fail：MySQL 没启动是环境问题，不是代码缺陷。
    但**库名不合规是配置错误，那种情况必须 fail**（见 _test_config）。
    """
    config = _test_config()
    if config is None:
        pytest.skip(
            f"未配置 {ENV_TEST_DATABASE}（以及 {ENV_HOST} / {ENV_USER}），"
            "跳过 MySQL 集成测试。"
        )
    engine = create_db_engine(config)
    try:
        create_all_tables(engine)
    except OperationalError as error:
        engine.dispose()
        pytest.skip(f"无法连接 {config.safe_url}：{error.orig}")
    yield engine
    engine.dispose()


@pytest.fixture()
def session_factory(mysql_engine: Engine) -> sessionmaker[Session]:
    """每个用例前清空全部表，保证用例之间互不影响。"""
    truncate_all_tables(mysql_engine)
    return sessionmaker(bind=mysql_engine, autoflush=False, expire_on_commit=False)
