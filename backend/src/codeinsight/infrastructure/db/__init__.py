"""MySQL 持久化层。

刻意与 ``infrastructure/run_store.py``（内存实现）并列而不是替换它：
两者实现同一组 ``domain/ports.py`` 契约，跑同一套语义测试。单元测试用内存
实现（快、无外部依赖），集成测试用 MySQL 实现（验证 CAS 与唯一约束真的
落在数据库上）。
"""
