from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from codeinsight.domain.memory import LAYER_SEMANTIC, LAYER_SESSION, MemoryRecord
from codeinsight.domain.ports import MemoryStore
from codeinsight.infrastructure.db.stores import MySqlMemoryStore


def _record(**overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "record_id": "session-record",
        "layer": LAYER_SESSION,
        "owner_id": "session-1",
        "content": "中文摘要 🔧",
        "token_estimate": 5,
        "is_trusted": False,
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "repo_id": "repo-a",
        "source": "explicit-confirmation",
        "confidence": 0.9,
        "expires_at_epoch_ms": 9_999_999_999_999,
        "consent": True,
    }
    values.update(overrides)
    return MemoryRecord(**values)  # type: ignore[arg-type]


def test_mysql_memory_store_roundtrip_and_scope_filter(
    session_factory: sessionmaker[Session],
) -> None:
    store = MySqlMemoryStore(session_factory)
    assert isinstance(store, MemoryStore)
    record = _record()
    store.save(record)

    assert (
        store.get(
            LAYER_SESSION,
            "session-1",
            "session-record",
            tenant_id="tenant-a",
            user_id="user-a",
            repo_id="repo-a",
        )
        == record
    )
    assert (
        store.get(
            LAYER_SESSION,
            "session-1",
            "session-record",
            tenant_id="tenant-a",
            user_id="user-b",
            repo_id="repo-a",
        )
        is None
    )


def test_mysql_memory_store_upserts_and_forgets(
    session_factory: sessionmaker[Session],
) -> None:
    store = MySqlMemoryStore(session_factory)
    store.save(_record(record_id="semantic-record", layer=LAYER_SEMANTIC))
    store.save(_record(record_id="semantic-record", content="updated"))
    loaded = store.get(
        LAYER_SESSION,
        "session-1",
        "semantic-record",
        tenant_id="tenant-a",
        user_id="user-a",
        repo_id="repo-a",
    )
    assert loaded is not None
    assert loaded.content == "updated"

    store.delete(
        LAYER_SESSION,
        "session-1",
        "semantic-record",
        tenant_id="tenant-a",
        user_id="user-a",
        repo_id="repo-a",
    )
    assert store.list(
        LAYER_SESSION,
        "session-1",
        tenant_id="tenant-a",
        user_id="user-a",
        repo_id="repo-a",
    ) == ()
