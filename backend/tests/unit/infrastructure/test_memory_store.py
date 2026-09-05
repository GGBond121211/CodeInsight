from __future__ import annotations

from codeinsight.domain.memory import LAYER_SESSION, MemoryRecord
from codeinsight.domain.ports import MemoryStore
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore


def _record(**overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "record_id": "session-record",
        "layer": LAYER_SESSION,
        "owner_id": "session-1",
        "content": "{\"summary\": \"已确认\"}",
        "token_estimate": 8,
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "repo_id": "repo-a",
        "source": "test",
    }
    values.update(overrides)
    return MemoryRecord(**values)  # type: ignore[arg-type]


def test_in_memory_store_implements_memory_port() -> None:
    assert isinstance(InMemoryMemoryStore(), MemoryStore)


def test_records_are_scoped_by_layer_owner_and_identity() -> None:
    store = InMemoryMemoryStore()
    record = _record()
    store.save(record)

    assert (
        store.get(
            LAYER_SESSION,
            "session-1",
            record.record_id,
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
            record.record_id,
            tenant_id="tenant-a",
            user_id="user-b",
            repo_id="repo-a",
        )
        is None
    )


def test_list_and_forget_only_touch_the_requested_scope() -> None:
    store = InMemoryMemoryStore()
    store.save(_record(record_id="a"))
    store.save(_record(record_id="b", owner_id="session-2"))
    assert [item.record_id for item in store.list(
        LAYER_SESSION,
        "session-1",
        tenant_id="tenant-a",
        user_id="user-a",
        repo_id="repo-a",
    )] == ["a"]

    store.delete(
        LAYER_SESSION,
        "session-1",
        "a",
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
    assert store.get(
        LAYER_SESSION,
        "session-2",
        "b",
        tenant_id="tenant-a",
        user_id="user-a",
        repo_id="repo-a",
    ) is not None
