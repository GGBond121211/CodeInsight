"""Administrative record lookup used only as repository evidence."""

ADMIN_RECORD_ROWS = {"A-100": {"status": "held"}}


def find_record(record_id: str) -> dict[str, str]:
    """Return a record for back-office operators."""
    row = ADMIN_RECORD_ROWS[record_id]
    return {"record_id": record_id, "scope": "admin", "status": row["status"]}
