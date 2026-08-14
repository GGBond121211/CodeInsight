"""Customer-visible record lookup used only as repository evidence."""

CUSTOMER_RECORD_ROWS = {"C-200": {"status": "packed", "owner": "customer-7"}}


def find_record(record_id: str, customer_id: str) -> dict[str, str]:
    """Return a record only when it belongs to the requesting customer."""
    row = CUSTOMER_RECORD_ROWS[record_id]
    if row["owner"] != customer_id:
        raise LookupError("record is not visible to this customer")
    return {"record_id": record_id, "scope": "customer", "status": row["status"]}
