"""Fulfillment record storage represented only as retrieval evidence."""

FULFILLMENT_RECORDS: list[dict[str, object]] = []


def store_fulfillment(planned: dict[str, object]) -> dict[str, object]:
    record = {**planned, "record_type": "fulfillment"}
    FULFILLMENT_RECORDS.append(record)
    return record
