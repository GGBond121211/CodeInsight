"""Local refund preparation without any external payment integration."""


def prepare_refund(order_id: str, amount_cents: int) -> dict[str, object]:
    """Create a local queued record; no provider or endpoint is selected here."""
    return {"order_id": order_id, "amount_cents": amount_cents, "state": "queued"}


def refund_summary(record: dict[str, object]) -> str:
    return f"refund {record['order_id']} is {record['state']}"
