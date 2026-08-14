"""Fulfillment input adapter represented only as retrieval evidence."""

from .pipeline import plan_fulfillment


def accept_fulfillment(payload: dict[str, object]) -> dict[str, object]:
    submission = {
        "sku": str(payload["sku"]),
        "quantity": int(payload["quantity"]),
        "destination": str(payload["destination"]),
    }
    return plan_fulfillment(submission)
