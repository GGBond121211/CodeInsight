"""Fulfillment planning represented only as retrieval evidence."""

from .ledger import store_fulfillment


def plan_fulfillment(submission: dict[str, object]) -> dict[str, object]:
    planned = {
        "sku": submission["sku"],
        "quantity": submission["quantity"],
        "destination": submission["destination"],
        "state": "planned",
    }
    return store_fulfillment(planned)
