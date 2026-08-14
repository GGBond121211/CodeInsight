"""Transport-shaped adapter for the fixture."""

from .models import OrderRequest
from .service import checkout


def create_order(payload: dict[str, object]) -> dict[str, object]:
    request = OrderRequest(
        sku=str(payload["sku"]),
        quantity=int(payload["quantity"]),
    )
    receipt = checkout(request)
    return {
        "sku": receipt.sku,
        "quantity": receipt.quantity,
        "total_cents": receipt.total_cents,
        "currency": receipt.currency,
    }
