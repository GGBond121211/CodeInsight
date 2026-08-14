"""Checkout orchestration."""

from .config import DEFAULT_CURRENCY
from .inventory import reserve
from .models import OrderRequest, Receipt
from .pricing import order_total
from .validation import validate_request


def checkout(request: OrderRequest) -> Receipt:
    validate_request(request)
    reserve(request.sku, request.quantity)
    total = order_total(request.sku, request.quantity)
    return Receipt(
        sku=request.sku,
        quantity=request.quantity,
        total_cents=total,
        currency=DEFAULT_CURRENCY,
    )
