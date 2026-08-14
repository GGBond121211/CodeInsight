"""Input validation for checkout."""

from .models import OrderRequest


def validate_request(request: OrderRequest) -> None:
    if not request.sku:
        raise ValueError("sku is required")
    if request.quantity <= 0:
        raise ValueError("quantity must be positive")
