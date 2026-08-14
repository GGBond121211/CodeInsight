"""Deterministic pricing helpers."""

PRICE_CENTS = {"book": 1500, "pen": 250}


def unit_price(sku: str) -> int:
    """Return the configured unit price."""
    return PRICE_CENTS[sku]


def order_total(sku: str, quantity: int) -> int:
    """Compute a total without tax or discounts."""
    price = unit_price(sku)
    return price * quantity
