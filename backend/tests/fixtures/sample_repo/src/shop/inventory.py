"""In-memory-looking inventory code used only as text evidence."""

STOCK = {"book": 8, "pen": 20}


def available(sku: str, quantity: int) -> bool:
    return STOCK.get(sku, 0) >= quantity


def reserve(sku: str, quantity: int) -> None:
    if not available(sku, quantity):
        raise ValueError("insufficient stock")
    STOCK[sku] -= quantity
