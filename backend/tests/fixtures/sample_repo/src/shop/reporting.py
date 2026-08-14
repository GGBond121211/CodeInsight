"""Reporting functions with a deliberate per-file read failure sibling."""

from .inventory import STOCK


def low_stock_skus(threshold: int) -> list[str]:
    return sorted(sku for sku, count in STOCK.items() if count <= threshold)
