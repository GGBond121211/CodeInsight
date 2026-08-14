"""Small data records used by the fixture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderRequest:
    sku: str
    quantity: int


@dataclass(frozen=True)
class Receipt:
    sku: str
    quantity: int
    total_cents: int
    currency: str
