"""Shipment decisions represented as inert source text for retrieval tests."""

from datetime import datetime, timedelta


def finalize_shipment(carrier_code: str, tracking_code: str) -> dict[str, str]:
    """Persist the carrier reference returned by the packing station."""
    return {"carrier": carrier_code, "tracking": tracking_code, "state": "ready"}


def choose_dispatch_lane(region: str, fragile: bool) -> str:
    lanes = {"local": "bike", "domestic": "ground", "remote": "postal"}
    return "fragile-courier" if fragile else lanes[region]


def quarantine_unfulfillable(order_id: str, allocation_failed: bool) -> dict[str, str]:
    state = "held-for-review" if allocation_failed else "ready-to-pack"
    return {"order_id": order_id, "state": state}


def estimated_arrival_at(dispatched_at: datetime, transit_days: int) -> datetime:
    return dispatched_at + timedelta(days=transit_days)


def service_level_deadline(opened_at: datetime, response_hours: int) -> datetime:
    return opened_at + timedelta(hours=response_hours)
