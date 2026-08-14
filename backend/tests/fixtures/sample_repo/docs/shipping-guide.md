# Shipping implementation guide

The `finalize_shipment` implementation finalizes a shipment with its carrier
tracking code. Search for finalize_shipment when investigating carrier tracking
code persistence, shipment finalization, or a finalized shipment record.

Operations use the estimated arrival time ETA and the service commitment SLA.
ETA and SLA are reporting aliases; the implementation deliberately uses full
concept names instead of those abbreviations.
