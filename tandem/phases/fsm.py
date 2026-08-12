"""Phase-order validation. Rejects physically-impossible orderings;
it does not detect anything. A violation means the telemetry (or the
detector) is untrustworthy for this recording."""
from __future__ import annotations

PHASE_ORDER = ["ground_pre", "climb", "freefall"]


def validate_order(segments) -> str | None:
    last_rank = -1
    last_start = float("-inf")
    for seg in segments:
        if seg.type not in PHASE_ORDER:
            continue
        rank = PHASE_ORDER.index(seg.type)
        if rank < last_rank or seg.start_s < last_start:
            return "PHASE_ORDER_VIOLATION"
        last_rank = rank
        last_start = seg.start_s
    return None
