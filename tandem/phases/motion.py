"""Corroborate freefall via accel_std and detect orbit (облёт) dips within it.

accel_std (rolling std of |a|, ~1s window) separates phases on real jump
footage: in-aircraft ~0.02, freefall 0.3-0.6, and an ОБЛЁТ (operator
fly-around) shows as a LOW-std dip (~0.13) INSIDE the freefall window —
the operator is flying smoothly around the tandem pair rather than
buffeting through relative wind. All results carry source="telemetry".
"""
from __future__ import annotations

from tandem.phases.detect import Event, Segment

FREEFALL_STD_MIN = 0.2   # freefall accel_std should exceed this (corroboration)
ORBIT_STD_MAX = 0.18     # an orbit dip's accel_std stays below this
ORBIT_MIN_DURATION_S = 3.0
ORBIT_MIN_AFTER_EXIT_S = 15.0  # skip the freefall-onset settling (|a| rising smoothly = low std,
                               # but not an orbit); a real облёт is a low-std dip in established freefall
# Operator break-off (отворот): a sustained body rotation (turn-away), seen far
# more cleanly on the GYROSCOPE than on linear accel. It is a net rotation in one
# direction, unlike free-fall buffeting (oscillatory, ~zero net). It precedes the
# operator's own canopy deploy (= where detect_freefall ends the window), so we
# search the late window but stop short of the deploy's own rotation spike.
BREAKOFF_NET_WINDOW_S = 1.5      # window for the signed (net) rotation = sustained turn
BREAKOFF_MIN_NET = 0.8          # rad/s; a real turn-away sustains at least this net rate
BREAKOFF_SEARCH_LATE_S = 18.0   # search starts this far before the deploy
BREAKOFF_END_MARGIN_S = 3.5     # ...and stops this short of it, to skip the deploy rotation


def _indices_in_window(sig, start_s, end_s):
    return [i for i, t in enumerate(sig.t_s) if start_s <= t <= end_s]


def freefall_std_ok(sig, freefall) -> bool:
    """True if mean accel_std over the freefall window exceeds FREEFALL_STD_MIN."""
    if freefall is None or not sig.accel_std:
        return False
    idxs = _indices_in_window(sig, freefall.start_s, freefall.end_s)
    if not idxs:
        return False
    values = [sig.accel_std[i] for i in idxs]
    return (sum(values) / len(values)) > FREEFALL_STD_MIN


def detect_orbit(sig, freefall) -> list[Segment]:
    """Find contiguous low-std runs within the freefall window (orbit candidates)."""
    if freefall is None or not sig.accel_std:
        return []
    # search only established freefall, past the onset settling period
    idxs = _indices_in_window(sig, freefall.start_s + ORBIT_MIN_AFTER_EXIT_S, freefall.end_s)
    if not idxs:
        return []

    segments: list[Segment] = []
    run_start = None
    for i in idxs:
        if sig.accel_std[i] < ORBIT_STD_MAX:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                _emit_if_long_enough(sig, run_start, i - 1, segments)
            run_start = None
    if run_start is not None:
        _emit_if_long_enough(sig, run_start, idxs[-1], segments)
    return segments


def _emit_if_long_enough(sig, start_idx, end_idx, segments):
    start_s = sig.t_s[start_idx]
    end_s = sig.t_s[end_idx]
    if end_s - start_s >= ORBIT_MIN_DURATION_S:
        values = sig.accel_std[start_idx:end_idx + 1]
        mean_std = sum(values) / len(values) if values else ORBIT_STD_MAX
        # Lower std relative to the ceiling -> higher confidence it's a real orbit dip.
        confidence = round(max(0.5, min(0.95, 1.0 - mean_std / ORBIT_STD_MAX * 0.5)), 3)
        segments.append(Segment(type="orbit", start_s=start_s, end_s=end_s,
                                 source="telemetry", confidence=confidence))


def _rolling_signed_mean(values, half):
    """Signed rolling mean over a centered window — a one-direction (net) rotation
    survives; oscillatory buffeting averages toward zero."""
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = values[lo:hi]
        out[i] = sum(window) / len(window)
    return out


def detect_breakoff(sig, freefall) -> Event | None:
    """Operator break-off (отворот): the sustained turn-away that ends useful
    pair-tracking, detected from the GYROSCOPE.

    The turn-away is a net rotation in one direction; free-fall buffeting is
    oscillatory (~zero net). So we take each gyro axis's signed rolling mean and
    find where its magnitude peaks in the LATE window — after established freefall,
    but stopping short of the operator's own deploy (freefall end), whose rotation
    spike would otherwise dominate. Mounting-agnostic: whichever axis turns most.
    """
    if freefall is None or not sig.has_gyro or not sig.gx:
        return None
    half = max(1, int(round(BREAKOFF_NET_WINDOW_S * sig.fs / 2)))
    nets = [_rolling_signed_mean(axis, half) for axis in (sig.gx, sig.gy, sig.gz)]

    lo = freefall.end_s - BREAKOFF_SEARCH_LATE_S
    hi = freefall.end_s - BREAKOFF_END_MARGIN_S
    idxs = _indices_in_window(sig, lo, hi)
    if not idxs:
        return None
    best_i, best = None, 0.0
    for i in idxs:
        turn = max(abs(nets[0][i]), abs(nets[1][i]), abs(nets[2][i]))
        if turn > best:
            best, best_i = turn, i
    if best_i is None or best < BREAKOFF_MIN_NET:
        return None
    conf = round(max(0.5, min(0.95, 0.5 + (best - BREAKOFF_MIN_NET) * 0.25)), 3)
    return Event(type="operator_breakoff", t_s=sig.t_s[best_i],
                 source="telemetry", confidence=conf)
