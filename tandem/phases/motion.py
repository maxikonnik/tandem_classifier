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
BREAKOFF_AXIS_EXCURSION_G = 0.8  # per-axis deviation from freefall baseline that flags break-off
BREAKOFF_BASELINE_S = 5.0        # window of established freefall used to compute each axis's baseline
BREAKOFF_SEARCH_LATE_S = 20.0    # the turn-away precedes the operator's deploy (=freefall end); search the last N s
BREAKOFF_SUSTAIN_S = 1.0         # the excursion must be sustained (not a transient buffeting spike)


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


def _axis_baseline(sig, values, freefall):
    """Mean of `values` over a stable window in ESTABLISHED freefall (after the
    onset settling, before any late turn-away) — the tracking orientation."""
    lo = freefall.start_s + ORBIT_MIN_AFTER_EXIT_S
    hi = min(lo + BREAKOFF_BASELINE_S, freefall.end_s)
    idxs = _indices_in_window(sig, lo, hi)
    if not idxs:
        return None
    return sum(values[i] for i in idxs) / len(idxs)


def detect_breakoff(sig, freefall) -> Event | None:
    """The operator turning away from the pair — a SUSTAINED axis excursion in
    the LATE part of the freefall window, BEFORE the operator's own opening
    shock (which is where detect_freefall ends the window). Marks the end of
    useful pair-tracking footage.

    Mounting-agnostic: which axis catches the turn-away varies by camera
    orientation, so this uses whichever axis deviates most from its
    established-freefall baseline. Restricting to the late window keeps a
    mid-freefall orbit (which also swings an axis but returns) from firing.
    """
    if freefall is None:
        return None
    axes = [sig.ax, sig.ay, sig.az]
    if not any(axes):
        return None
    baselines = [_axis_baseline(sig, axis, freefall) for axis in axes]
    if any(b is None for b in baselines):
        return None

    lo = max(freefall.start_s + ORBIT_MIN_AFTER_EXIT_S + BREAKOFF_BASELINE_S,
             freefall.end_s - BREAKOFF_SEARCH_LATE_S)
    idxs = _indices_in_window(sig, lo, freefall.end_s)
    if not idxs:
        return None
    need = max(1, int(round(BREAKOFF_SUSTAIN_S * sig.fs)))
    run_start = None
    for i in idxs:
        max_dev = max(abs(axis[i] - b) for axis, b in zip(axes, baselines))
        if max_dev > BREAKOFF_AXIS_EXCURSION_G:
            if run_start is None:
                run_start = i
            if i - run_start + 1 >= need:
                dev0 = max(abs(axis[run_start] - b) for axis, b in zip(axes, baselines))
                conf = round(max(0.5, min(0.99, 0.5 + (dev0 - BREAKOFF_AXIS_EXCURSION_G) * 0.3)), 3)
                return Event(type="operator_breakoff", t_s=sig.t_s[run_start],
                             source="telemetry", confidence=conf)
        else:
            run_start = None
    return None
