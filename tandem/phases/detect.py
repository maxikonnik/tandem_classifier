"""Detect physical phases from accelerometer magnitude and 3D speed.

Only phases up to freefall are emitted — the operator's canopy and
landing are on a different flight profile and never represent the
tandem (spec §5.1). All results carry source="telemetry".
"""
from __future__ import annotations

from dataclasses import dataclass

G = 9.80665
EXIT_DROPOUT_G = 0.35        # |a| below this (in g) marks free-fall onset
EXIT_MIN_DURATION_S = 1.0
PRE_EXIT_G = 0.8          # the drop must come FROM a steady ~1g (jerk gate)
JERK_LOOKBACK_S = 1.5     # measure the pre-drop level this far before the dip
OPENING_SHOCK_G = 2.5      # a real canopy opening (~2-6g) towers over freefall buffeting (~2-3g)
FREEFALL_MIN_S = 15.0
FREEFALL_MAX_S = 90.0
FREEFALL_MIN_MS = 45.0
FREEFALL_MAX_MS = 60.0
ONEG_LOW_G = 0.6
ONEG_HIGH_G = 1.4
GROUND_SPEED_MS = 3.0


@dataclass
class Event:
    type: str
    t_s: float
    source: str
    confidence: float


def _pre_drop_level(sig, i):
    """Mean |a| over ~1 s ending JERK_LOOKBACK_S before index i (the in-aircraft level)."""
    back = int(round(JERK_LOOKBACK_S * sig.fs))
    lo = max(0, i - back)
    hi = min(len(sig.accel_mag), lo + max(1, int(round(1.0 * sig.fs))))
    seg = sig.accel_mag[lo:hi]
    return sum(seg) / len(seg) if seg else 0.0


def detect_exit(sig):
    if not sig.has_accel or not sig.accel_min:
        return None
    thresh = EXIT_DROPOUT_G * G
    min_samples = max(1, int(round(EXIT_MIN_DURATION_S * sig.fs)))
    run_start = None
    for i, a in enumerate(sig.accel_min):
        if a < thresh:
            if run_start is None:
                run_start = i
            if i - run_start + 1 >= min_samples:
                # jerk gate: the dip must have dropped rapidly FROM ~1g, not be a
                # gradual low-g stretch. (Duration already filters <1s ground bumps.)
                if _pre_drop_level(sig, run_start) < PRE_EXIT_G * G:
                    run_start = None
                    continue
                depth = 1.0 - (sig.accel_min[run_start] / G)
                conf = max(0.0, min(1.0, depth))
                if sig.has_gps and sig.speed_3d[run_start:] and \
                        max(sig.speed_3d[run_start:]) >= FREEFALL_MIN_MS:
                    conf = min(1.0, conf + 0.1)   # GPS corroboration
                return Event(type="exit", t_s=sig.t_s[run_start], source="telemetry",
                             confidence=round(conf, 3))
        else:
            run_start = None
    return None


@dataclass
class Segment:
    type: str
    start_s: float
    end_s: float
    source: str
    confidence: float


def _is_freefall_sample(a: float, v: float) -> bool:
    return (FREEFALL_MIN_MS <= v <= FREEFALL_MAX_MS
            and ONEG_LOW_G * G <= a <= ONEG_HIGH_G * G)


def detect_freefall(sig, exit_event):
    if exit_event is None or not sig.speed_3d:
        return None
    # Find the exit index, then the longest freefall run at/after it.
    start_idx = min(range(len(sig.t_s)), key=lambda i: abs(sig.t_s[i] - exit_event.t_s))
    best = None
    run_start = None
    for i in range(start_idx, len(sig.t_s)):
        if _is_freefall_sample(sig.accel_mag[i], sig.speed_3d[i]):
            if run_start is None:
                run_start = i
            run_end = i
            if best is None or (run_end - run_start) > (best[1] - best[0]):
                best = (run_start, run_end)
        else:
            run_start = None
    if best is None:
        return None
    lo, hi = best
    return Segment(type="freefall", start_s=sig.t_s[lo], end_s=sig.t_s[hi],
                   source="telemetry", confidence=0.9)


def detect_ground_climb(sig, exit_event):
    if exit_event is None or not sig.speed_3d:
        return []
    exit_idx = min(range(len(sig.t_s)), key=lambda i: abs(sig.t_s[i] - exit_event.t_s))
    if exit_idx <= 0:
        return []
    # First index where speed rises above the ground threshold = takeoff.
    takeoff = None
    for i in range(exit_idx):
        if sig.speed_3d[i] > GROUND_SPEED_MS:
            takeoff = i
            break
    segments = []
    if takeoff is None:
        # Never left the ground before exit (unusual) — whole span is ground.
        segments.append(Segment("ground_pre", sig.t_s[0], sig.t_s[exit_idx],
                                 "telemetry", 0.9))
        return segments
    if takeoff > 0:
        segments.append(Segment("ground_pre", sig.t_s[0], sig.t_s[takeoff],
                                 "telemetry", 0.95))
    segments.append(Segment("climb", sig.t_s[takeoff], sig.t_s[exit_idx],
                             "telemetry", 0.9))
    return segments
