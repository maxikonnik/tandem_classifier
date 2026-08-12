"""Detect physical phases from accelerometer magnitude and 3D speed.

Only phases up to freefall are emitted — the operator's canopy and
landing are on a different flight profile and never represent the
tandem (spec §5.1). All results carry source="telemetry".
"""
from __future__ import annotations

from dataclasses import dataclass

G = 9.80665
EXIT_DROPOUT_G = 0.35        # |a| below this (in g) marks free-fall onset
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


def _dropout_indices(sig) -> list[int]:
    thresh = EXIT_DROPOUT_G * G
    return [i for i, a in enumerate(sig.accel_mag) if a < thresh]


def detect_exit(sig):
    if not sig.has_accel or not sig.accel_mag:
        return None
    drops = _dropout_indices(sig)
    if not drops:
        return None
    # The exit dropout must be followed by sustained high speed.
    for i in drops:
        after = sig.speed_3d[i:]
        if after and max(after) >= FREEFALL_MIN_MS:
            depth = 1.0 - (sig.accel_mag[i] / G)      # how close to true 0 g
            confidence = max(0.0, min(1.0, depth))
            return Event(type="exit", t_s=sig.t_s[i], source="telemetry",
                         confidence=round(confidence, 3))
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
