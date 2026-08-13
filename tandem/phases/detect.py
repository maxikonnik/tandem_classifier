"""Detect physical phases from accelerometer magnitude and 3D speed.

Only phases up to freefall are emitted — the operator's canopy and
landing are on a different flight profile and never represent the
tandem (spec §5.1). All results carry source="telemetry".
"""
from __future__ import annotations

from dataclasses import dataclass

G = 9.80665
EXIT_SMOOTH_S = 1.0          # smoothing window for the |a| envelope (real exit dips are noisy)
EXIT_SMOOTH_G = 0.7          # smoothed |a| (in g) below this = free-fall-onset region
EXIT_MIN_DURATION_S = 1.0
PRE_EXIT_G = 0.8          # the drop must come FROM a steady ~1g (jerk gate)
JERK_LOOKBACK_S = 1.5     # measure the pre-drop level this far before the dip
OPENING_SHOCK_G = 2.5      # a real canopy opening (~2-6g) towers over freefall buffeting (~2-3g)
FREEFALL_MIN_S = 15.0
FREEFALL_MAX_S = 90.0
FREEFALL_MIN_MS = 45.0
GROUND_SPEED_MS = 3.0
STILL_WINDOW_S = 1.0      # window for the local-std stillness check (no-GPS ground split)
GROUND_STD_G = 0.05       # local |a| std below this (in g) counts as "stationary"
GROUND_LEVEL_TOL_G = 0.15 # local |a| mean must be within this of 1g to count as "stationary"


@dataclass
class Event:
    type: str
    t_s: float
    source: str
    confidence: float


def _smooth(xs, k):
    """Moving average over a window of k samples."""
    if len(xs) < k or k < 2:
        return list(xs)
    half = k // 2
    out = []
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def detect_exit(sig):
    # Real free-fall onset is a NOISY dip: |a| plunges toward 0 g but buffeting
    # spikes it back above any raw threshold within fractions of a second, so a
    # "continuous sub-threshold run" never forms (verified on real footage). So
    # threshold the SMOOTHED |a| envelope instead. GPS optional; accel suffices.
    if not sig.has_accel or not sig.accel_mag:
        return None
    fs = sig.fs
    sm = _smooth([a / G for a in sig.accel_mag], max(2, int(round(EXIT_SMOOTH_S * fs))))
    min_samples = max(1, int(round(EXIT_MIN_DURATION_S * fs)))
    back = int(round(JERK_LOOKBACK_S * fs))
    run_start = None
    for i, v in enumerate(sm):
        if v < EXIT_SMOOTH_G:
            if run_start is None:
                run_start = i
            if i - run_start + 1 >= min_samples:
                # jerk gate: the dip must have dropped FROM a steady ~1 g
                if sm[max(0, run_start - back)] < PRE_EXIT_G:
                    run_start = None
                    continue
                conf = max(0.0, min(1.0, 1.0 - sm[run_start]))
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


def detect_freefall(sig, exit_event):
    if exit_event is None or not sig.accel_mag:
        return None
    start_idx = min(range(len(sig.t_s)), key=lambda i: abs(sig.t_s[i] - exit_event.t_s))
    lo = start_idx + int(round(FREEFALL_MIN_S * sig.fs))
    hi = min(len(sig.accel_mag), start_idx + int(round(FREEFALL_MAX_S * sig.fs)))
    end_idx = None
    if lo < hi:
        peak = max(range(lo, hi), key=lambda i: sig.accel_mag[i])
        if sig.accel_mag[peak] >= OPENING_SHOCK_G * G:   # the opening towers over buffeting
            end_idx = peak
    if end_idx is None:                                   # no clear opening -> cap at max plausible freefall
        end_idx = min(hi, len(sig.t_s) - 1)
    if end_idx <= start_idx:
        return None
    return Segment(type="freefall", start_s=sig.t_s[start_idx], end_s=sig.t_s[end_idx],
                   source="telemetry", confidence=0.85)


def _local_mean_std(values, lo, hi):
    seg = values[lo:hi]
    n = len(seg)
    if n == 0:
        return None, None
    mean = sum(seg) / n
    var = sum((x - mean) ** 2 for x in seg) / n
    return mean, var ** 0.5


def _stationary_prefix_end(sig, exit_idx):
    """Longest leading run of windows that are both low-std AND near 1g, in samples.

    Walks the pre-exit span window by window from t=0 and stops at the
    first window that isn't a steady ~1g reading — low local std alone
    isn't enough, since a stuck/saturated sensor or a miscalibrated
    offset can be just as flat while sitting nowhere near 1g. Returns
    None if even the first window doesn't qualify (no confident prefix).
    """
    window = max(1, int(round(STILL_WINDOW_S * sig.fs)))
    std_threshold = GROUND_STD_G * G
    level_tol = GROUND_LEVEL_TOL_G * G
    end = 0
    i = 0
    while i < exit_idx:
        hi = min(i + window, exit_idx)
        mean, std = _local_mean_std(sig.accel_mag, i, hi)
        if std is None or std > std_threshold or abs(mean - G) > level_tol:
            break
        end = hi
        i = hi
    return end if end > 0 else None


def detect_ground_climb(sig, exit_event):
    if exit_event is None:
        return []
    exit_idx = min(range(len(sig.t_s)), key=lambda i: abs(sig.t_s[i] - exit_event.t_s))
    if exit_idx <= 0:
        return []

    if sig.has_gps:
        if not sig.speed_3d:
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

    # No GPS: GPS speed is unavailable, so split by accelerometer stillness
    # instead. A confidently-detected leading low-variance prefix (parked,
    # engine off) becomes ground_pre; everything after it up to exit is
    # climb (aircraft). Without a confident prefix, emit just climb.
    if not sig.accel_mag:
        return []
    ground_end = _stationary_prefix_end(sig, exit_idx)
    segments = []
    climb_start = 0
    if ground_end:
        segments.append(Segment("ground_pre", sig.t_s[0], sig.t_s[ground_end],
                                 "telemetry", 0.85))
        climb_start = ground_end
    if climb_start < exit_idx:
        segments.append(Segment("climb", sig.t_s[climb_start], sig.t_s[exit_idx],
                                 "telemetry", 0.8))
    return segments
