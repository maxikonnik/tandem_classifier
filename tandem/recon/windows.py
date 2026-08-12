"""Rough freefall/canopy windowing from GPS 3D speed.

Reconnaissance-only: aims the frame sampler, never emits events. Uses
the GPS5 3D-speed field directly — GPS altitude is unusably noisy, so
we never differentiate it. The production detector (spec section 5.1)
uses accelerometer signatures and a phase state machine instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from tandem.recon.telemetry import Telemetry


@dataclass
class Windows:
    freefall_start_s: float
    freefall_end_s: float
    canopy_open_s: float | None


def smooth(xs: list[float], k: int = 5) -> list[float]:
    if len(xs) < k:
        return list(xs)
    out = []
    half = k // 2
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        window = xs[lo:hi]
        out.append(sum(window) / len(window))
    return out


def estimate_windows(tel: Telemetry, freefall_ms: float = 40.0) -> Windows | None:
    if not tel.speed_3d_ms or not tel.t_s:
        return None
    speed = smooth(tel.speed_3d_ms)
    freefall_idx = [i for i, v in enumerate(speed) if v >= freefall_ms]
    if not freefall_idx:
        # No clear freefall signature; hand back the whole recording.
        return Windows(freefall_start_s=tel.t_s[0], freefall_end_s=tel.t_s[-1], canopy_open_s=None)
    start_i, end_i = freefall_idx[0], freefall_idx[-1]
    canopy_open_s = None
    for i in range(end_i, len(speed)):
        if speed[i] < freefall_ms:
            canopy_open_s = tel.t_s[i]
            break
    return Windows(
        freefall_start_s=tel.t_s[start_i],
        freefall_end_s=tel.t_s[end_i],
        canopy_open_s=canopy_open_s,
    )
