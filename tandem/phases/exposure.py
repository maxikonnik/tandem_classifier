"""Exit detection from camera EXPOSURE — an independent corroborator.

Leaving the dark aircraft cabin for daylight forces a sharp exposure change that
the GoPro logs in GPMF: ISO drops from ~500-1200 to ~100, and shutter speeds up
from ~1/220 to ~1/2000. This is a second, physically-independent signal for the
exit event (the accelerometer sees the |a| dip; the exposure sees the light).

Agreement between the two raises confidence in the exit; a large disagreement is a
useful abstention / annotation-priority flag — the two detectors rest on different
physics, so they should not fail the same way.

Note: exposure settles a few seconds AFTER the accelerometer dip (the operator
climbs out, the meter adapts), so the exposure exit typically lags the accel exit
by ~3-6 s. That offset is expected, not a disagreement.
"""
from __future__ import annotations

from tandem.phases.detect import Event

ISO_DAYLIGHT = 130.0          # ISO at/below this = daylight (cabin runs 500-1200)
SHUTTER_DAYLIGHT = 1.0 / 1500  # exposure time (s) at/below this = fast daylight shutter
DAYLIGHT_SUSTAIN_S = 4.0      # daylight must hold this long (ignore transient glare)


def detect_exit_exposure(sig) -> Event | None:
    """First onset of sustained daylight exposure — the exit into open air.

    Returns an Event(type="exit_exposure") at the start of the first run where ISO
    and shutter both read daylight for DAYLIGHT_SUSTAIN_S, or None."""
    if not getattr(sig, "has_exposure", False) or not sig.iso or not sig.shutter:
        return None
    need = max(1, int(round(DAYLIGHT_SUSTAIN_S * sig.fs)))
    run = 0
    for i in range(len(sig.t_s)):
        daylight = (sig.iso[i] <= ISO_DAYLIGHT
                    and 0.0 < sig.shutter[i] <= SHUTTER_DAYLIGHT)
        if daylight:
            run += 1
            if run >= need:
                start_i = i - need + 1
                return Event(type="exit_exposure", t_s=sig.t_s[start_i],
                             source="telemetry", confidence=0.7)
        else:
            run = 0
    return None
