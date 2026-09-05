"""Parse Insta360 camera telemetry — placeholder for a format not yet decoded.

Insta360 cameras (ONE X / X3 / X4, GO series, Ace) embed an IMU (accelerometer +
gyroscope) used for stabilisation. As with DJI, the layout is proprietary rather
than GoPro GPMF, so it needs reverse-engineering before jumps filmed on Insta360
can be segmented from telemetry. This module is wired into the signal-source
registry (see ``tandem/phases/signals.py``) so that the moment ``read_telemetry``
returns real data, Insta360 footage flows through the same detector as GoPro and
DJI — no other change required.

To implement, mirror ``tandem/recon/dji.py``:

1. Find where the telemetry lives. Inspect a real file with
   ``ffprobe -show_streams`` / ``-show_packets``: Insta360 has variously used a
   dedicated data track, a trailing custom section, or per-frame metadata. Note the
   camera model — layouts differ across ONE X / X3 / X4 / GO.
2. Decode the accelerometer (in g) and gyroscope (native angular rate) per frame.
   Validate against the known jump signature: accel ~1 g at rest collapsing toward 0
   in free-fall, gyro quiet on the ground and spiking in free-fall and at the opening.
3. Return the same contract the detector expects.

``read_telemetry(path)`` must return ``(accels_g, gyros, duration_s)`` — ``accels_g``
and ``gyros`` per-frame lists of ``(x, y, z)`` (accel in g), ``duration_s`` a float —
or ``None`` when the file carries no Insta360 telemetry. ``_signals_from_imu`` in
signals.py turns that tuple into a Signals; nothing else needs to change.
"""
from __future__ import annotations


def read_telemetry(path: str):
    """Return ``(accels_g, gyros, duration_s)`` or None. Not yet implemented — see the
    module docstring for the format-research steps and the required contract."""
    return None
