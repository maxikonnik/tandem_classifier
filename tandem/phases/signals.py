"""Build an analysis-grade signal set from GPMF telemetry.

Two signals only: accelerometer magnitude (m/s^2) and GPS 3D speed
(m/s). GPS altitude is never used. Accel (~200 Hz) and GPS (~18 Hz)
are resampled onto one uniform grid (default 10 Hz). Per-stream timing
is assumed uniform across the recording — precise GPMF payload timing
is a later refinement; seconds-scale phase boundaries do not need it.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field

from tandem.recon.ffprobe import find_gpmf_stream_index
from tandem.recon.gpmf import decode_numbers, iter_klv, walk


@dataclass
class Signals:
    t_s: list[float] = field(default_factory=list)
    accel_mag: list[float] = field(default_factory=list)
    speed_3d: list[float] = field(default_factory=list)
    fs: float = 10.0
    has_accel: bool = False
    has_gps: bool = False


def _flatten(klv) -> list[float]:
    return [v for sample in decode_numbers(klv) for v in sample]


def resample(values: list[float], n_out: int) -> list[float]:
    if not values or n_out <= 0:
        return []
    if len(values) == 1:
        return [float(values[0])] * n_out
    if n_out == 1:
        return [float(values[0])]
    n_in = len(values)
    out = []
    for j in range(n_out):
        pos = j * (n_in - 1) / (n_out - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n_in - 1)
        frac = pos - lo
        out.append(values[lo] * (1.0 - frac) + values[hi] * frac)
    return out


def _stream_children(blob: bytes):
    for _path, strm in walk(blob):
        if strm.key == "STRM":
            yield list(iter_klv(strm.payload))


def _accel_magnitudes(children) -> list[float] | None:
    scal = None
    accl = None
    for c in children:
        if c.key == "SCAL":
            scal = _flatten(c)
        elif c.key == "ACCL":
            accl = c
    if accl is None:
        return None
    divisor = float(scal[0]) if (scal and scal[0]) else 1.0
    mags = []
    for sample in decode_numbers(accl):
        mags.append(math.sqrt(sum((v / divisor) ** 2 for v in sample)))
    return mags


def _gps_speeds(children) -> list[float] | None:
    scal = None
    gps5 = None
    for c in children:
        if c.key == "SCAL":
            scal = _flatten(c)
        elif c.key == "GPS5":
            gps5 = c
    if gps5 is None:
        return None
    if scal and len(scal) >= 5 and scal[4]:
        divisor = float(scal[4])
    elif scal and len(scal) == 1 and scal[0]:
        divisor = float(scal[0])
    else:
        divisor = 1.0
    return [sample[4] / divisor for sample in decode_numbers(gps5)]


def build_signals(blob: bytes, fs: float = 10.0) -> Signals:
    accel_raw = None
    speed_raw = None
    for children in _stream_children(blob):
        if accel_raw is None:
            accel_raw = _accel_magnitudes(children)
        if speed_raw is None:
            speed_raw = _gps_speeds(children)
    sig = Signals(fs=fs, has_accel=accel_raw is not None, has_gps=speed_raw is not None)
    # Recording duration: longer of the two streams at their nominal rates.
    accel_dur = (len(accel_raw) / 200.0) if accel_raw else 0.0
    gps_dur = (len(speed_raw) / 18.0) if speed_raw else 0.0
    duration = max(accel_dur, gps_dur)
    if duration <= 0:
        return sig
    n_out = max(2, int(round(duration * fs)))
    sig.t_s = [i / fs for i in range(n_out)]
    sig.accel_mag = resample(accel_raw, n_out) if accel_raw else []
    sig.speed_3d = resample(speed_raw, n_out) if speed_raw else []
    return sig


def build_signals_from_file(path: str, fs: float = 10.0) -> Signals | None:
    index = find_gpmf_stream_index(path)
    if index is None:
        return None
    out = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-map", f"0:{index}",
         "-codec", "copy", "-f", "data", "-"],
        check=True, capture_output=True,
    )
    if not out.stdout:
        return None
    return build_signals(out.stdout, fs=fs)
