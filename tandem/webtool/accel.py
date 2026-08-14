"""Decode GoPro ACCL from a GPMF blob onto the video's time base.

Each top-level DEVC record is one metadata packet; inside it, one STRM
holds the accelerometer stream (ACCL) and its SCAL divisor. Samples are
spread evenly across the packet's real duration so the resulting time
axis matches video.currentTime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from tandem.recon.gpmf import decode_numbers, iter_klv


@dataclass
class AccelSeries:
    t: list[float] = field(default_factory=list)
    ax: list[float] = field(default_factory=list)
    ay: list[float] = field(default_factory=list)
    az: list[float] = field(default_factory=list)
    amag: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _packet_accel(devc_payload: bytes) -> tuple[list[tuple[float, float, float]], bool]:
    """Return (scaled xyz samples, missing_scal) for one DEVC payload."""
    for strm in iter_klv(devc_payload):
        if strm.key != "STRM":
            continue
        scal = None
        accl = None
        for child in iter_klv(strm.payload):
            if child.key == "SCAL":
                scal = decode_numbers(child)[0][0]
            elif child.key == "ACCL":
                accl = child
        if accl is not None:
            divisor = float(scal) if scal else 1.0
            samples = [(x / divisor, y / divisor, z / divisor)
                       for x, y, z in decode_numbers(accl)]
            return samples, scal is None
    return [], False


def parse_accel(
    blob: bytes,
    packet_times: list[tuple[float, float]],
    video_duration: float | None = None,
) -> AccelSeries:
    packets: list[list[tuple[float, float, float]]] = []
    missing_scal = False
    for devc in iter_klv(blob):
        if devc.key != "DEVC":
            continue
        samples, no_scal = _packet_accel(devc.payload)
        missing_scal = missing_scal or no_scal
        packets.append(samples)

    series = AccelSeries()
    aligned = len(packets) == len(packet_times) and len(packet_times) > 0

    if aligned:
        for samples, (pts, dur) in zip(packets, packet_times):
            n = len(samples)
            if n == 0:
                continue
            for j, (x, y, z) in enumerate(samples):
                series.t.append(pts + (j + 0.5) / n * dur)
                series.ax.append(x)
                series.ay.append(y)
                series.az.append(z)
    else:
        series.warnings.append("GPMF packet count mismatch; uniform timing")
        flat = [s for pk in packets for s in pk]
        n = len(flat)
        if video_duration:
            total = video_duration
        elif packet_times:
            total = packet_times[-1][0] + packet_times[-1][1]
        else:
            total = float(n)
        for i, (x, y, z) in enumerate(flat):
            series.t.append((i + 0.5) / n * total if n else 0.0)
            series.ax.append(x)
            series.ay.append(y)
            series.az.append(z)

    series.amag = [math.sqrt(x * x + y * y + z * z)
                   for x, y, z in zip(series.ax, series.ay, series.az)]
    if missing_scal:
        series.warnings.append("missing SCAL; accelerometer in raw units")
    return series
