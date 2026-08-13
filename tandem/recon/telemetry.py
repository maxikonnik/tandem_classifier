"""Extract and assemble GPMF telemetry into a UTC-scaled object.

Descent rate comes from the GPS5 3D-speed field (already m/s after the
SCAL divisor), never from GPS altitude — GPS altitude is too noisy to
be worth differentiating. Even so this stays reconnaissance-grade: it
only aims frame sampling. Production phase detection uses the
accelerometer signatures of spec section 5.1.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime

from tandem.recon.ffprobe import find_gpmf_stream_index
from tandem.recon.gpmf import decode_numbers, decode_utc, iter_klv, walk


FIRMWARE_DEFAULT_YEAR = 2021


@dataclass
class Telemetry:
    device_id: int | None = None
    first_utc: datetime | None = None
    has_gps: bool = False
    has_utc: bool = False
    utc_reliable: bool = False
    speed_3d_ms: list[float] = field(default_factory=list)
    t_s: list[float] = field(default_factory=list)


def _flatten(klv) -> list[float]:
    return [v for sample in decode_numbers(klv) for v in sample]


def parse_telemetry_blob(blob: bytes) -> Telemetry:
    tel = Telemetry()
    # Device id lives directly under DEVC.
    for _path, item in walk(blob):
        if item.key == "DVID" and tel.device_id is None:
            tel.device_id = decode_numbers(item)[0][0]
    # GPSU, SCAL and GPS5 are siblings inside the GPS STRM container.
    for _path, strm in walk(blob):
        if strm.key != "STRM":
            continue
        scal = None
        gps5 = None
        gpsu = None
        gpsf = None
        for c in iter_klv(strm.payload):
            if c.key == "GPSU":
                gpsu = c
            elif c.key == "GPSF":
                gpsf = decode_numbers(c)[0][0]
            elif c.key == "SCAL":
                scal = _flatten(c)
            elif c.key == "GPS5":
                gps5 = c
        if gpsu is not None:
            when = decode_utc(gpsu.payload)
            fixed = gpsf is not None and gpsf >= 2
            if fixed and not tel.utc_reliable:
                tel.first_utc = when
                tel.has_utc = True
                tel.utc_reliable = True
            elif tel.first_utc is None:
                tel.first_utc = when          # fallback, but unreliable
                tel.has_utc = True
                tel.utc_reliable = when.year != FIRMWARE_DEFAULT_YEAR
        if gps5 is not None:
            tel.has_gps = True
            if scal and len(scal) >= 5 and scal[4]:
                divisor = float(scal[4])
            elif scal and len(scal) == 1 and scal[0]:
                divisor = float(scal[0])
            else:
                divisor = 1.0
            for sample in decode_numbers(gps5):
                tel.speed_3d_ms.append(sample[4] / divisor)
    # Uniform time base: GPS5 is ~18 Hz; for windowing we only need a
    # monotonic axis, so space samples evenly.
    n = len(tel.speed_3d_ms)
    if n:
        tel.t_s = [i * (1.0 / 18.0) for i in range(n)]
    return tel


def extract_gpmf_blob(path: str) -> bytes:
    index = find_gpmf_stream_index(path)
    if index is None:
        return b""
    out = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-map", f"0:{index}",
         "-codec", "copy", "-f", "data", "-"],
        check=True, capture_output=True,
    )
    return out.stdout


def read_telemetry(path: str) -> Telemetry | None:
    blob = extract_gpmf_blob(path)
    if not blob:
        return None
    return parse_telemetry_blob(blob)
