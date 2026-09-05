"""Parse worn DJI action-cam telemetry (the ``djmd`` "CAM meta" stream).

DJI OsmoAction cameras log per-frame telemetry as protobuf (``dvtm_ac206``), not
GoPro GPMF, so ``extract_gpmf_blob`` finds nothing on DJI-filmed jumps. This module
decodes the accelerometer AND gyroscope out of that stream so the same phase
detector can segment DJI footage: accel reads ~1 g at rest and collapses toward 0
in free-fall (weightlessness), while gyro is quiet on the ground and spikes during
free-fall tumbling and the break-off turn — the same signatures GoPro's ACCL/GYRO
give.

Per frame the message carries two length-delimited sub-messages: field 9 (tag
``0x4a``) is the accelerometer, field 10 (tag ``0x52``) the gyroscope; each holds
its vector as wire-type-5 floats in fields 1/2/3 (a zero component is omitted, per
protobuf). This is reverse-engineered, not a documented schema — best-effort: any
failure returns None/empty so the caller falls back cleanly.
"""
from __future__ import annotations

import struct
import subprocess

# Plausible specific-force magnitude band, in g. The upper bound rejects mis-decoded
# triples; the lower bound rejects parse artifacts: a frame whose accel sub-message we
# fail to locate decodes to (0,0,0), and a real accelerometer never reads exactly zero
# — even true free-fall keeps ~0.2 g of buffeting. Left in, those zeros poison the
# free-fall detector's per-window minimum and fabricate a jump on a ground clip.
_MAG_MIN_G = 0.02
_MAG_MAX_G = 16.0
_ACCEL_TAG = 0x4A          # protobuf field 9, wire type 2 (accel sub-message)
_GYRO_TAG = 0x52           # protobuf field 10, wire type 2 (gyro sub-message)


def _djmd_stream_index(path: str) -> int | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_tag_string",
             "-of", "csv=p=0", path], capture_output=True, text=True, check=False).stdout
    except OSError:
        return None
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].strip() == "djmd":
            try:
                return int(parts[0])
            except ValueError:
                return None
    return None


def probe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], capture_output=True, text=True, check=False).stdout.strip()
        return float(out)
    except (OSError, ValueError):
        return 0.0


def _packet_sizes(path: str, idx: int) -> list[int]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", str(idx),
             "-show_entries", "packet=size", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=False).stdout
    except OSError:
        return []
    sizes = []
    for line in out.splitlines():
        line = line.strip().rstrip(",")
        if line.isdigit():
            sizes.append(int(line))
    return sizes


def _extract_stream(path: str, idx: int) -> bytes:
    try:
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-map", f"0:{idx}", "-c", "copy",
             "-f", "data", "-"], capture_output=True, check=False).stdout
    except OSError:
        return b""


def _vec3(sub: bytes) -> tuple[float, float, float]:
    """Read a 3-vector from a sub-message: wire-type-5 floats in fields 1/2/3,
    a missing (zero) component defaulting to 0.0."""
    vals: dict[int, float] = {}
    i, n = 0, len(sub)
    while i < n:
        tag = sub[i]; f = tag >> 3; wt = tag & 7; i += 1
        if wt == 5:
            if i + 4 > n:
                break
            vals[f] = struct.unpack("<f", sub[i:i + 4])[0]; i += 4
        elif wt == 0:
            while i < n and sub[i] & 0x80:
                i += 1
            i += 1
        elif wt == 2:
            if i >= n:
                break
            ln = sub[i]; i += 1 + ln
        else:
            break
    return (vals.get(1, 0.0), vals.get(2, 0.0), vals.get(3, 0.0))


def _find_vec(frame: bytes, tag: int) -> tuple[float, float, float] | None:
    """Locate the length-delimited sub-message with the given field tag and decode
    its 3-vector. Scans for the tag byte, validating the length is sub-message-sized."""
    i = frame.find(bytes([tag]))
    while i != -1 and i + 2 <= len(frame):
        ln = frame[i + 1]
        if 4 <= ln <= 40 and i + 2 + ln <= len(frame):
            return _vec3(frame[i + 2:i + 2 + ln])
        i = frame.find(bytes([tag]), i + 1)
    return None


def read_telemetry(path: str):
    """Return ``(accels_g, gyros, duration_s)`` for a DJI recording, or None.

    ``accels_g`` and ``gyros`` are per-frame lists of ``(x, y, z)`` — accel in g,
    gyro in the camera's native angular-rate units. None when the file carries no
    DJI telemetry or it cannot be framed/parsed.
    """
    idx = _djmd_stream_index(path)
    if idx is None:
        return None
    blob = _extract_stream(path, idx)
    sizes = _packet_sizes(path, idx)
    if not blob or not sizes:
        return None
    frames, off = [], 0
    for s in sizes:
        frames.append(blob[off:off + s]); off += s
    accels, gyros = [], []
    for fr in frames:
        a = _find_vec(fr, _ACCEL_TAG)
        if a is None:
            continue
        m = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5
        if not (_MAG_MIN_G <= m < _MAG_MAX_G):
            continue
        g = _find_vec(fr, _GYRO_TAG) or (0.0, 0.0, 0.0)
        accels.append(a)
        gyros.append(g)
    if len(accels) < 2:
        return None
    return accels, gyros, probe_duration(path)
