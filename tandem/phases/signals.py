"""Build an analysis-grade signal set from GPMF telemetry.

Two signals only: accelerometer magnitude (m/s^2) and GPS 3D speed
(m/s). GPS altitude is never used. Accel (~200 Hz) and GPS (~18 Hz)
are resampled onto one uniform grid (default 10 Hz). Per-stream timing
is assumed uniform across the recording — precise GPMF payload timing
is a later refinement; seconds-scale phase boundaries do not need it.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from tandem.recon.gpmf import decode_numbers, iter_klv, walk
from tandem.recon.telemetry import extract_gpmf_blob

G = 9.80665  # standard gravity, m/s^2 -> g conversion
STD_WINDOW_S = 1.0  # rolling-std window for the oscillation feature


@dataclass
class Signals:
    t_s: list[float] = field(default_factory=list)
    accel_mag: list[float] = field(default_factory=list)
    accel_min: list[float] = field(default_factory=list)
    speed_3d: list[float] = field(default_factory=list)
    ax: list[float] = field(default_factory=list)
    ay: list[float] = field(default_factory=list)
    az: list[float] = field(default_factory=list)
    accel_std: list[float] = field(default_factory=list)
    gyro_mag: list[float] = field(default_factory=list)   # |angular velocity|, rad/s
    gx: list[float] = field(default_factory=list)          # per-axis rate, rad/s
    gy: list[float] = field(default_factory=list)
    gz: list[float] = field(default_factory=list)
    iso: list[float] = field(default_factory=list)         # ISO (ISOE) — high in cabin, ~100 in daylight
    shutter: list[float] = field(default_factory=list)     # exposure time, s (SHUT) — slow in cabin, fast in daylight
    face_count: list[float] = field(default_factory=list)  # faces detected (FACE) — pair close = interview/exit/canopy
    smile: list[float] = field(default_factory=list)       # max smile %, 0..100 (0 when no face)
    blink: list[float] = field(default_factory=list)       # max blink %, 0..100
    face_area: list[float] = field(default_factory=list)   # max face bbox area, fraction of frame (close-up = large)
    audio_level: list[float] = field(default_factory=list) # AALP RMS audio level, dBFS — drops under canopy (wind falls)
    scene_indoor: list[float] = field(default_factory=list)  # SCEN INDO probability — high in the cabin, drops at exit
    fs: float = 10.0
    has_accel: bool = False
    has_gps: bool = False
    has_gyro: bool = False
    has_exposure: bool = False
    has_face: bool = False
    has_audio: bool = False
    has_scene: bool = False
    source: str = "gpmf"   # "gpmf" (GoPro) or "dji" (worn DJI action-cam, accel-only)


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


def rolling_std(values: list[float], win: int) -> list[float]:
    """Population std over a centered window, clamped to the array bounds.

    For index i, uses values[max(0, i-win//2) : min(n, i+win//2+1)].
    Returns a list the same length as `values`.
    """
    n = len(values)
    half = max(win, 0) // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = values[lo:hi]
        m = sum(window) / len(window)
        var = sum((v - m) ** 2 for v in window) / len(window)
        out.append(math.sqrt(var))
    return out


def pool_min(values: list[float], n_out: int) -> list[float]:
    if not values or n_out <= 0:
        return []
    n_in = len(values)
    out = []
    for j in range(n_out):
        lo = (j * n_in) // n_out
        hi = max(lo + 1, ((j + 1) * n_in) // n_out)
        out.append(min(values[lo:hi]))
    return out


def _stream_children(blob: bytes):
    for _path, strm in walk(blob):
        if strm.key == "STRM":
            yield list(iter_klv(strm.payload))


def _accel_magnitudes(children) -> tuple[list[float], list[float], list[float], list[float]] | None:
    """Return (|a| in m/s^2, ax, ay, az in g) for one ACCL payload, or None."""
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
    mags: list[float] = []
    ax: list[float] = []
    ay: list[float] = []
    az: list[float] = []
    for sample in decode_numbers(accl):
        corrected = [v / divisor for v in sample]
        mags.append(math.sqrt(sum(v ** 2 for v in corrected)))
        if len(corrected) >= 3:
            ax.append(corrected[0] / G)
            ay.append(corrected[1] / G)
            az.append(corrected[2] / G)
    return mags, ax, ay, az


def _gyro_rates(children) -> tuple[list[float], list[float], list[float], list[float]] | None:
    """Return (|w| in rad/s, gx, gy, gz in rad/s) for one GYRO payload, or None.

    The operator break-off (отворот) is a sustained body rotation, so angular
    velocity separates it from free-fall buffeting far better than linear accel.
    """
    scal = None
    gyro = None
    for c in children:
        if c.key == "SCAL":
            scal = _flatten(c)
        elif c.key == "GYRO":
            gyro = c
    if gyro is None:
        return None
    divisor = float(scal[0]) if (scal and scal[0]) else 1.0
    mags: list[float] = []
    gx: list[float] = []
    gy: list[float] = []
    gz: list[float] = []
    for sample in decode_numbers(gyro):
        corrected = [v / divisor for v in sample]
        if len(corrected) >= 3:
            gx.append(corrected[0])
            gy.append(corrected[1])
            gz.append(corrected[2])
            mags.append(math.sqrt(sum(v ** 2 for v in corrected[:3])))
    return mags, gx, gy, gz


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
    try:
        return [sample[4] / divisor for sample in decode_numbers(gps5)]
    except (ValueError, struct.error):
        # A corrupt GPS5 KLV (e.g. a garbage type byte from a misaligned stream)
        # must not sink the whole recording — GPS is only a fallback for the
        # freefall/descent cues, which the accelerometer also carries. Skip this
        # payload's GPS samples and let ACCL/GYRO segmentation stand.
        return None


def _exposure_values(children) -> tuple[list[float] | None, list[float] | None]:
    """(ISO values, shutter times in s) from one payload's ISOE/SHUT, or (None, None).

    Both are raw (no SCAL). Exposure jumps at exit — leaving the dark cabin for
    daylight drops ISO from ~500-1200 to ~100 and shutter from ~1/220 to ~1/2000 —
    so it is an independent corroborator of the accelerometer exit event.
    """
    iso = shut = None
    for c in children:
        if c.key == "ISOE":
            iso = c
        elif c.key == "SHUT":
            shut = c
    try:
        iso_v = [s[0] for s in decode_numbers(iso)] if iso is not None else None
        shut_v = [s[0] for s in decode_numbers(shut)] if shut is not None else None
    except (ValueError, struct.error):
        return None, None  # corrupt exposure KLV: drop this payload, keep the file
    return iso_v, shut_v


# FACE payload is a compound type "BBSSSSSBB" (14 bytes/face on HERO10/11):
# version(4), confidence, id, bbox x, y, w, h, blink %, smile %. Faces are detected
# when the pair is close (cabin / exit / under canopy), so FACE mainly serves the
# interview and passenger-reaction scenes, not mid-free-fall (pair too distant).
_FACE_STRUCT = ">BBHHHHHBB"


def _face_values(children):
    """Per-payload FACE aggregate (n_faces, max_smile, max_blink, max_area) or None."""
    face = None
    for c in children:
        if c.key == "FACE":
            face = c
    if face is None:
        return None
    n = face.repeat
    smile = blink = area = 0.0
    if face.sample_size == 14:
        for i in range(n):
            rec = face.payload[i * 14:(i + 1) * 14]
            if len(rec) == 14:
                v = struct.unpack(_FACE_STRUCT, rec)
                blink = max(blink, float(v[7]))
                smile = max(smile, float(v[8]))
                area = max(area, (v[5] * v[6]) / (65535.0 ** 2))
    return float(n), smile, blink, area


def _audio_level(children) -> float | None:
    """Mean RMS audio level (dBFS) for one AALP payload, ignoring -128 (silence/invalid)."""
    aalp = next((c for c in children if c.key == "AALP"), None)
    if aalp is None:
        return None
    try:
        vals = [v[0] for v in decode_numbers(aalp) if v and v[0] > -128]
    except (ValueError, struct.error):
        return None  # corrupt audio-level KLV: drop this payload, keep the file
    return (sum(vals) / len(vals)) if vals else None


def _scene_indoor(children) -> float | None:
    """SCEN indoor (INDO) probability — a compound "Ff" of (FourCC class, float prob)."""
    scen = next((c for c in children if c.key == "SCEN"), None)
    if scen is None or scen.sample_size != 8:
        return None
    indoor = 0.0
    for i in range(scen.repeat):
        rec = scen.payload[i * 8:(i + 1) * 8]
        if len(rec) == 8 and rec[:4] == b"INDO":
            indoor = struct.unpack(">f", rec[4:8])[0]
    return indoor


def build_signals(blob: bytes, fs: float = 10.0) -> Signals:
    accel_raw: list[float] = []
    ax_raw: list[float] = []
    ay_raw: list[float] = []
    az_raw: list[float] = []
    speed_raw: list[float] = []
    gmag_raw: list[float] = []
    gx_raw: list[float] = []
    gy_raw: list[float] = []
    gz_raw: list[float] = []
    iso_raw: list[float] = []
    shut_raw: list[float] = []
    fc_raw: list[float] = []
    smile_raw: list[float] = []
    blink_raw: list[float] = []
    area_raw: list[float] = []
    audio_raw: list[float] = []
    scene_raw: list[float] = []
    saw_accel = False
    saw_gps = False
    saw_gyro = False
    saw_exposure = False
    saw_face = False
    saw_audio = False
    saw_scene = False
    for children in _stream_children(blob):
        a = _accel_magnitudes(children)
        if a is not None:
            saw_accel = True
            mags, ax, ay, az = a
            accel_raw.extend(mags)
            ax_raw.extend(ax)
            ay_raw.extend(ay)
            az_raw.extend(az)
        g = _gyro_rates(children)
        if g is not None:
            saw_gyro = True
            gmag, gx, gy, gz = g
            gmag_raw.extend(gmag)
            gx_raw.extend(gx)
            gy_raw.extend(gy)
            gz_raw.extend(gz)
        s = _gps_speeds(children)
        if s is not None:
            saw_gps = True
            speed_raw.extend(s)
        iso_v, shut_v = _exposure_values(children)
        if iso_v:
            saw_exposure = True
            iso_raw.extend(iso_v)
        if shut_v:
            saw_exposure = True
            shut_raw.extend(shut_v)
        fv = _face_values(children)
        if fv is not None:
            saw_face = True
            fc_raw.append(fv[0])
            smile_raw.append(fv[1])
            blink_raw.append(fv[2])
            area_raw.append(fv[3])
        av = _audio_level(children)
        if av is not None:
            saw_audio = True
            audio_raw.append(av)
        sc = _scene_indoor(children)
        if sc is not None:
            saw_scene = True
            scene_raw.append(sc)
    sig = Signals(fs=fs, has_accel=saw_accel, has_gps=saw_gps, has_gyro=saw_gyro,
                  has_exposure=saw_exposure, has_face=saw_face,
                  has_audio=saw_audio, has_scene=saw_scene)
    # Recording duration: longer of the two streams at their nominal rates.
    accel_dur = (len(accel_raw) / 200.0) if accel_raw else 0.0
    gps_dur = (len(speed_raw) / 18.0) if speed_raw else 0.0
    duration = max(accel_dur, gps_dur)
    if duration <= 0:
        return sig
    n_out = max(2, int(round(duration * fs)))
    sig.t_s = [i / fs for i in range(n_out)]
    sig.accel_mag = resample(accel_raw, n_out) if accel_raw else [0.0] * n_out
    sig.accel_min = pool_min(accel_raw, n_out) if accel_raw else [0.0] * n_out
    sig.speed_3d = resample(speed_raw, n_out) if speed_raw else [0.0] * n_out
    sig.ax = resample(ax_raw, n_out) if ax_raw else [0.0] * n_out
    sig.ay = resample(ay_raw, n_out) if ay_raw else [0.0] * n_out
    sig.az = resample(az_raw, n_out) if az_raw else [0.0] * n_out
    # accel_std is in g (oscillation amplitude): the orbit/freefall thresholds are
    # calibrated in g against real footage, so std must be on |a|/G, not the m/s^2 magnitude.
    sig.accel_std = rolling_std([a / G for a in sig.accel_mag], round(STD_WINDOW_S * fs))
    if gmag_raw:
        sig.gyro_mag = resample(gmag_raw, n_out)
        sig.gx = resample(gx_raw, n_out)
        sig.gy = resample(gy_raw, n_out)
        sig.gz = resample(gz_raw, n_out)
    else:
        sig.gyro_mag = [0.0] * n_out
        sig.gx = [0.0] * n_out
        sig.gy = [0.0] * n_out
        sig.gz = [0.0] * n_out
    sig.iso = resample(iso_raw, n_out) if iso_raw else [0.0] * n_out
    sig.shutter = resample(shut_raw, n_out) if shut_raw else [0.0] * n_out
    sig.face_count = resample(fc_raw, n_out) if fc_raw else [0.0] * n_out
    sig.smile = resample(smile_raw, n_out) if smile_raw else [0.0] * n_out
    sig.blink = resample(blink_raw, n_out) if blink_raw else [0.0] * n_out
    sig.face_area = resample(area_raw, n_out) if area_raw else [0.0] * n_out
    sig.audio_level = resample(audio_raw, n_out) if audio_raw else [0.0] * n_out
    sig.scene_indoor = resample(scene_raw, n_out) if scene_raw else [0.0] * n_out
    return sig


def build_signals_from_dji(path: str, fs: float = 10.0) -> Signals | None:
    """Build a Signals from a worn DJI action-cam's ``djmd`` telemetry (accel + gyro).

    DJI cameras log no GoPro GPMF, so the normal path yields nothing; this decodes
    their protobuf accelerometer and gyroscope instead. GPS/exposure are not decoded,
    but accel gives exit and free-fall and gyro gives the break-off turn, so all
    telemetry boundaries are available; the visual probe still supplies drogue/deploy.
    Returns None when there is no DJI telemetry or too few samples.
    """
    from tandem.recon.dji import read_telemetry
    tel = read_telemetry(path)
    if tel is None:
        return None
    accels, gyros, duration = tel
    if duration <= 0:
        duration = len(accels) / 60.0  # DJI logs one sample per frame (~60 Hz)
    # DJI accel is in g; the detector's thresholds are in m/s^2, so scale by G.
    mag = [(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5 * G for a in accels]
    ax = [a[0] * G for a in accels]
    ay = [a[1] * G for a in accels]
    az = [a[2] * G for a in accels]
    gx = [g[0] for g in gyros]
    gy = [g[1] for g in gyros]
    gz = [g[2] for g in gyros]
    gmag = [(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5 for g in gyros]

    sig = Signals(fs=fs, has_accel=True, has_gps=False, has_gyro=True,
                  has_exposure=False, has_face=False, has_audio=False, has_scene=False,
                  source="dji")
    n_out = max(2, int(round(duration * fs)))
    sig.t_s = [i / fs for i in range(n_out)]
    sig.accel_mag = resample(mag, n_out)
    sig.accel_min = pool_min(mag, n_out)
    sig.ax = resample(ax, n_out)
    sig.ay = resample(ay, n_out)
    sig.az = resample(az, n_out)
    sig.accel_std = rolling_std([a / G for a in sig.accel_mag], round(STD_WINDOW_S * fs))
    sig.gyro_mag = resample(gmag, n_out)
    sig.gx = resample(gx, n_out)
    sig.gy = resample(gy, n_out)
    sig.gz = resample(gz, n_out)
    zeros = [0.0] * n_out
    sig.speed_3d = list(zeros)
    sig.iso = list(zeros); sig.shutter = list(zeros)
    sig.face_count = list(zeros); sig.smile = list(zeros)
    sig.blink = list(zeros); sig.face_area = list(zeros)
    sig.audio_level = list(zeros); sig.scene_indoor = list(zeros)
    return sig


def build_signals_from_file(path: str, fs: float = 10.0) -> Signals | None:
    blob = extract_gpmf_blob(path)
    if blob:
        return build_signals(blob, fs=fs)
    # No GoPro GPMF: fall back to worn DJI action-cam telemetry (accel-only).
    return build_signals_from_dji(path, fs=fs)
