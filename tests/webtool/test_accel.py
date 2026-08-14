import math
import struct

from tandem.webtool.accel import parse_accel


def _klv(key: bytes, type_char: bytes, sample_size: int, repeat: int, payload: bytes) -> bytes:
    header = key + type_char + bytes([sample_size]) + struct.pack(">H", repeat)
    pad = (-len(payload)) % 4
    return header + payload + b"\x00" * pad


def _devc_with_accl(scal: int | None, samples: list[tuple[int, int, int]]) -> bytes:
    children = b""
    if scal is not None:
        children += _klv(b"SCAL", b"s", 2, 1, struct.pack(">h", scal))
    payload = b"".join(struct.pack(">hhh", *s) for s in samples)
    children += _klv(b"ACCL", b"s", 6, len(samples), payload)
    strm = _klv(b"STRM", b"\x00", 1, len(children), children)
    return _klv(b"DEVC", b"\x00", 1, len(strm), strm)


def test_parse_accel_scales_and_times_per_packet():
    blob = _devc_with_accl(100, [(100, 0, 0), (0, 200, 0)])
    series = parse_accel(blob, packet_times=[(10.0, 1.0)])
    # two samples spread across the packet: centre of each half
    assert series.t == [10.25, 10.75]
    assert series.ax == [1.0, 0.0]
    assert series.ay == [0.0, 2.0]
    assert series.amag == [1.0, 2.0]
    assert series.warnings == []


def test_parse_accel_missing_scal_warns_and_keeps_raw():
    blob = _devc_with_accl(None, [(3, 4, 0)])
    series = parse_accel(blob, packet_times=[(0.0, 1.0)])
    assert series.amag == [5.0]  # raw 3-4-5, no scaling
    assert any("SCAL" in w for w in series.warnings)


def test_parse_accel_packet_mismatch_falls_back_to_uniform():
    # two DEVC packets, but only one packet_time -> uniform fallback over duration
    blob = _devc_with_accl(1, [(1, 0, 0)]) + _devc_with_accl(1, [(0, 1, 0)])
    series = parse_accel(blob, packet_times=[(0.0, 1.0)], video_duration=4.0)
    assert series.t == [1.0, 3.0]  # centres of two halves of [0, 4]
    assert any("mismatch" in w for w in series.warnings)
