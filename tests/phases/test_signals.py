import struct

from tandem.phases.signals import Signals, resample, build_signals


def _klv(key, type_char, sample_size, repeat, payload):
    header = key + type_char + bytes([sample_size]) + struct.pack(">H", repeat)
    pad = (-len(payload)) % 4
    return header + payload + b"\x00" * pad


def test_resample_linear_endpoints_and_midpoint():
    out = resample([0.0, 10.0], 3)
    assert out == [0.0, 5.0, 10.0]


def test_resample_single_value_series():
    assert resample([7.0], 4) == [7.0, 7.0, 7.0, 7.0]


def test_build_signals_computes_magnitude_and_flags():
    # One STRM with SCAL+ACCL (3 axis int16), one with SCAL+GPS5.
    accl_scal = _klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 100))  # divide raw by 100 -> m/s^2
    # two accel samples: (300,400,0)->|a|=5.0 ; (600,800,0)->|a|=10.0 after /100
    accl_payload = struct.pack(">3h", 300, 400, 0) + struct.pack(">3h", 600, 800, 0)
    accl = _klv(b"ACCL", b"s", 6, 2, accl_payload)
    strm_accl = _klv(b"STRM", b"\x00", 1, len(accl_scal + accl), accl_scal + accl)

    gps_scal = _klv(b"SCAL", b"l", 4, 5, struct.pack(">5i", 10000000, 10000000, 1000, 1000, 1000))
    gps5 = _klv(b"GPS5", b"l", 20, 2,
                struct.pack(">5i", 0, 0, 0, 0, 55000) + struct.pack(">5i", 0, 0, 0, 0, 6000))
    strm_gps = _klv(b"STRM", b"\x00", 1, len(gps_scal + gps5), gps_scal + gps5)

    devc = _klv(b"DEVC", b"\x00", 1, len(strm_accl + strm_gps), strm_accl + strm_gps)

    sig = build_signals(devc, fs=10.0)
    assert sig.has_accel is True
    assert sig.has_gps is True
    assert sig.fs == 10.0
    # grids share length and axis
    assert len(sig.accel_mag) == len(sig.speed_3d) == len(sig.t_s)
    # magnitude endpoints preserved after resample (5.0 .. 10.0)
    assert abs(sig.accel_mag[0] - 5.0) < 1e-6
    assert abs(sig.accel_mag[-1] - 10.0) < 1e-6
    # speed endpoints preserved (55.0 .. 6.0)
    assert abs(sig.speed_3d[0] - 55.0) < 1e-6
    assert abs(sig.speed_3d[-1] - 6.0) < 1e-6


def test_build_signals_without_streams_sets_flags_false():
    devc = _klv(b"DEVC", b"\x00", 1, 0, b"")
    sig = build_signals(devc)
    assert sig.has_accel is False
    assert sig.has_gps is False
    assert sig.accel_mag == [] and sig.speed_3d == []
