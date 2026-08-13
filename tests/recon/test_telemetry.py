import struct

from tandem.recon.telemetry import parse_telemetry_blob


def _klv(key, type_char, sample_size, repeat, payload):
    header = key + type_char + bytes([sample_size]) + struct.pack(">H", repeat)
    pad = (-len(payload)) % 4
    return header + payload + b"\x00" * pad


def test_parse_blob_reads_device_utc_and_scaled_speed():
    dvid = _klv(b"DVID", b"L", 4, 1, struct.pack(">I", 0x1001))
    gpsu = _klv(b"GPSU", b"U", 16, 1, b"260812091403.250")
    # SCAL: one divisor per GPS5 field (lat, lon, alt, speed_2d, speed_3d)
    scal = _klv(b"SCAL", b"l", 4, 5, struct.pack(">5i", 10000000, 10000000, 1000, 1000, 1000))
    # Two GPS5 samples [lat, lon, alt, s2d, s3d] int32; raw 3D speed 55000 -> 55.0 m/s
    gps5_payload = struct.pack(">5i", 0, 0, 0, 0, 55000) + struct.pack(">5i", 0, 0, 0, 0, 54000)
    gps5 = _klv(b"GPS5", b"l", 20, 2, gps5_payload)
    inner_strm = gpsu + scal + gps5
    strm = _klv(b"STRM", b"\x00", 1, len(inner_strm), inner_strm)
    inner = dvid + strm
    devc = _klv(b"DEVC", b"\x00", 1, len(inner), inner)

    tel = parse_telemetry_blob(devc)
    assert tel.device_id == 0x1001
    assert tel.has_utc is True
    assert tel.has_gps is True
    assert tel.first_utc.hour == 9 and tel.first_utc.minute == 14
    assert tel.speed_3d_ms == [55.0, 54.0]
    assert tel.t_s[0] == 0.0


def test_parse_blob_without_gps_flags_absence():
    dvid = _klv(b"DVID", b"L", 4, 1, struct.pack(">I", 7))
    devc = _klv(b"DEVC", b"\x00", 1, len(dvid), dvid)
    tel = parse_telemetry_blob(devc)
    assert tel.has_gps is False
    assert tel.has_utc is False
    assert tel.first_utc is None


def test_utc_from_fixed_sample_is_reliable():
    # STRM 1: GPSF=0 (no fix), GPSU=firmware default. STRM 2: GPSF=3, GPSU=real.
    gpsf0 = _klv(b"GPSF", b"L", 4, 1, struct.pack(">I", 0))
    gpsu0 = _klv(b"GPSU", b"U", 16, 1, b"210307000002.300")
    strm0 = _klv(b"STRM", b"\x00", 1, len(gpsf0 + gpsu0), gpsf0 + gpsu0)
    gpsf3 = _klv(b"GPSF", b"L", 4, 1, struct.pack(">I", 3))
    gpsu3 = _klv(b"GPSU", b"U", 16, 1, b"260523143500.000")
    strm3 = _klv(b"STRM", b"\x00", 1, len(gpsf3 + gpsu3), gpsf3 + gpsu3)
    devc = _klv(b"DEVC", b"\x00", 1, len(strm0 + strm3), strm0 + strm3)

    tel = parse_telemetry_blob(devc)
    assert tel.utc_reliable is True
    assert tel.first_utc.year == 2026 and tel.first_utc.month == 5


def test_only_prefix_default_utc_is_unreliable():
    gpsf0 = _klv(b"GPSF", b"L", 4, 1, struct.pack(">I", 0))
    gpsu0 = _klv(b"GPSU", b"U", 16, 1, b"210307000002.300")
    strm0 = _klv(b"STRM", b"\x00", 1, len(gpsf0 + gpsu0), gpsf0 + gpsu0)
    devc = _klv(b"DEVC", b"\x00", 1, len(strm0), strm0)

    tel = parse_telemetry_blob(devc)
    assert tel.utc_reliable is False
