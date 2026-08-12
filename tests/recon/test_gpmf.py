import struct
from datetime import timezone

from tandem.recon.gpmf import iter_klv, walk, decode_numbers, decode_utc


def _klv(key: bytes, type_char: bytes, sample_size: int, repeat: int, payload: bytes) -> bytes:
    header = key + type_char + bytes([sample_size]) + struct.pack(">H", repeat)
    pad = (-len(payload)) % 4
    return header + payload + b"\x00" * pad


def test_iter_klv_reads_leaf_and_padding():
    payload = struct.pack(">hhh", 1, 2, 3)  # one ACCL sample: 3 int16 -> size 6
    blob = _klv(b"ACCL", b"s", 6, 1, payload)
    items = list(iter_klv(blob))
    assert len(items) == 1
    item = items[0]
    assert item.key == "ACCL"
    assert item.type == "s"
    assert item.sample_size == 6
    assert item.repeat == 1
    assert decode_numbers(item) == [(1, 2, 3)]


def test_walk_descends_into_nested_containers():
    inner = _klv(b"DVID", b"L", 4, 1, struct.pack(">I", 0xDEADBEEF))
    devc = _klv(b"DEVC", b"\x00", 1, len(inner), inner)
    paths = {path + (k.key,): k for path, k in walk(devc)}
    dvid = paths[("DEVC", "DVID")]
    assert decode_numbers(dvid) == [(0xDEADBEEF,)]


def test_decode_utc_parses_gopro_timestamp():
    dt = decode_utc(b"260812091403.250")
    assert (dt.year, dt.month, dt.day) == (2026, 8, 12)
    assert (dt.hour, dt.minute, dt.second) == (9, 14, 3)
    assert dt.microsecond == 250000
    assert dt.tzinfo == timezone.utc
