"""Minimal GPMF (GoPro Metadata Format) KLV parser.

Only the pieces reconnaissance needs: iterate records, descend into
nested containers, decode numeric leaves, and decode the GPSU UTC
timestamp. Intentionally dependency-free so it can be tested on
hand-built byte strings.
"""
from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

# GPMF type char -> (struct format char, element byte size)
_NUMERIC = {
    "b": ("b", 1),
    "B": ("B", 1),
    "s": ("h", 2),
    "S": ("H", 2),
    "l": ("i", 4),
    "L": ("I", 4),
    "j": ("q", 8),
    "J": ("Q", 8),
    "f": ("f", 4),
    "d": ("d", 8),
}


@dataclass
class KLV:
    key: str
    type: str
    sample_size: int
    repeat: int
    payload: bytes

    @property
    def is_nested(self) -> bool:
        return self.type == ""


def iter_klv(data: bytes) -> Iterator[KLV]:
    offset = 0
    n = len(data)
    while offset + 8 <= n:
        key = data[offset : offset + 4].decode("latin-1")
        type_byte = data[offset + 4]
        sample_size = data[offset + 5]
        repeat = struct.unpack(">H", data[offset + 6 : offset + 8])[0]
        payload_len = sample_size * repeat
        start = offset + 8
        payload = data[start : start + payload_len]
        type_char = "" if type_byte == 0 else chr(type_byte)
        yield KLV(key=key, type=type_char, sample_size=sample_size, repeat=repeat, payload=payload)
        padded = payload_len + ((-payload_len) % 4)
        offset = start + padded


def walk(data: bytes, _path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], KLV]]:
    for item in iter_klv(data):
        yield _path, item
        if item.is_nested:
            yield from walk(item.payload, _path + (item.key,))


def decode_numbers(klv: KLV) -> list[tuple]:
    fmt = _NUMERIC.get(klv.type)
    if fmt is None:
        raise ValueError(f"non-numeric GPMF type {klv.type!r} for key {klv.key}")
    struct_char, elem_size = fmt
    per_sample = klv.sample_size // elem_size
    result = []
    for i in range(klv.repeat):
        chunk = klv.payload[i * klv.sample_size : (i + 1) * klv.sample_size]
        result.append(struct.unpack(">" + struct_char * per_sample, chunk))
    return result


def decode_utc(payload: bytes) -> datetime:
    text = payload.decode("latin-1").strip("\x00")
    # Format: yymmddhhmmss.sss
    year = 2000 + int(text[0:2])
    month = int(text[2:4])
    day = int(text[4:6])
    hour = int(text[6:8])
    minute = int(text[8:10])
    second = int(text[10:12])
    micro = int(round(float(text[12:]) * 1_000_000)) if len(text) > 12 else 0
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=timezone.utc)
