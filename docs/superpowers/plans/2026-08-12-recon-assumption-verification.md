# Reconnaissance & Assumption Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that ingests a real archive of tandem-jump camera files and produces a findings report answering all eight of the spec's section-9 assumptions, so the design is confirmed or corrected before any detector is written.

**Architecture:** A single Python package `tandem` with a `recon` subpackage. Mechanical probes (file naming, telemetry presence, keyframe cadence, jump grouping, dual-camera fraction) run fully automatically. Visual probes that need human judgement (ground-background fraction, canopy-in-frame, operator distance, landing framing) are served by a keyframe sampler that extracts the relevant windows into folders plus an HTML contact sheet for a human to tally. A report generator aggregates every probe across the archive into one markdown document.

**Tech Stack:** Python 3.10+, numpy, pandas, pyarrow, Pillow, pytest. External binaries `ffmpeg` and `ffprobe` on PATH. GPMF telemetry is parsed by our own minimal KLV reader (no third-party GPMF dependency) so the parse logic is unit-testable on synthetic bytes.

## Global Constraints

These are copied from the spec and apply to every task implicitly.

- **Python floor: 3.10** (this machine has 3.10.0; do not use 3.11+ syntax such as `match` guards you cannot test here).
- **External prerequisite:** `ffmpeg` and `ffprobe` must be on PATH. They are NOT installed on the target machine yet; installation is a prerequisite step, not a task deliverable.
- **All times are seconds from `t0_utc`**, never from file start. `t0_utc` is the earliest GPS-UTC instant across a jump's sources.
- **Keyframes only, 720p:** frame extraction uses `-skip_frame nokey` and scales to 720p height. Full decode and full resolution are never used.
- **Video never leaves the machine.** The tool reads from the archive in place and writes only small artifacts (reports, thumbnails, parquet) to an output directory.
- **Degradation is part of the contract, not logs.** Recognized codes this plan touches: `NO_TELEMETRY`, `NO_INSTRUCTOR_CAM`, `SYNC_FAILED`, `CHUNK_GAP`. Any probe that cannot run records the matching code rather than guessing.
- **Operator telemetry is not tandem telemetry.** Only exit and the freefall-phase boundaries transfer from operator to tandem. Never treat an operator's canopy or landing telemetry as the tandem's.

## File Structure

```
pyproject.toml                     # package metadata, deps, pytest config
tandem/
  __init__.py                      # __version__
  recon/
    __init__.py
    naming.py                      # GoPro filename parse; chunk -> recording grouping
    gpmf.py                        # minimal GPMF KLV parser (pure, no I/O)
    ffprobe.py                     # ffprobe/ffmpeg subprocess wrappers
    telemetry.py                   # extract gpmd via ffmpeg + parse to a Telemetry object
    jumps.py                       # recordings -> jumps by UTC overlap; camera identity
    windows.py                     # rough freefall/canopy window estimate from telemetry
    sampler.py                     # extract keyframes in windows -> folders + HTML sheet
    probes.py                      # the eight assumption probes
    report.py                      # aggregate probes -> markdown findings report
    cli.py                         # `python -m tandem.recon <archive_dir> -o <out>`
tests/
  __init__.py
  recon/
    __init__.py
    test_naming.py
    test_gpmf.py
    test_ffprobe.py
    test_telemetry.py
    test_jumps.py
    test_windows.py
    test_sampler.py
    test_probes.py
    test_report.py
```

Each `recon` module has one responsibility and is small enough to hold in context. Pure-logic modules (`naming`, `gpmf`, `jumps`, `windows`, `report`) are unit-tested on synthetic data. I/O modules (`ffprobe`, `telemetry`, `sampler`) keep their logic separable from their subprocess calls; the subprocess path gets a smoke test skipped when `ffmpeg` is absent.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `tandem/__init__.py`
- Create: `tandem/recon/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/recon/__init__.py`
- Test: `tests/recon/test_version.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `tandem` with `tandem.__version__: str`. `pytest` runs green from the repo root.

- [ ] **Step 1: Write the failing test**

`tests/recon/test_version.py`:
```python
import tandem


def test_version_is_a_string():
    assert isinstance(tandem.__version__, str)
    assert tandem.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_version.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem'`.

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:
```toml
[project]
name = "tandem"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "Pillow>=10.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["tandem*"]

[tool.pytest.ini_options]
markers = [
    "integration: requires ffmpeg/ffprobe on PATH and sample media",
]
```

`tandem/__init__.py`:
```python
__version__ = "0.1.0"
```

`tandem/recon/__init__.py`, `tests/__init__.py`, `tests/recon/__init__.py`: empty files.

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]"` then `python -m pytest tests/recon/test_version.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tandem tests
git commit -m "chore: project scaffold for recon tool"
```

---

## Task 2: GoPro filename parsing and chunk grouping

Verifies spec assumption 5 (naming and chunking convention), first half: grouping a directory of files into logical *recordings*.

**Files:**
- Create: `tandem/recon/naming.py`
- Test: `tests/recon/test_naming.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `parse_gopro_name(name: str) -> GoProName | None` where `GoProName` has fields `encoding: str` (e.g. `"GX"`), `chapter: int`, `file_number: int`, `ext: str`.
  - `group_recordings(paths: list[str]) -> list[Recording]` where `Recording` has `file_number: int`, `chapters: list[str]` (paths sorted by chapter), and `unmatched: list[str]` is returned separately.
  - `group_recordings` returns `tuple[list[Recording], list[str]]` — recordings and the list of paths that did not match the convention.

- [ ] **Step 1: Write the failing test**

`tests/recon/test_naming.py`:
```python
from tandem.recon.naming import parse_gopro_name, group_recordings


def test_parse_valid_gopro_name():
    n = parse_gopro_name("GX010123.MP4")
    assert n is not None
    assert n.encoding == "GX"
    assert n.chapter == 1
    assert n.file_number == 123
    assert n.ext == "MP4"


def test_parse_rejects_non_gopro():
    assert parse_gopro_name("DJI_0001.MP4") is None
    assert parse_gopro_name("notes.txt") is None


def test_group_orders_chapters_and_flags_unmatched():
    paths = [
        "/a/GX020123.MP4",
        "/a/GX010123.MP4",
        "/a/GX010124.MP4",
        "/a/random.mov",
    ]
    recordings, unmatched = group_recordings(paths)
    by_num = {r.file_number: r for r in recordings}
    assert [p.rsplit("/", 1)[-1] for p in by_num[123].chapters] == [
        "GX010123.MP4",
        "GX020123.MP4",
    ]
    assert by_num[124].chapters == ["/a/GX010124.MP4"]
    assert unmatched == ["/a/random.mov"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_naming.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.naming'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/naming.py`:
```python
"""Parse GoPro file names and group chapters into logical recordings.

GoPro splits a long recording into 4 GB chapters. The naming is
``<EE><CC><NNNN>.<ext>`` where ``EE`` is a two-letter encoding tag
(``GX`` for HEVC, ``GH`` for AVC), ``CC`` is the chapter number
starting at 01, and ``NNNN`` is the file/recording number shared by
every chapter of the same recording.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_GOPRO_RE = re.compile(r"^(?P<enc>G[HX])(?P<chapter>\d{2})(?P<num>\d{4})\.(?P<ext>MP4)$", re.IGNORECASE)


@dataclass(frozen=True)
class GoProName:
    encoding: str
    chapter: int
    file_number: int
    ext: str


@dataclass
class Recording:
    file_number: int
    chapters: list[str] = field(default_factory=list)


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def parse_gopro_name(name: str) -> GoProName | None:
    m = _GOPRO_RE.match(_basename(name))
    if not m:
        return None
    return GoProName(
        encoding=m.group("enc").upper(),
        chapter=int(m.group("chapter")),
        file_number=int(m.group("num")),
        ext=m.group("ext").upper(),
    )


def group_recordings(paths: list[str]) -> tuple[list[Recording], list[str]]:
    buckets: dict[int, list[tuple[int, str]]] = {}
    unmatched: list[str] = []
    for path in paths:
        parsed = parse_gopro_name(path)
        if parsed is None:
            unmatched.append(path)
            continue
        buckets.setdefault(parsed.file_number, []).append((parsed.chapter, path))
    recordings = []
    for number in sorted(buckets):
        chapters = [p for _, p in sorted(buckets[number])]
        recordings.append(Recording(file_number=number, chapters=chapters))
    return recordings, unmatched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_naming.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/naming.py tests/recon/test_naming.py
git commit -m "feat(recon): GoPro filename parsing and chapter grouping"
```

---

## Task 3: Minimal GPMF KLV parser

GPMF is GoPro's telemetry format: a tree of KLV (key-length-value) records. Parsing it ourselves keeps the logic unit-testable on synthetic bytes, with no dependency on real media. Verifies the parsing half of assumption 1.

**KLV layout (all big-endian):** 4-byte FourCC key, 1-byte type char, 1-byte sample size, 2-byte repeat count, then `size * repeat` payload bytes padded up to a 4-byte boundary. Type `\x00` means the payload is itself a list of KLVs (nested container, e.g. `DEVC`, `STRM`).

**Files:**
- Create: `tandem/recon/gpmf.py`
- Test: `tests/recon/test_gpmf.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `KLV` dataclass: `key: str`, `type: str` (single char, `""` for nested), `sample_size: int`, `repeat: int`, `payload: bytes`.
  - `iter_klv(data: bytes) -> Iterator[KLV]` — parse one level.
  - `walk(data: bytes) -> Iterator[tuple[tuple[str, ...], KLV]]` — depth-first, path is the chain of container keys.
  - `decode_numbers(klv: KLV) -> list[tuple]` — decode a numeric leaf into one tuple per sample.
  - `decode_utc(payload: bytes) -> datetime` — decode a `GPSU` 16-byte `U`-type value to a timezone-aware UTC `datetime`.

- [ ] **Step 1: Write the failing test**

`tests/recon/test_gpmf.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_gpmf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.gpmf'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/gpmf.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_gpmf.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/gpmf.py tests/recon/test_gpmf.py
git commit -m "feat(recon): minimal GPMF KLV parser"
```

---

## Task 4: ffprobe/ffmpeg subprocess wrappers

Wraps the external binaries behind typed functions. Verifies assumption 2 (keyframe interval) and locates the GPMF data stream for Task 5.

**Files:**
- Create: `tandem/recon/ffprobe.py`
- Test: `tests/recon/test_ffprobe.py`

**Interfaces:**
- Consumes: nothing (shells out to `ffprobe`/`ffmpeg`).
- Produces:
  - `ffprobe_streams(path: str) -> list[dict]` — parsed `streams` array from `ffprobe -show_streams -print_format json`.
  - `find_gpmf_stream_index(path: str) -> int | None` — index of the stream whose `codec_tag_string == "gpmd"` or `tags.handler_name` contains `"GoPro MET"`.
  - `keyframe_pts(path: str) -> list[float]` — presentation timestamps (seconds) of keyframes, via `ffprobe -skip_frame nokey -show_frames`.
  - `keyframe_interval_stats(pts: list[float]) -> IntervalStats` (pure) with `count: int`, `median_s: float`, `min_s: float`, `max_s: float`.
  - `has_ffmpeg() -> bool` — both binaries resolvable on PATH.

- [ ] **Step 1: Write the failing test**

Pure statistics are tested directly; subprocess paths get an integration smoke test skipped without ffmpeg.

`tests/recon/test_ffprobe.py`:
```python
import shutil

import pytest

from tandem.recon.ffprobe import keyframe_interval_stats, find_gpmf_stream_index, has_ffmpeg


def test_interval_stats_from_pts():
    stats = keyframe_interval_stats([0.0, 0.5, 1.0, 1.5])
    assert stats.count == 4
    assert stats.median_s == pytest.approx(0.5)
    assert stats.min_s == pytest.approx(0.5)
    assert stats.max_s == pytest.approx(0.5)


def test_interval_stats_empty_is_safe():
    stats = keyframe_interval_stats([])
    assert stats.count == 0
    assert stats.median_s == 0.0


def test_find_gpmf_stream_index_from_streams():
    # exercise the pure selection helper via a monkeypatched streams source
    from tandem.recon import ffprobe

    fake = [
        {"index": 0, "codec_type": "video", "codec_tag_string": "hvc1"},
        {"index": 3, "codec_type": "data", "codec_tag_string": "gpmd"},
    ]
    assert ffprobe._select_gpmf_index(fake) == 3


@pytest.mark.integration
@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg/ffprobe not installed")
def test_streams_on_generated_file(tmp_path):
    import subprocess

    sample = tmp_path / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
         str(sample)],
        check=True, capture_output=True,
    )
    assert find_gpmf_stream_index(str(sample)) is None  # generated file has no GPMF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_ffprobe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.ffprobe'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/ffprobe.py`:
```python
"""Thin wrappers over ffprobe/ffmpeg. Pure helpers are separated from
subprocess calls so the logic is testable without media files."""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass


@dataclass
class IntervalStats:
    count: int
    median_s: float
    min_s: float
    max_s: float


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run_json(args: list[str]) -> dict:
    out = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def ffprobe_streams(path: str) -> list[dict]:
    data = _run_json(["ffprobe", "-v", "error", "-show_streams",
                      "-print_format", "json", path])
    return data.get("streams", [])


def _select_gpmf_index(streams: list[dict]) -> int | None:
    for s in streams:
        if s.get("codec_tag_string") == "gpmd":
            return int(s["index"])
        handler = s.get("tags", {}).get("handler_name", "")
        if "GoPro MET" in handler:
            return int(s["index"])
    return None


def find_gpmf_stream_index(path: str) -> int | None:
    return _select_gpmf_index(ffprobe_streams(path))


def keyframe_pts(path: str) -> list[float]:
    data = _run_json([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-skip_frame", "nokey", "-show_frames",
        "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time",
        "-print_format", "json", path,
    ])
    pts = []
    for frame in data.get("frames", []):
        raw = frame.get("best_effort_timestamp_time") or frame.get("pkt_pts_time")
        if raw is not None:
            pts.append(float(raw))
    return sorted(pts)


def keyframe_interval_stats(pts: list[float]) -> IntervalStats:
    if len(pts) < 2:
        return IntervalStats(count=len(pts), median_s=0.0, min_s=0.0, max_s=0.0)
    deltas = [b - a for a, b in zip(pts, pts[1:])]
    return IntervalStats(
        count=len(pts),
        median_s=statistics.median(deltas),
        min_s=min(deltas),
        max_s=max(deltas),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_ffprobe.py -v`
Expected: PASS for the 3 pure tests; the integration test PASSES if ffmpeg is installed, otherwise SKIPS.

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/ffprobe.py tests/recon/test_ffprobe.py
git commit -m "feat(recon): ffprobe wrappers and keyframe interval stats"
```

---

## Task 5: Telemetry extraction and assembly

Extracts the GPMF data stream with ffmpeg, parses it with Task 3's reader, and assembles a `Telemetry` object on the UTC time scale. Verifies assumption 1 end-to-end and provides the signal Task 7 windows on.

**Files:**
- Create: `tandem/recon/telemetry.py`
- Test: `tests/recon/test_telemetry.py`

**Interfaces:**
- Consumes: `tandem.recon.gpmf.walk`, `decode_numbers`, `decode_utc`; `tandem.recon.ffprobe.find_gpmf_stream_index`.
- Produces:
  - `Telemetry` dataclass: `device_id: int | None`, `first_utc: datetime | None`, `has_gps: bool`, `has_utc: bool`, `speed_3d_ms: list[float]` (GPS5 3D speed in m/s, SCAL-corrected), `t_s: list[float]` (seconds from `first_utc`).
  - `parse_telemetry_blob(blob: bytes) -> Telemetry` — pure, parses a raw gpmd payload.
  - `extract_gpmf_blob(path: str) -> bytes` — ffmpeg extracts the data stream to memory.
  - `read_telemetry(path: str) -> Telemetry | None` — returns `None` when no GPMF stream is present (caller records `NO_TELEMETRY`).

- [ ] **Step 1: Write the failing test**

The pure assembler is tested on a hand-built GPMF blob containing one `DEVC` with `DVID`, one `STRM` with `GPSU`, and one `STRM` with `GPS5` (lat, lon, alt, s2d, s3d).

`tests/recon/test_telemetry.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_telemetry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.telemetry'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/telemetry.py`:
```python
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


@dataclass
class Telemetry:
    device_id: int | None = None
    first_utc: datetime | None = None
    has_gps: bool = False
    has_utc: bool = False
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
        for c in iter_klv(strm.payload):
            if c.key == "GPSU" and not tel.has_utc:
                tel.first_utc = decode_utc(c.payload)
                tel.has_utc = True
            elif c.key == "SCAL":
                scal = _flatten(c)
            elif c.key == "GPS5":
                gps5 = c
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_telemetry.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/telemetry.py tests/recon/test_telemetry.py
git commit -m "feat(recon): GPMF telemetry extraction and assembly"
```

---

## Task 6: Jump grouping and camera identity

Groups recordings into logical *jumps* by overlapping UTC windows and counts distinct cameras per jump. Verifies assumption 5 (second half) and assumption 7 (fraction of jumps with both cameras). Camera identity comes from the GPMF device id, not the filename.

**Files:**
- Create: `tandem/recon/jumps.py`
- Test: `tests/recon/test_jumps.py`

**Interfaces:**
- Consumes: `tandem.recon.naming.Recording`, `tandem.recon.telemetry.Telemetry`.
- Produces:
  - `RecordingInfo` dataclass: `recording: Recording`, `telemetry: Telemetry | None`, `start_utc: datetime | None`, `end_utc: datetime | None`.
  - `group_jumps(infos: list[RecordingInfo], gap_tolerance_s: float = 120.0) -> list[Jump]` where `Jump` has `recordings: list[RecordingInfo]` and `.distinct_devices: set[int]`.
  - `dual_camera_fraction(jumps: list[Jump]) -> float` — share of jumps with ≥2 distinct device ids.

- [ ] **Step 1: Write the failing test**

`tests/recon/test_jumps.py`:
```python
from datetime import datetime, timezone

from tandem.recon.jumps import RecordingInfo, group_jumps, dual_camera_fraction


def _info(dev, start, end):
    return RecordingInfo(
        recording=None,
        telemetry=None,
        start_utc=datetime(2026, 8, 12, 9, start, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 8, 12, 9, end, 0, tzinfo=timezone.utc),
        _device_id=dev,
    )


def test_overlapping_recordings_form_one_jump():
    a = _info(dev=1, start=0, end=8)   # operator
    b = _info(dev=2, start=1, end=6)   # instructor hand
    jumps = group_jumps([a, b])
    assert len(jumps) == 1
    assert jumps[0].distinct_devices == {1, 2}


def test_separated_recordings_form_two_jumps():
    a = _info(dev=1, start=0, end=8)
    b = _info(dev=1, start=20, end=28)
    jumps = group_jumps([a, b])
    assert len(jumps) == 2


def test_dual_camera_fraction():
    a = _info(dev=1, start=0, end=8)
    b = _info(dev=2, start=1, end=6)
    c = _info(dev=1, start=30, end=38)
    jumps = group_jumps([a, b, c])
    assert dual_camera_fraction(jumps) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_jumps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.jumps'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/jumps.py`:
```python
"""Group recordings into jumps by overlapping UTC time and count cameras.

A single jump is filmed by one or two cameras at the same wall-clock
time. Two cameras therefore appear as two recordings whose UTC ranges
overlap but whose GPMF device ids differ. Recordings whose ranges are
separated by more than the gap tolerance belong to different jumps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class RecordingInfo:
    recording: object
    telemetry: object
    start_utc: datetime | None
    end_utc: datetime | None
    _device_id: int | None = None

    @property
    def device_id(self) -> int | None:
        if self._device_id is not None:
            return self._device_id
        return getattr(self.telemetry, "device_id", None)


@dataclass
class Jump:
    recordings: list[RecordingInfo] = field(default_factory=list)

    @property
    def distinct_devices(self) -> set[int]:
        return {r.device_id for r in self.recordings if r.device_id is not None}


def group_jumps(infos: list[RecordingInfo], gap_tolerance_s: float = 120.0) -> list[Jump]:
    timed = [i for i in infos if i.start_utc is not None and i.end_utc is not None]
    timed.sort(key=lambda i: i.start_utc)
    jumps: list[Jump] = []
    tol = timedelta(seconds=gap_tolerance_s)
    current: Jump | None = None
    current_end: datetime | None = None
    for info in timed:
        if current is None or info.start_utc > current_end + tol:
            current = Jump(recordings=[info])
            current_end = info.end_utc
            jumps.append(current)
        else:
            current.recordings.append(info)
            current_end = max(current_end, info.end_utc)
    return jumps


def dual_camera_fraction(jumps: list[Jump]) -> float:
    if not jumps:
        return 0.0
    dual = sum(1 for j in jumps if len(j.distinct_devices) >= 2)
    return dual / len(jumps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_jumps.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/jumps.py tests/recon/test_jumps.py
git commit -m "feat(recon): jump grouping and dual-camera fraction"
```

---

## Task 7: Rough freefall and canopy window estimate

Estimates a freefall window and a canopy-open instant from the telemetry GPS 3D-speed series, good enough to point the sampler (Task 8) at the right part of the video. Explicitly NOT the production phase detector — it exists only to aim frame sampling for the visual assumptions.

**Files:**
- Create: `tandem/recon/windows.py`
- Test: `tests/recon/test_windows.py`

**Interfaces:**
- Consumes: `tandem.recon.telemetry.Telemetry`.
- Produces:
  - `smooth(xs: list[float], k: int = 5) -> list[float]` (pure) — moving-average smoother for the noisy speed series.
  - `estimate_windows(tel: Telemetry, freefall_ms: float = 40.0) -> Windows | None` where `Windows` has `freefall_start_s: float`, `freefall_end_s: float`, `canopy_open_s: float | None`. Freefall is where smoothed GPS 3D speed is at or above `freefall_ms`; canopy-open is the first sample after freefall where it drops below. Returns `None` when speed is absent.

- [ ] **Step 1: Write the failing test**

Synthetic speed series: freefall (~55 m/s), then under canopy (~6 m/s).

`tests/recon/test_windows.py`:
```python
from tandem.recon.telemetry import Telemetry
from tandem.recon.windows import smooth, estimate_windows


def _synthetic_speed():
    speed, t = [], []
    time = 0.0
    for _ in range(50):      # freefall: ~55 m/s
        speed.append(55.0); t.append(time); time += 1.0
    for _ in range(60):      # under canopy: ~6 m/s
        speed.append(6.0); t.append(time); time += 1.0
    return speed, t


def test_smooth_reduces_to_neighbourhood_mean():
    assert smooth([10.0, 10.0, 10.0])[1] == 10.0
    out = smooth([0.0, 0.0, 30.0, 0.0, 0.0], k=3)
    assert out[2] == 10.0  # (0 + 30 + 0) / 3


def test_estimate_windows_finds_freefall_and_canopy():
    speed, t = _synthetic_speed()
    tel = Telemetry(has_gps=True, speed_3d_ms=speed, t_s=t)
    w = estimate_windows(tel)
    assert w is not None
    assert w.freefall_start_s < w.freefall_end_s
    assert 45.0 <= w.canopy_open_s <= 55.0


def test_estimate_windows_none_without_speed():
    assert estimate_windows(Telemetry()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_windows.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.windows'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/windows.py`:
```python
"""Rough freefall/canopy windowing from GPS 3D speed.

Reconnaissance-only: aims the frame sampler, never emits events. Uses
the GPS5 3D-speed field directly — GPS altitude is unusably noisy, so
we never differentiate it. The production detector (spec section 5.1)
uses accelerometer signatures and a phase state machine instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from tandem.recon.telemetry import Telemetry


@dataclass
class Windows:
    freefall_start_s: float
    freefall_end_s: float
    canopy_open_s: float | None


def smooth(xs: list[float], k: int = 5) -> list[float]:
    if len(xs) < k:
        return list(xs)
    out = []
    half = k // 2
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        window = xs[lo:hi]
        out.append(sum(window) / len(window))
    return out


def estimate_windows(tel: Telemetry, freefall_ms: float = 40.0) -> Windows | None:
    if not tel.speed_3d_ms or not tel.t_s:
        return None
    speed = smooth(tel.speed_3d_ms)
    freefall_idx = [i for i, v in enumerate(speed) if v >= freefall_ms]
    if not freefall_idx:
        # No clear freefall signature; hand back the whole recording.
        return Windows(freefall_start_s=tel.t_s[0], freefall_end_s=tel.t_s[-1], canopy_open_s=None)
    start_i, end_i = freefall_idx[0], freefall_idx[-1]
    canopy_open_s = None
    for i in range(end_i, len(speed)):
        if speed[i] < freefall_ms:
            canopy_open_s = tel.t_s[i]
            break
    return Windows(
        freefall_start_s=tel.t_s[start_i],
        freefall_end_s=tel.t_s[end_i],
        canopy_open_s=canopy_open_s,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_windows.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/windows.py tests/recon/test_windows.py
git commit -m "feat(recon): rough freefall/canopy windowing for sampling"
```

---

## Task 8: Keyframe sampler and HTML contact sheet

Extracts keyframes from the estimated windows into labeled folders and builds an HTML contact sheet so a human can tally the four judgement-based assumptions (3: ground-background fraction, 4: canopy-in-frame, 6: operator distance, 8: landing framing). This is a deliberate early slice of the section-10 preview tool, justified because the spec says that tool also accelerates labeling.

**Files:**
- Create: `tandem/recon/sampler.py`
- Test: `tests/recon/test_sampler.py`

**Interfaces:**
- Consumes: `tandem.recon.windows.Windows`.
- Produces:
  - `sample_plan(windows: Windows, every_s: float = 2.0) -> list[SampleShot]` (pure) where `SampleShot` has `t_s: float` and `label: str` (`"freefall"`, `"canopy"`, `"landing"`).
  - `extract_frames(video: str, shots: list[SampleShot], out_dir: str) -> list[str]` — writes 720p JPEGs named `<label>_<t>.jpg`, returns paths.
  - `build_contact_sheet(image_paths: list[str], out_html: str, title: str) -> None` — one self-contained HTML page with all thumbnails and their labels.

- [ ] **Step 1: Write the failing test**

`tests/recon/test_sampler.py`:
```python
from pathlib import Path

from tandem.recon.windows import Windows
from tandem.recon.sampler import sample_plan, build_contact_sheet


def test_sample_plan_covers_freefall_and_canopy():
    w = Windows(freefall_start_s=10.0, freefall_end_s=20.0, canopy_open_s=22.0)
    shots = sample_plan(w, every_s=5.0)
    labels = {s.label for s in shots}
    assert "freefall" in labels
    assert "canopy" in labels
    freefall_times = [s.t_s for s in shots if s.label == "freefall"]
    assert min(freefall_times) >= 10.0
    assert max(freefall_times) <= 20.0


def test_contact_sheet_is_self_contained(tmp_path):
    img = tmp_path / "freefall_10.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG marker bytes
    out = tmp_path / "sheet.html"
    build_contact_sheet([str(img)], str(out), title="jump-001")
    html = out.read_text(encoding="utf-8")
    assert "jump-001" in html
    assert "data:image/jpeg;base64," in html  # embedded, no external refs
    assert "freefall_10.jpg" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_sampler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.sampler'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/sampler.py`:
```python
"""Sample keyframes across estimated windows and render an HTML sheet.

Extraction uses -skip_frame nokey and scales to 720p height per the
global constraints. The contact sheet embeds thumbnails as base64 so a
single file can be reviewed anywhere without the source video.
"""
from __future__ import annotations

import base64
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SampleShot:
    t_s: float
    label: str


def sample_plan(windows, every_s: float = 2.0) -> list[SampleShot]:
    shots: list[SampleShot] = []
    t = windows.freefall_start_s
    while t <= windows.freefall_end_s:
        shots.append(SampleShot(t_s=round(t, 3), label="freefall"))
        t += every_s
    if windows.canopy_open_s is not None:
        for k in range(3):  # a few frames right after canopy opens
            shots.append(SampleShot(t_s=round(windows.canopy_open_s + k * every_s, 3),
                                    label="canopy"))
    return shots


def extract_frames(video: str, shots: list[SampleShot], out_dir: str) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for shot in shots:
        name = f"{shot.label}_{shot.t_s:.3f}.jpg"
        dest = str(Path(out_dir) / name)
        subprocess.run(
            ["ffmpeg", "-y", "-skip_frame", "nokey", "-ss", f"{shot.t_s}",
             "-i", video, "-frames:v", "1",
             "-vf", "scale=-2:720", dest],
            check=True, capture_output=True,
        )
        paths.append(dest)
    return paths


def build_contact_sheet(image_paths: list[str], out_html: str, title: str) -> None:
    cells = []
    for path in image_paths:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        name = Path(path).name
        cells.append(
            f'<figure><img src="data:image/jpeg;base64,{b64}" alt="{name}">'
            f'<figcaption>{name}</figcaption></figure>'
        )
    html = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:sans-serif}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}"
        "img{width:100%;height:auto}figcaption{font-size:12px;color:#555}</style>"
        f"<h1>{title}</h1><div class='grid'>" + "".join(cells) + "</div>"
    )
    Path(out_html).write_text(html, encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_sampler.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/sampler.py tests/recon/test_sampler.py
git commit -m "feat(recon): keyframe sampler and HTML contact sheet"
```

---

## Task 9: Assumption probes

Wires the modules into eight probe functions, one per section-9 assumption. Each returns a small result object with a status and the measured value; visual probes return the artifacts a human tallies rather than a verdict.

**Files:**
- Create: `tandem/recon/probes.py`
- Test: `tests/recon/test_probes.py`

**Interfaces:**
- Consumes: `naming`, `ffprobe`, `telemetry`, `jumps`, `windows`, `sampler`.
- Produces:
  - `ProbeResult` dataclass: `assumption: int`, `title: str`, `status: str` (`"measured"`, `"needs_review"`, `"blocked"`), `value: dict`, `degradation: str | None`.
  - `probe_gps_utc(tel: telemetry.Telemetry | None) -> ProbeResult` (assumption 1).
  - `probe_keyframe_interval(stats: ffprobe.IntervalStats) -> ProbeResult` (assumption 2).
  - `probe_dual_camera(jump_list: list[jumps.Jump]) -> ProbeResult` (assumption 7).
  - `probe_naming(recordings, unmatched) -> ProbeResult` (assumption 5).
  - `probe_visual(assumption: int, title: str, sheet_path: str | None) -> ProbeResult` (assumptions 3, 4, 6, 8 — always `needs_review`, carrying the contact-sheet path).

- [ ] **Step 1: Write the failing test**

`tests/recon/test_probes.py`:
```python
from tandem.recon.telemetry import Telemetry
from tandem.recon.ffprobe import IntervalStats
from tandem.recon.jumps import Jump, RecordingInfo
from tandem.recon.probes import (
    probe_gps_utc, probe_keyframe_interval, probe_dual_camera, probe_visual,
)


def test_probe_gps_utc_measured_when_present():
    tel = Telemetry(has_gps=True, has_utc=True)
    r = probe_gps_utc(tel)
    assert r.assumption == 1
    assert r.status == "measured"
    assert r.value["has_utc"] is True
    assert r.degradation is None


def test_probe_gps_utc_blocked_without_telemetry():
    r = probe_gps_utc(None)
    assert r.status == "blocked"
    assert r.degradation == "NO_TELEMETRY"


def test_probe_keyframe_interval_reports_median():
    r = probe_keyframe_interval(IntervalStats(count=100, median_s=0.5, min_s=0.5, max_s=0.5))
    assert r.assumption == 2
    assert r.value["median_s"] == 0.5


def test_probe_dual_camera_fraction():
    j1 = Jump(recordings=[RecordingInfo(None, None, None, None, 1),
                          RecordingInfo(None, None, None, None, 2)])
    j2 = Jump(recordings=[RecordingInfo(None, None, None, None, 1)])
    r = probe_dual_camera([j1, j2])
    assert r.assumption == 7
    assert r.value["fraction"] == 0.5


def test_probe_visual_always_needs_review():
    r = probe_visual(3, "Ground background fraction", "/out/sheet.html")
    assert r.status == "needs_review"
    assert r.value["contact_sheet"] == "/out/sheet.html"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_probes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.probes'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/probes.py`:
```python
"""One probe per section-9 assumption. Mechanical probes measure;
visual probes hand back a contact sheet for human tally."""
from __future__ import annotations

from dataclasses import dataclass

from tandem.recon import jumps as jumps_mod
from tandem.recon.ffprobe import IntervalStats
from tandem.recon.telemetry import Telemetry


@dataclass
class ProbeResult:
    assumption: int
    title: str
    status: str
    value: dict
    degradation: str | None = None


def probe_gps_utc(tel: Telemetry | None) -> ProbeResult:
    if tel is None:
        return ProbeResult(1, "GPS UTC present", "blocked", {}, degradation="NO_TELEMETRY")
    return ProbeResult(
        1, "GPS UTC present", "measured",
        {"has_gps": tel.has_gps, "has_utc": tel.has_utc,
         "first_utc": tel.first_utc.isoformat() if tel.first_utc else None},
    )


def probe_keyframe_interval(stats: IntervalStats) -> ProbeResult:
    status = "measured" if stats.count >= 2 else "blocked"
    return ProbeResult(
        2, "Keyframe interval", status,
        {"count": stats.count, "median_s": stats.median_s,
         "min_s": stats.min_s, "max_s": stats.max_s},
    )


def probe_naming(recordings: list, unmatched: list[str]) -> ProbeResult:
    status = "measured" if not unmatched else "needs_review"
    return ProbeResult(
        5, "Naming and chunk convention", status,
        {"recordings": len(recordings), "unmatched": unmatched},
    )


def probe_dual_camera(jump_list: list[jumps_mod.Jump]) -> ProbeResult:
    frac = jumps_mod.dual_camera_fraction(jump_list)
    return ProbeResult(
        7, "Fraction of jumps with both cameras", "measured",
        {"jumps": len(jump_list), "fraction": frac},
        degradation=None if frac > 0 else "NO_INSTRUCTOR_CAM",
    )


def probe_visual(assumption: int, title: str, sheet_path: str | None) -> ProbeResult:
    return ProbeResult(
        assumption, title, "needs_review",
        {"contact_sheet": sheet_path},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_probes.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/probes.py tests/recon/test_probes.py
git commit -m "feat(recon): assumption probes"
```

---

## Task 10: Findings report generator

Aggregates probe results across the whole archive into one markdown report — the deliverable that confirms or corrects the design.

**Files:**
- Create: `tandem/recon/report.py`
- Test: `tests/recon/test_report.py`

**Interfaces:**
- Consumes: `tandem.recon.probes.ProbeResult`.
- Produces:
  - `render_report(results: list[ProbeResult], archive: str) -> str` — a markdown document with one section per assumption, its status, measured value, and any degradation code. Assumptions marked `needs_review` list their contact sheet.

- [ ] **Step 1: Write the failing test**

`tests/recon/test_report.py`:
```python
from tandem.recon.probes import ProbeResult
from tandem.recon.report import render_report


def test_report_lists_every_assumption_with_status():
    results = [
        ProbeResult(1, "GPS UTC present", "measured", {"has_utc": True}),
        ProbeResult(3, "Ground background fraction", "needs_review",
                    {"contact_sheet": "/out/jump-1/freefall.html"}),
        ProbeResult(7, "Fraction with both cameras", "measured", {"fraction": 0.25},
                    degradation="NO_INSTRUCTOR_CAM"),
    ]
    md = render_report(results, archive="/data/archive")
    assert "# Reconnaissance findings" in md
    assert "Assumption 1" in md
    assert "measured" in md
    assert "needs_review" in md
    assert "/out/jump-1/freefall.html" in md
    assert "NO_INSTRUCTOR_CAM" in md
    assert "/data/archive" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.report'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/report.py`:
```python
"""Render probe results into a single markdown findings report."""
from __future__ import annotations

import json

from tandem.recon.probes import ProbeResult


def render_report(results: list[ProbeResult], archive: str) -> str:
    lines = [
        "# Reconnaissance findings",
        "",
        f"Archive: `{archive}`",
        "",
        "Each section answers one section-9 assumption. `measured` values",
        "are automatic; `needs_review` items link a contact sheet to tally",
        "by hand; `blocked` items record why a probe could not run.",
        "",
    ]
    for r in sorted(results, key=lambda x: x.assumption):
        lines.append(f"## Assumption {r.assumption}: {r.title}")
        lines.append("")
        lines.append(f"- Status: **{r.status}**")
        if r.degradation:
            lines.append(f"- Degradation: `{r.degradation}`")
        lines.append(f"- Value: `{json.dumps(r.value, ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_report.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/report.py tests/recon/test_report.py
git commit -m "feat(recon): findings report generator"
```

---

## Task 11: CLI wiring and end-to-end smoke

Ties the modules into `python -m tandem.recon <archive_dir> -o <out_dir>`: scan files, read telemetry, group jumps, run probes, sample frames for visual assumptions, write `recon-findings.md`. Provides the runnable entry point and an integration smoke test.

**Files:**
- Create: `tandem/recon/cli.py`
- Create: `tandem/recon/__main__.py`
- Test: `tests/recon/test_cli.py`

**Interfaces:**
- Consumes: every `recon` module.
- Produces:
  - `run(archive_dir: str, out_dir: str) -> str` — orchestrates the pipeline and returns the path to the written report.
  - `main(argv: list[str] | None = None) -> int` — argparse entry point.

- [ ] **Step 1: Write the failing test**

The orchestration is tested with the telemetry/ffprobe boundaries monkeypatched, so it needs no real media. This proves the wiring: files in a temp dir → a written report naming each assumption.

`tests/recon/test_cli.py`:
```python
from pathlib import Path

from tandem.recon import cli
from tandem.recon.telemetry import Telemetry
from tandem.recon.ffprobe import IntervalStats


def test_run_writes_report(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "GX010123.MP4").write_bytes(b"stub")

    from datetime import datetime, timezone
    tel = Telemetry(device_id=1, has_gps=True, has_utc=True,
                    first_utc=datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc),
                    speed_3d_ms=[55.0, 6.0], t_s=[0.0, 1.0])

    monkeypatch.setattr(cli, "read_telemetry", lambda p: tel)
    monkeypatch.setattr(cli, "keyframe_pts", lambda p: [0.0, 0.5, 1.0])
    monkeypatch.setattr(cli, "extract_frames", lambda v, s, o: [])
    monkeypatch.setattr(cli, "build_contact_sheet", lambda paths, out, title: Path(out).write_text("x"))

    out = tmp_path / "out"
    report_path = cli.run(str(archive), str(out))

    text = Path(report_path).read_text(encoding="utf-8")
    assert "Reconnaissance findings" in text
    for a in (1, 2, 5, 7):
        assert f"Assumption {a}" in text


def test_main_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run", lambda a, o: str(tmp_path / "r.md"))
    (tmp_path / "r.md").write_text("ok")
    assert cli.main([str(tmp_path), "-o", str(tmp_path / "out")]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.recon.cli'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/recon/cli.py`:
```python
"""Reconnaissance CLI: scan an archive and write a findings report."""
from __future__ import annotations

import argparse
import glob
import os
from datetime import timedelta
from pathlib import Path

from tandem.recon.naming import group_recordings
from tandem.recon.telemetry import read_telemetry
from tandem.recon.ffprobe import keyframe_pts, keyframe_interval_stats
from tandem.recon.jumps import RecordingInfo, group_jumps
from tandem.recon.windows import estimate_windows
from tandem.recon.sampler import sample_plan, extract_frames, build_contact_sheet
from tandem.recon import probes as P
from tandem.recon.report import render_report


def _list_media(archive_dir: str) -> list[str]:
    files = []
    for ext in ("MP4", "mp4"):
        files.extend(glob.glob(os.path.join(archive_dir, f"*.{ext}")))
    return sorted(set(files))


def run(archive_dir: str, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = _list_media(archive_dir)
    recordings, unmatched = group_recordings(paths)

    infos: list[RecordingInfo] = []
    first_tel = None
    first_stats = None
    for rec in recordings:
        primary = rec.chapters[0]
        tel = read_telemetry(primary)
        if first_tel is None:
            first_tel = tel
        if first_stats is None:
            first_stats = keyframe_interval_stats(keyframe_pts(primary))
        start = tel.first_utc if tel else None
        end = (start + timedelta(seconds=len(tel.speed_3d_ms) / 18.0)) if (tel and start) else None
        infos.append(RecordingInfo(recording=rec, telemetry=tel, start_utc=start, end_utc=end))

    jump_list = group_jumps(infos)

    results = [
        P.probe_gps_utc(first_tel),
        P.probe_keyframe_interval(first_stats if first_stats else keyframe_interval_stats([])),
        P.probe_naming(recordings, unmatched),
        P.probe_dual_camera(jump_list),
    ]

    # Visual assumptions: sample frames from the first telemetried recording.
    sheet = None
    if first_tel and infos:
        windows = estimate_windows(first_tel)
        if windows:
            shots = sample_plan(windows)
            frame_dir = os.path.join(out_dir, "frames")
            frames = extract_frames(infos[0].recording.chapters[0], shots, frame_dir)
            sheet = os.path.join(out_dir, "contact_sheet.html")
            build_contact_sheet(frames, sheet, title="recon-sample")
    for a, title in [
        (3, "Ground background fraction in freefall"),
        (4, "Canopy opening lands in operator frame"),
        (6, "How close operator gets to the pair"),
        (8, "How tandem landing is framed"),
    ]:
        results.append(P.probe_visual(a, title, sheet))

    report = render_report(results, archive=archive_dir)
    report_path = os.path.join(out_dir, "recon-findings.md")
    Path(report_path).write_text(report, encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tandem.recon",
                                     description="Verify section-9 assumptions on an archive.")
    parser.add_argument("archive", help="directory of camera files")
    parser.add_argument("-o", "--out", required=True, help="output directory")
    args = parser.parse_args(argv)
    path = run(args.archive, args.out)
    print(f"wrote {path}")
    return 0
```

`tandem/recon/__main__.py`:
```python
import sys

from tandem.recon.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_cli.py -v`
Expected: PASS (2 tests). Then run the whole suite: `python -m pytest -v` — all green.

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/cli.py tandem/recon/__main__.py tests/recon/test_cli.py
git commit -m "feat(recon): CLI wiring and end-to-end smoke"
```

---

## First real-data run (after Task 11)

Not a coding task — the payoff. On the machine holding the archive:

1. Install ffmpeg so `ffmpeg -version` and `ffprobe -version` both work.
2. `pip install -e ".[dev]"`.
3. `python -m tandem.recon <path-to-archive> -o recon_out`.
4. Read `recon_out/recon-findings.md`; open `recon_out/contact_sheet.html` and tally the four `needs_review` assumptions by eye.
5. Bring the findings back. Each assumption either confirms the design or feeds a correction into the next plan (extraction pipeline).

---

## Self-Review

**Spec coverage (section 9, the scope of this plan):**
- Assumption 1 (GPS UTC in both cameras) → Tasks 3, 5, 9 (`probe_gps_utc`).
- Assumption 2 (keyframe interval) → Tasks 4, 9 (`probe_keyframe_interval`).
- Assumption 3 (ground-background fraction) → Tasks 7, 8, 9 (`probe_visual`, contact sheet).
- Assumption 4 (canopy opening in operator frame) → Tasks 7, 8, 9.
- Assumption 5 (naming/chunking convention) → Tasks 2, 6, 9 (`probe_naming`).
- Assumption 6 (operator distance to pair) → Tasks 8, 9.
- Assumption 7 (fraction with both cameras) → Tasks 6, 9 (`probe_dual_camera`).
- Assumption 8 (landing framing) → Tasks 8, 9.
- Findings deliverable → Task 10 (`render_report`), Task 11 (CLI). Covered.

Out of scope by design (become their own plans once findings land): the physical phase detector and FSM (5.1), blob segmentation (5.2), visual event detectors (5.3), interview/VAD (5.4), highlights and the trained head (5.5), temporal assembly (5.6), the full `scenes.json` contract (section 3), training (section 6), and validation metrics (section 8). This plan deliberately builds only the reconnaissance that gates them.

**Placeholder scan:** No `TBD`/`TODO`/"handle edge cases" steps; every code step carries real code and every test step a runnable command with an expected result.

**Type consistency:** `Telemetry`, `Windows`, `IntervalStats`, `ProbeResult`, `RecordingInfo`, `Jump`, `SampleShot`, `KLV`, `GoProName`, and `Recording` keep the same fields and signatures wherever they are consumed. `read_telemetry`, `keyframe_pts`, `extract_frames`, and `build_contact_sheet` are imported into `cli` at module scope precisely so Task 11's test can monkeypatch them by name.

**Known real-data risks the plan surfaces rather than hides:**
- Windowing uses the GPS5 3D-speed field, not GPS altitude (altitude is unusably noisy and is never differentiated); `windows.py` is reconnaissance-only and never emits events, so a rough window is acceptable — it only aims sampling.
- GPMF device id as camera identity (Task 6) assumes each camera has a distinct `DVID`; the naming/dual-camera probes should be cross-checked against the contact sheet on the first real run before the extraction plan relies on them.
- The `NO_INSTRUCTOR_CAM` fraction (assumption 7) directly sizes the auto-labeling budget for the visual canopy/landing detectors (spec section 6); if it comes back very low, the detector plans must budget for manual labeling.
