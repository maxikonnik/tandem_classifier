# GoPro accelerometer player — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web tool that plots a GoPro clip's accelerometer (from GPMF) on a clickable canvas timeline under a video player, where clicking the curve seeks the video and a playhead tracks playback.

**Architecture:** A stdlib `http.server` backend reuses the existing dependency-free GPMF parser (`tandem/recon/gpmf.py`, `tandem/recon/ffprobe.py`), adding `ACCL` decoding and per-packet timing so the curve lands on the video's own time base. The frontend is one vanilla-JS page: an HTML5 `<video>` plus a hand-drawn `<canvas>` scrubber. HEVC clips that a browser cannot decode are transcoded to an H.264 proxy on demand.

**Tech Stack:** Python 3.10+ standard library (`http.server`, `subprocess`, `argparse`), the existing `tandem.recon` helpers, ffmpeg/ffprobe on PATH, vanilla JavaScript + Canvas 2D.

## Global Constraints

- Python `requires-python = ">=3.10"`; use `from __future__ import annotations` and `X | None` unions, matching the codebase.
- No new Python dependencies. Backend is standard library only.
- No JavaScript frameworks or chart libraries. Frontend is vanilla JS + Canvas.
- Pure logic is separated from subprocess/HTTP so it is unit-testable without media, matching `tandem/recon/ffprobe.py`.
- ffmpeg/ffprobe are invoked exactly as the project already does (`subprocess.run(..., check=True, capture_output=True)`); tests that need them use the `integration` pytest marker.
- Tests live under `tests/webtool/` mirroring `tests/recon/`, using struct-built byte fixtures like `tests/recon/test_gpmf.py`.

---

## File structure

- Create `tandem/webtool/__init__.py` — package marker.
- Create `tandem/webtool/accel.py` — `parse_accel(blob, packet_times, video_duration)` → `AccelSeries`. Pure; consumes `gpmf.py`.
- Create `tandem/webtool/fsbrowse.py` — filesystem listing + path-safety helpers for the Обзор modal.
- Create `tandem/webtool/proxy.py` — `ensure_h264_proxy(src)` and pure `proxy_path(src)`.
- Create `tandem/webtool/server.py` — request handler: routing, Range video streaming, JSON endpoints. Pure `parse_range` helper.
- Create `tandem/webtool/__main__.py` — CLI entry (`python -m tandem.webtool`), argparse, launches the server.
- Create `tandem/webtool/static/index.html`, `static/app.js`, `static/style.css` — the frontend.
- Modify `tandem/recon/ffprobe.py` — add `gpmd_packet_times(path, index)`, `probe_duration(path)`, and pure `_parse_packet_times(data)`.
- Create tests `tests/webtool/__init__.py`, `test_accel.py`, `test_fsbrowse.py`, `test_proxy.py`, `test_server.py`, and add cases to `tests/recon/test_ffprobe.py`.

---

## Task 1: ffprobe packet-time and duration helpers

**Files:**
- Modify: `tandem/recon/ffprobe.py`
- Test: `tests/recon/test_ffprobe.py`

**Interfaces:**
- Consumes: existing `_run_json(args)` in the same module.
- Produces:
  - `_parse_packet_times(data: dict) -> list[tuple[float, float]]` — pure; extracts `(pts_time, duration_time)` pairs from an ffprobe `-show_packets` JSON dict, skipping entries missing either field.
  - `gpmd_packet_times(path: str, stream_index: int) -> list[tuple[float, float]]` — runs ffprobe on one stream, returns the parsed pairs.
  - `probe_duration(path: str) -> float | None` — video container duration in seconds, or `None`.

- [ ] **Step 1: Write the failing test for the pure parser**

Add to `tests/recon/test_ffprobe.py`:

```python
from tandem.recon.ffprobe import _parse_packet_times


def test_parse_packet_times_extracts_pairs_and_skips_incomplete():
    data = {
        "packets": [
            {"pts_time": "0.000000", "duration_time": "1.001000"},
            {"pts_time": "1.001000", "duration_time": "1.001000"},
            {"pts_time": "2.002000"},  # missing duration -> skipped
        ]
    }
    assert _parse_packet_times(data) == [(0.0, 1.001), (1.001, 1.001)]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/recon/test_ffprobe.py::test_parse_packet_times_extracts_pairs_and_skips_incomplete -v`
Expected: FAIL with `ImportError` / `cannot import name '_parse_packet_times'`.

- [ ] **Step 3: Implement the helpers**

Append to `tandem/recon/ffprobe.py`:

```python
def _parse_packet_times(data: dict) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for pkt in data.get("packets", []):
        pts = pkt.get("pts_time")
        dur = pkt.get("duration_time")
        if pts is not None and dur is not None:
            pairs.append((float(pts), float(dur)))
    return pairs


def gpmd_packet_times(path: str, stream_index: int) -> list[tuple[float, float]]:
    data = _run_json([
        "ffprobe", "-v", "error", "-select_streams", str(stream_index),
        "-show_packets", "-show_entries", "packet=pts_time,duration_time",
        "-print_format", "json", path,
    ])
    return _parse_packet_times(data)


def probe_duration(path: str) -> float | None:
    data = _run_json([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-print_format", "json", path,
    ])
    raw = data.get("format", {}).get("duration")
    return float(raw) if raw is not None else None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/recon/test_ffprobe.py::test_parse_packet_times_extracts_pairs_and_skips_incomplete -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/ffprobe.py tests/recon/test_ffprobe.py
git commit -m "feat(ffprobe): gpmd packet times and container duration helpers"
```

---

## Task 2: accelerometer parsing (`accel.py`)

**Files:**
- Create: `tandem/webtool/__init__.py` (empty)
- Create: `tandem/webtool/accel.py`
- Create: `tests/webtool/__init__.py` (empty)
- Create: `tests/webtool/test_accel.py`

**Interfaces:**
- Consumes: `tandem.recon.gpmf.iter_klv`, `tandem.recon.gpmf.decode_numbers`.
- Produces:
  - `AccelSeries` dataclass: `t: list[float]`, `ax: list[float]`, `ay: list[float]`, `az: list[float]`, `amag: list[float]`, `warnings: list[str]`.
  - `parse_accel(blob: bytes, packet_times: list[tuple[float, float]], video_duration: float | None = None) -> AccelSeries`.

- [ ] **Step 1: Write the failing tests**

Create `tests/webtool/__init__.py` (empty) and `tests/webtool/test_accel.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/webtool/test_accel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tandem.webtool'`.

- [ ] **Step 3: Implement `accel.py`**

Create `tandem/webtool/__init__.py` (empty file) and `tandem/webtool/accel.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/webtool/test_accel.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tandem/webtool/__init__.py tandem/webtool/accel.py tests/webtool/__init__.py tests/webtool/test_accel.py
git commit -m "feat(webtool): decode ACCL onto the video time base"
```

---

## Task 3: filesystem browser (`fsbrowse.py`)

**Files:**
- Create: `tandem/webtool/fsbrowse.py`
- Create: `tests/webtool/test_fsbrowse.py`

**Interfaces:**
- Produces:
  - `DirEntry` dataclass: `name: str`, `path: str`, `is_dir: bool`.
  - `list_dir(path: str) -> list[DirEntry]` — subdirectories then `.mp4`/`.MP4` files, each sorted case-insensitively.
  - `resolve_within(path: str, roots: list[str]) -> str | None` — absolute, normalized path if it sits inside one of `roots` (after the same normalization), else `None`.
  - `drive_roots() -> list[str]` — filesystem roots (Windows drive letters that exist; `["/"]` elsewhere).

- [ ] **Step 1: Write the failing tests**

Create `tests/webtool/test_fsbrowse.py`:

```python
import os

from tandem.webtool.fsbrowse import DirEntry, list_dir, resolve_within


def test_list_dir_returns_dirs_then_mp4_sorted(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "Zclip.MP4").write_bytes(b"x")
    (tmp_path / "aclip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")  # ignored

    entries = list_dir(str(tmp_path))
    names = [(e.name, e.is_dir) for e in entries]
    assert names == [
        ("a_dir", True), ("b_dir", True),
        ("aclip.mp4", False), ("Zclip.MP4", False),
    ]
    assert all(isinstance(e, DirEntry) for e in entries)


def test_resolve_within_accepts_child_and_rejects_escape(tmp_path):
    root = str(tmp_path)
    child = os.path.join(root, "sub", "clip.mp4")
    assert resolve_within(child, [root]) == os.path.abspath(child)

    escape = os.path.join(root, "..", "secret.mp4")
    assert resolve_within(escape, [root]) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/webtool/test_fsbrowse.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name`.

- [ ] **Step 3: Implement `fsbrowse.py`**

Create `tandem/webtool/fsbrowse.py`:

```python
"""Server-side filesystem listing for the browse modal, with a
path-safety guard so requests cannot escape the active roots."""
from __future__ import annotations

import os
import string
from dataclasses import dataclass

VIDEO_EXTS = (".mp4",)


@dataclass
class DirEntry:
    name: str
    path: str
    is_dir: bool


def list_dir(path: str) -> list[DirEntry]:
    dirs: list[DirEntry] = []
    files: list[DirEntry] = []
    with os.scandir(path) as it:
        for entry in it:
            full = os.path.join(path, entry.name)
            if entry.is_dir():
                dirs.append(DirEntry(entry.name, full, True))
            elif os.path.splitext(entry.name)[1].lower() in VIDEO_EXTS:
                files.append(DirEntry(entry.name, full, False))
    dirs.sort(key=lambda e: e.name.lower())
    files.sort(key=lambda e: e.name.lower())
    return dirs + files


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def resolve_within(path: str, roots: list[str]) -> str | None:
    target = _norm(path)
    for root in roots:
        base = _norm(root)
        if target == base or target.startswith(base + os.sep):
            return os.path.abspath(path)
    return None


def drive_roots() -> list[str]:
    if os.name != "nt":
        return ["/"]
    roots = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            roots.append(drive)
    return roots
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/webtool/test_fsbrowse.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tandem/webtool/fsbrowse.py tests/webtool/test_fsbrowse.py
git commit -m "feat(webtool): server-side filesystem browser with path guard"
```

---

## Task 4: H.264 proxy (`proxy.py`)

**Files:**
- Create: `tandem/webtool/proxy.py`
- Create: `tests/webtool/test_proxy.py`

**Interfaces:**
- Produces:
  - `proxy_path(src: str) -> str` — pure; sibling path `.<stem>.proxy.mp4` next to `src`.
  - `ensure_h264_proxy(src: str, runner=subprocess.run) -> str` — returns an existing proxy or transcodes one with ffmpeg, then returns its path. `runner` is injected for testing.

- [ ] **Step 1: Write the failing tests**

Create `tests/webtool/test_proxy.py`:

```python
import os

from tandem.webtool.proxy import ensure_h264_proxy, proxy_path


def test_proxy_path_is_hidden_sibling():
    p = proxy_path(os.path.join("videos", "GX012253.MP4"))
    assert os.path.basename(p) == ".GX012253.proxy.mp4"
    assert os.path.dirname(p) == "videos"


def test_ensure_returns_existing_without_transcoding(tmp_path):
    src = tmp_path / "clip.MP4"
    src.write_bytes(b"x")
    existing = tmp_path / ".clip.proxy.mp4"
    existing.write_bytes(b"proxy")
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)

    result = ensure_h264_proxy(str(src), runner=runner)
    assert result == str(existing)
    assert calls == []  # no transcode when proxy already present


def test_ensure_transcodes_when_missing(tmp_path):
    src = tmp_path / "clip.MP4"
    src.write_bytes(b"x")
    called = {}

    def runner(args, **kwargs):
        called["args"] = args
        open(proxy_path(str(src)), "wb").write(b"proxy")

    result = ensure_h264_proxy(str(src), runner=runner)
    assert result == proxy_path(str(src))
    assert called["args"][0] == "ffmpeg"
    assert "libx264" in called["args"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/webtool/test_proxy.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `proxy.py`**

Create `tandem/webtool/proxy.py`:

```python
"""Lazily transcode an HEVC clip to a browser-friendly H.264 proxy.

Invoked only when the frontend reports the original will not decode. The
proxy is cached as a hidden sibling file so it is made once per clip.
"""
from __future__ import annotations

import os
import subprocess


def proxy_path(src: str) -> str:
    directory = os.path.dirname(src)
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(directory, f".{stem}.proxy.mp4")


def ensure_h264_proxy(src: str, runner=subprocess.run) -> str:
    out = proxy_path(src)
    if os.path.exists(out):
        return out
    runner(
        ["ffmpeg", "-y", "-i", src,
         "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", out],
        check=True,
    )
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/webtool/test_proxy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tandem/webtool/proxy.py tests/webtool/test_proxy.py
git commit -m "feat(webtool): lazy H.264 proxy for HEVC playback"
```

---

## Task 5: HTTP server (`server.py`)

**Files:**
- Create: `tandem/webtool/server.py`
- Create: `tests/webtool/test_server.py`

**Interfaces:**
- Consumes: `parse_accel`/`AccelSeries` (Task 2), `list_dir`/`resolve_within`/`drive_roots`/`DirEntry` (Task 3), `ensure_h264_proxy`/`proxy_path` (Task 4), `find_gpmf_stream_index`/`gpmd_packet_times`/`probe_duration` (Task 1), `tandem.recon.telemetry.extract_gpmf_blob`.
- Produces:
  - `parse_range(header: str | None, size: int) -> tuple[int, int] | None` — pure; returns inclusive `(start, end)` byte offsets for a `Range: bytes=` header, clamped to `size`, or `None` when absent/unparseable.
  - `make_handler(roots: list[str]) -> type` — a `BaseHTTPRequestHandler` subclass bound to the active roots.
  - `serve(roots: list[str], port: int) -> None` — run the server (used by `__main__`).

- [ ] **Step 1: Write the failing test for the pure range parser**

Create `tests/webtool/test_server.py`:

```python
from tandem.webtool.server import parse_range


def test_parse_range_none_when_absent():
    assert parse_range(None, 1000) is None


def test_parse_range_open_ended_clamps_to_size():
    assert parse_range("bytes=500-", 1000) == (500, 999)


def test_parse_range_explicit_end_is_inclusive():
    assert parse_range("bytes=0-99", 1000) == (0, 99)


def test_parse_range_end_beyond_size_is_clamped():
    assert parse_range("bytes=900-5000", 1000) == (900, 999)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/webtool/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name 'parse_range'`.

- [ ] **Step 3: Implement `server.py`**

Create `tandem/webtool/server.py`:

```python
"""Local HTTP server for the accelerometer player.

Transport only: it turns a file path into an accelerometer JSON series
(via the recon GPMF helpers and accel.py), streams video with Range
support, browses the filesystem for the picker, and triggers the H.264
proxy on demand. All path access is confined to the active roots.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from tandem.recon.ffprobe import (
    find_gpmf_stream_index,
    gpmd_packet_times,
    probe_duration,
)
from tandem.recon.telemetry import extract_gpmf_blob
from tandem.webtool.accel import parse_accel
from tandem.webtool.fsbrowse import drive_roots, list_dir, resolve_within
from tandem.webtool.proxy import ensure_h264_proxy

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_CHUNK = 1 << 20


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].split(",")[0].strip()
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":
            return None
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    if start > end:
        return None
    return start, end


def make_handler(roots: list[str]) -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _safe(self, path: str) -> str | None:
            allowed = roots + drive_roots()
            return resolve_within(path, allowed)

        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)
            if route == "/" or route == "/index.html":
                return self._static("index.html")
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route == "/api/roots":
                return self._json({"roots": roots, "drives": drive_roots()})
            if route == "/api/browse":
                return self._browse(query.get("path", [roots[0]])[0])
            if route == "/api/accel":
                return self._accel(query.get("path", [""])[0])
            if route == "/api/video":
                return self._video(query.get("path", [""])[0])
            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/api/proxy":
                return self._proxy(query.get("path", [""])[0])
            self.send_error(404)

        def _static(self, rel):
            full = os.path.join(STATIC_DIR, rel)
            if not os.path.isfile(full):
                return self.send_error(404)
            ctype = {
                ".html": "text/html", ".js": "text/javascript",
                ".css": "text/css",
            }.get(os.path.splitext(full)[1], "application/octet-stream")
            data = open(full, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _browse(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isdir(safe):
                return self._json({"error": "path not allowed"}, 403)
            entries = [
                {"name": e.name, "path": e.path, "is_dir": e.is_dir}
                for e in list_dir(safe)
            ]
            parent = os.path.dirname(safe.rstrip(os.sep)) or safe
            return self._json({"cwd": safe, "parent": parent, "entries": entries})

        def _accel(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isfile(safe):
                return self._json({"error": "path not allowed"}, 403)
            index = find_gpmf_stream_index(safe)
            if index is None:
                return self._json({"error": "no GPMF stream"}, 422)
            blob = extract_gpmf_blob(safe)
            times = gpmd_packet_times(safe, index)
            series = parse_accel(blob, times, probe_duration(safe))
            if not series.t:
                return self._json({"error": "no ACCL samples"}, 422)
            return self._json({
                "t": series.t, "ax": series.ax, "ay": series.ay,
                "az": series.az, "amag": series.amag,
                "warnings": series.warnings,
            })

        def _video(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isfile(safe):
                return self.send_error(403)
            size = os.path.getsize(safe)
            rng = parse_range(self.headers.get("Range"), size)
            with open(safe, "rb") as f:
                if rng is None:
                    self.send_response(200)
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", "video/mp4")
                    self.end_headers()
                    self._copy(f, 0, size - 1)
                else:
                    start, end = rng
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", "video/mp4")
                    self.end_headers()
                    self._copy(f, start, end)

        def _copy(self, f, start, end):
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

        def _proxy(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isfile(safe):
                return self._json({"error": "path not allowed"}, 403)
            try:
                out = ensure_h264_proxy(safe)
            except Exception as exc:  # surface transcode failure to the UI
                return self._json({"error": f"proxy failed: {exc}"}, 500)
            return self._json({"path": out})

    return Handler


def serve(roots: list[str], port: int) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(roots))
    httpd.serve_forever()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/webtool/test_server.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tandem/webtool/server.py tests/webtool/test_server.py
git commit -m "feat(webtool): HTTP server with Range streaming and JSON endpoints"
```

---

## Task 6: frontend (`static/`)

**Files:**
- Create: `tandem/webtool/static/index.html`
- Create: `tandem/webtool/static/style.css`
- Create: `tandem/webtool/static/app.js`

**Interfaces:**
- Consumes the server endpoints: `GET /api/roots`, `GET /api/browse?path=`, `GET /api/accel?path=`, `GET /api/video?path=`, `POST /api/proxy?path=`.
- No unit tests (DOM/canvas in this stack is verified by manual acceptance in Task 7). Keep logic in small named functions so it stays readable.

- [ ] **Step 1: Create `index.html`**

Create `tandem/webtool/static/index.html`:

```html
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GoPro accelerometer player</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div class="bar">
    <label for="filesel">Файл</label>
    <select id="filesel"></select>
    <button id="browse">Обзор</button>
  </div>

  <div id="stage">
    <video id="video" controls preload="metadata"></video>
    <div id="videomsg"></div>
  </div>

  <canvas id="scrub" height="240"></canvas>

  <div class="controls">
    <span id="tlabel" class="mono">00:00 / 00:00</span>
    <span class="spacer"></span>
    <div id="mode">
      <button data-m="mag" class="mbtn on">|a|</button>
      <button data-m="xyz" class="mbtn">X / Y / Z</button>
    </div>
  </div>
  <div id="warnings"></div>

  <div id="modal" class="hidden">
    <div class="dialog">
      <div class="dhead">
        <span id="crumb" class="mono"></span>
        <button id="usefolder">Выбрать эту папку</button>
        <button id="closem">✕</button>
      </div>
      <div id="fslist"></div>
    </div>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `style.css`**

Create `tandem/webtool/static/style.css`:

```css
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 16px; max-width: 900px; }
.bar { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.bar select { flex: 1; padding: 6px; }
#stage { position: relative; background: #000; border-radius: 8px;
  min-height: 240px; display: flex; align-items: center; justify-content: center; }
#video { width: 100%; max-height: 60vh; border-radius: 8px; }
#videomsg { position: absolute; color: #ccc; font-size: 13px; }
#scrub { width: 100%; height: 120px; margin-top: 10px; background: #f4f4f2;
  border: 1px solid #ddd; border-radius: 8px; cursor: pointer; display: block; }
.controls { display: flex; align-items: center; gap: 12px; margin-top: 10px; }
.spacer { flex: 1; }
.mono { font-family: ui-monospace, monospace; font-size: 13px; color: #555; }
.mbtn { padding: 6px 14px; border: 1px solid #bbb; background: #fff; cursor: pointer; }
.mbtn.on { background: #eee; }
#warnings { color: #a3521d; font-size: 13px; margin-top: 8px; }
.hidden { display: none !important; }
#modal { position: fixed; inset: 0; background: rgba(0,0,0,.45);
  display: flex; align-items: center; justify-content: center; }
.dialog { background: #fff; width: 480px; max-width: 92%; border-radius: 10px; overflow: hidden; }
.dhead { display: flex; align-items: center; gap: 8px; padding: 10px 12px;
  border-bottom: 1px solid #eee; }
.dhead #crumb { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#fslist { max-height: 320px; overflow-y: auto; padding: 6px; }
.row { display: flex; align-items: center; gap: 8px; padding: 7px 10px;
  border-radius: 6px; cursor: pointer; font-size: 14px; }
.row:hover { background: #f2f2f0; }
```

- [ ] **Step 3: Create `app.js`**

Create `tandem/webtool/static/app.js`:

```javascript
const video = document.getElementById("video");
const canvas = document.getElementById("scrub");
const ctx = canvas.getContext("2d");
const tlabel = document.getElementById("tlabel");
const warnings = document.getElementById("warnings");
const videomsg = document.getElementById("videomsg");

let data = null;      // {t, ax, ay, az, amag, warnings}
let mode = "mag";
let curPath = null;

function fmt(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

function duration() {
  if (video.duration && isFinite(video.duration)) return video.duration;
  return data && data.t.length ? data.t[data.t.length - 1] : 1;
}

function draw() {
  const w = canvas.width = canvas.clientWidth * 2;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!data || !data.t.length) return;
  const D = duration();
  const series = mode === "mag"
    ? [[data.amag, "#26215c", 2.4]]
    : [[data.ax, "#378ADD", 1.4], [data.ay, "#1D9E75", 1.4], [data.az, "#D85A30", 1.4]];
  let maxv = 1;
  for (const [arr] of series) for (const v of arr) maxv = Math.max(maxv, Math.abs(v));
  const pad = 8;
  const X = (t) => t / D * w;
  const Y = (v) => h - pad - (v / maxv) * (h - 2 * pad);
  // min/max envelope per pixel column preserves spikes
  for (const [arr, color, lw] of series) {
    ctx.beginPath();
    let px = -1, lo = Infinity, hi = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      const x = Math.floor(X(data.t[i]));
      if (x !== px && px >= 0) {
        ctx.moveTo(px, Y(lo)); ctx.lineTo(px, Y(hi));
        lo = Infinity; hi = -Infinity;
      }
      lo = Math.min(lo, arr[i]); hi = Math.max(hi, arr[i]); px = x;
    }
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
  }
  const cx = X(video.currentTime || 0);
  ctx.strokeStyle = "#E24B4A"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
}

function loop() {
  tlabel.textContent = `${fmt(video.currentTime)} / ${fmt(duration())}`;
  draw();
  requestAnimationFrame(loop);
}

canvas.addEventListener("click", (e) => {
  const r = canvas.getBoundingClientRect();
  video.currentTime = (e.clientX - r.left) / r.width * duration();
});

document.querySelectorAll(".mbtn").forEach((b) => {
  b.addEventListener("click", () => {
    mode = b.dataset.m;
    document.querySelectorAll(".mbtn").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    draw();
  });
});

async function openFile(path) {
  curPath = path;
  videomsg.textContent = "";
  video.src = `/api/video?path=${encodeURIComponent(path)}`;
  warnings.textContent = "";
  try {
    const res = await fetch(`/api/accel?path=${encodeURIComponent(path)}`);
    const body = await res.json();
    if (body.error) { warnings.textContent = body.error; data = null; return; }
    data = body;
    warnings.textContent = (body.warnings || []).join(" · ");
  } catch (err) {
    warnings.textContent = String(err);
  }
}

video.addEventListener("error", async () => {
  if (!curPath || video.src.includes("proxy")) return;
  videomsg.textContent = "готовлю совместимую копию…";
  const res = await fetch(`/api/proxy?path=${encodeURIComponent(curPath)}`, { method: "POST" });
  const body = await res.json();
  if (body.error) { videomsg.textContent = body.error; return; }
  videomsg.textContent = "";
  video.src = `/api/video?path=${encodeURIComponent(body.path)}&proxy=1`;
});

// --- file picker ---
const modal = document.getElementById("modal");
const fslist = document.getElementById("fslist");
const crumb = document.getElementById("crumb");
let browseCwd = null;

function makeRow(icon, label, onclick) {
  const d = document.createElement("div");
  d.className = "row";
  d.textContent = `${icon}  ${label}`;
  d.addEventListener("click", onclick);
  return d;
}

async function browse(path) {
  const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
  const body = await res.json();
  if (body.error) { crumb.textContent = body.error; return; }
  browseCwd = body.cwd;
  crumb.textContent = body.cwd;
  fslist.innerHTML = "";
  fslist.appendChild(makeRow("⬆", "..", () => browse(body.parent)));
  for (const e of body.entries) {
    if (e.is_dir) fslist.appendChild(makeRow("📁", e.name, () => browse(e.path)));
    else fslist.appendChild(makeRow("🎬", e.name, () => {
      modal.classList.add("hidden");
      addFileOption(e.path, e.name);
      openFile(e.path);
    }));
  }
}

function addFileOption(path, name) {
  const sel = document.getElementById("filesel");
  const opt = document.createElement("option");
  opt.value = path; opt.textContent = name; opt.selected = true;
  sel.insertBefore(opt, sel.firstChild);
}

async function populateList(root) {
  const res = await fetch(`/api/browse?path=${encodeURIComponent(root)}`);
  const body = await res.json();
  const sel = document.getElementById("filesel");
  sel.innerHTML = "";
  for (const e of (body.entries || []).filter((x) => !x.is_dir)) {
    const opt = document.createElement("option");
    opt.value = e.path; opt.textContent = e.name;
    sel.appendChild(opt);
  }
  if (sel.value) openFile(sel.value);
}

document.getElementById("filesel").addEventListener("change", (e) => openFile(e.target.value));
document.getElementById("browse").addEventListener("click", () => {
  modal.classList.remove("hidden"); browse(browseCwd || rootDir);
});
document.getElementById("closem").addEventListener("click", () => modal.classList.add("hidden"));
document.getElementById("usefolder").addEventListener("click", () => {
  modal.classList.add("hidden");
  if (browseCwd) { rootDir = browseCwd; populateList(rootDir); }
});

let rootDir = "";
(async function init() {
  const res = await fetch("/api/roots");
  const body = await res.json();
  rootDir = body.roots[0];
  await populateList(rootDir);
  requestAnimationFrame(loop);
})();
```

- [ ] **Step 4: Commit**

```bash
git add tandem/webtool/static/index.html tandem/webtool/static/style.css tandem/webtool/static/app.js
git commit -m "feat(webtool): player frontend with canvas scrubber and file picker"
```

---

## Task 7: CLI entry point and end-to-end acceptance

**Files:**
- Create: `tandem/webtool/__main__.py`

**Interfaces:**
- Consumes: `serve(roots, port)` (Task 5).
- Produces: `python -m tandem.webtool [--root DIR ...] [--port N] [--no-open]`.

- [ ] **Step 1: Implement `__main__.py`**

Create `tandem/webtool/__main__.py`:

```python
"""Launch the accelerometer player: python -m tandem.webtool."""
from __future__ import annotations

import argparse
import os
import threading
import webbrowser

from tandem.recon.ffprobe import has_ffmpeg
from tandem.webtool.server import serve


def main() -> None:
    ap = argparse.ArgumentParser(description="GoPro accelerometer player")
    ap.add_argument("--root", action="append", default=[],
                    help="root directory to browse (repeatable)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true",
                    help="do not open the browser automatically")
    args = ap.parse_args()

    if not has_ffmpeg():
        print("warning: ffmpeg/ffprobe not found on PATH; playback proxy "
              "and metadata will fail.")

    roots = [os.path.abspath(r) for r in args.root] or [os.path.abspath("Samples")]
    roots = [r for r in roots if os.path.isdir(r)] or [os.path.abspath(".")]
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {roots} at {url}")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    serve(roots, args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the whole test suite**

Run: `python -m pytest tests/webtool -v`
Expected: PASS (all tasks' unit tests green).

- [ ] **Step 3: Manual acceptance on a real jump**

Run:

```bash
python -m tandem.webtool --root "Samples"
```

Confirm in the browser:
- The file dropdown lists `.MP4` files from `Samples`; selecting one loads video + curve.
- The `|a|` curve shows a spike at the exit and a larger spike at the opening shock.
- Clicking a spike on the canvas seeks the video to that instant; the red playhead tracks playback.
- The `|a|` / `X / Y / Z` toggle switches the plot without a reload.
- If the HEVC clip shows a black frame, "готовлю совместимую копию…" appears and playback resumes on the proxy.
- Обзор opens the filesystem browser; navigating folders and picking a file loads it; "Выбрать эту папку" repopulates the list.

- [ ] **Step 4: Commit**

```bash
git add tandem/webtool/__main__.py
git commit -m "feat(webtool): CLI entry point and browser launch"
```

---

## Self-review

**Spec coverage:**
- Purpose / clickable timeline seek + playhead → Task 6 (`draw`, canvas click, `loop`).
- Reuse `gpmf.py`, add ACCL → Task 2.
- Per-packet `ffprobe` timing alignment → Task 1 + Task 2.
- Module boundaries `accel`/`fsbrowse`/`server`/`proxy` → Tasks 2–5.
- Data flow + JSON series → Task 5 `_accel`.
- Video Range streaming → Task 5 `_video`, `parse_range`.
- Lazy H.264 proxy on decode failure → Task 4 + Task 5 `_proxy` + Task 6 `video.onerror`.
- File list + Обзор + browser-constraint (no OS dialog / drag-drop) → Task 3 + Task 6 picker.
- Path-safety guard (403 outside roots) → Task 3 `resolve_within` + Task 5 `_safe`.
- `|a|` / X-Y-Z toggle, min/max downsample → Task 6 `draw`.
- Error surfacing in UI → Task 5 JSON errors + Task 6 `warnings`/`videomsg`.
- Testing plan (unit + integration marker) → Tasks 1–5; manual acceptance Task 7.
- No new deps / stdlib only → Global Constraints, Task 5/7.

**Placeholder scan:** none — every code and test step carries full content.

**Type consistency:** `AccelSeries` fields (`t/ax/ay/az/amag/warnings`) are consumed unchanged in Task 5 `_accel` and serialized to the same JSON keys consumed by Task 6 `draw`. `resolve_within(path, roots)`, `list_dir`, `drive_roots`, `proxy_path`, `ensure_h264_proxy(src, runner=...)`, `gpmd_packet_times(path, index)`, `probe_duration(path)`, `parse_range(header, size)` signatures match across their producing and consuming tasks.
