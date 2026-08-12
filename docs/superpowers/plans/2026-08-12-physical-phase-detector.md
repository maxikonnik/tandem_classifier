# Physical Phase Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a GoPro telemetry stream, detect the physical flight phases up to and including freefall (ground, climb, exit, freefall) with confidences and a phase-order sanity check, so the exit event and freefall boundaries can be auto-labelled without human effort.

**Architecture:** A new `tandem/phases/` subpackage. `signals.py` turns a raw GPMF blob into a common-grid signal set (accelerometer magnitude + GPS 3D speed on one time axis). `detect.py` finds each phase/event from those two signals using the spec §5.1 signatures. `fsm.py` validates the physically-mandatory phase order and raises a degradation when it is violated. `api.py` assembles phases + events with `source="telemetry"` and handles the no-telemetry case. Every module is pure logic, unit-tested on synthetic signals — no archive, ffmpeg, or GPU.

**Tech Stack:** Python 3.10+, numpy, pytest. Reuses `tandem/recon/gpmf.py` (KLV parser) and `tandem/recon/ffprobe.py` (stream location). No new third-party dependencies.

## Global Constraints

Copied from the spec and binding on every task.

- **Python floor: 3.10** (machine has 3.10.0); `from __future__ import annotations` in any module using `X | None`; no 3.11+ syntax.
- **No GPS altitude.** Descent/phase logic uses accelerometer magnitude and the GPS5 3D-speed field only. GPS altitude is unusably noisy and is never read or differentiated.
- **Operator telemetry is not tandem telemetry.** Only **exit** and the **freefall boundaries** transfer from the operator's telemetry to the tandem. The operator's canopy and landing happen on the operator's own flight profile and must never be emitted as tandem events. This detector therefore stops at freefall; it does not claim tandem canopy/landing.
- **`source` is part of the contract.** Every phase and event this detector emits carries `source="telemetry"`.
- **Times are recording-relative seconds** (seconds from the start of this telemetry stream). Mapping to `t0_utc` (seconds from the jump's first GPS-UTC) is done by a downstream assembly layer, not here.
- **Degradation is contract, not logs.** This plan emits `NO_TELEMETRY` (no usable accel/GPS signal) and `PHASE_ORDER_VIOLATION` (the phase-order FSM did not converge). Codes are strings, matching the spec's degradation table.
- **Thresholds are provisional.** The numeric thresholds below come from the spec §5.1 table and are calibration defaults. They are module-level named constants so the eventual real-telemetry calibration changes one place. Do not inline magic numbers.

## File Structure

```
tandem/phases/__init__.py
tandem/phases/signals.py    # GPMF blob -> Signals (accel_mag m/s^2 + speed_3d m/s on a common grid)
tandem/phases/detect.py     # exit event, freefall/ground/climb segments from Signals
tandem/phases/fsm.py        # phase-order validation -> ok | PHASE_ORDER_VIOLATION
tandem/phases/api.py        # assemble PhaseResult (phases + events + degradations); NO_TELEMETRY
tests/phases/__init__.py
tests/phases/test_signals.py
tests/phases/test_detect.py
tests/phases/test_fsm.py
tests/phases/test_api.py
```

Each module has one responsibility and is unit-testable in isolation. `signals.py` is the only module that touches the GPMF byte format (via the existing parser); `detect.py`/`fsm.py`/`api.py` work on plain numeric arrays and dataclasses.

## Design notes (read before Task 1)

**The two signals.** The accelerometer (`ACCL`, m/s², ~200 Hz) gives |a|; the GPS (`GPS5`, 3D speed m/s, ~18 Hz) gives descent/plane speed. They arrive at different rates, so `signals.py` resamples both onto one uniform grid (default 10 Hz) spanning the recording. Precise GPMF per-payload timing (via `TSMP`/payload cadence) is a later refinement; this plan makes the same pragmatic assumption `recon` already makes — samples of a stream are spread uniformly across the recording duration — and documents it. This is good enough for phase boundaries measured in seconds.

**Why exit is the pivot.** In the aircraft, |a| ≈ 1 g steadily. At exit, |a| collapses toward 0 (true free fall) for a few seconds, then returns toward 1 g as drag builds. So the deep |a| dropout that is *followed by sustained high 3D speed* is the exit. Everything before it is in-aircraft; everything after (until the operator's canopy, which we ignore) is freefall. Ground vs climb is then split purely by 3D speed: on the ground the aircraft is stationary (speed ≈ 0), during climb it is moving (speed elevated). No altitude required.

**Units.** `Signals.accel_mag` is m/s². Detection thresholds are expressed in g via `G = 9.80665`.

---

## Task 1: Signals extraction

Turn a GPMF blob into a `Signals` object: accelerometer magnitude and GPS 3D speed resampled onto one uniform time grid.

**Files:**
- Create: `tandem/phases/__init__.py` (empty)
- Create: `tandem/phases/signals.py`
- Create: `tests/phases/__init__.py` (empty)
- Create: `tests/phases/test_signals.py`

**Interfaces:**
- Consumes: `tandem.recon.gpmf.iter_klv`, `walk`, `decode_numbers`; `tandem.recon.ffprobe.find_gpmf_stream_index` (only in the thin I/O helper).
- Produces:
  - `Signals` dataclass: `t_s: list[float]`, `accel_mag: list[float]` (m/s²), `speed_3d: list[float]` (m/s), `fs: float` (grid Hz), `has_accel: bool`, `has_gps: bool`.
  - `resample(values: list[float], n_out: int) -> list[float]` (pure) — linear resample of a uniformly-sampled series to `n_out` points.
  - `build_signals(blob: bytes, fs: float = 10.0) -> Signals` — parse ACCL + GPS5 + their SCAL, compute |a| per ACCL sample, resample both streams to a common grid at `fs` over the recording duration.

- [ ] **Step 1: Write the failing test**

`tests/phases/test_signals.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.phases.signals'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/phases/signals.py`:
```python
"""Build an analysis-grade signal set from GPMF telemetry.

Two signals only: accelerometer magnitude (m/s^2) and GPS 3D speed
(m/s). GPS altitude is never used. Accel (~200 Hz) and GPS (~18 Hz)
are resampled onto one uniform grid (default 10 Hz). Per-stream timing
is assumed uniform across the recording — precise GPMF payload timing
is a later refinement; seconds-scale phase boundaries do not need it.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field

from tandem.recon.ffprobe import find_gpmf_stream_index
from tandem.recon.gpmf import decode_numbers, iter_klv, walk


@dataclass
class Signals:
    t_s: list[float] = field(default_factory=list)
    accel_mag: list[float] = field(default_factory=list)
    speed_3d: list[float] = field(default_factory=list)
    fs: float = 10.0
    has_accel: bool = False
    has_gps: bool = False


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


def _stream_children(blob: bytes):
    for _path, strm in walk(blob):
        if strm.key == "STRM":
            yield list(iter_klv(strm.payload))


def _accel_magnitudes(children) -> list[float] | None:
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
    mags = []
    for sample in decode_numbers(accl):
        mags.append(math.sqrt(sum((v / divisor) ** 2 for v in sample)))
    return mags


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
    return [sample[4] / divisor for sample in decode_numbers(gps5)]


def build_signals(blob: bytes, fs: float = 10.0) -> Signals:
    accel_raw = None
    speed_raw = None
    for children in _stream_children(blob):
        if accel_raw is None:
            accel_raw = _accel_magnitudes(children)
        if speed_raw is None:
            speed_raw = _gps_speeds(children)
    sig = Signals(fs=fs, has_accel=accel_raw is not None, has_gps=speed_raw is not None)
    # Recording duration: longer of the two streams at their nominal rates.
    accel_dur = (len(accel_raw) / 200.0) if accel_raw else 0.0
    gps_dur = (len(speed_raw) / 18.0) if speed_raw else 0.0
    duration = max(accel_dur, gps_dur)
    if duration <= 0:
        return sig
    n_out = max(2, int(round(duration * fs)))
    sig.t_s = [i / fs for i in range(n_out)]
    sig.accel_mag = resample(accel_raw, n_out) if accel_raw else []
    sig.speed_3d = resample(speed_raw, n_out) if speed_raw else []
    return sig


def build_signals_from_file(path: str, fs: float = 10.0) -> Signals | None:
    index = find_gpmf_stream_index(path)
    if index is None:
        return None
    out = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-map", f"0:{index}",
         "-codec", "copy", "-f", "data", "-"],
        check=True, capture_output=True,
    )
    if not out.stdout:
        return None
    return build_signals(out.stdout, fs=fs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_signals.py -v`
Expected: PASS (4 tests). Note: the magnitude endpoints test depends only on the first and last ACCL samples, which resample preserves exactly.

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/__init__.py tandem/phases/signals.py tests/phases/__init__.py tests/phases/test_signals.py
git commit -m "feat(phases): GPMF signals extraction (accel magnitude + 3D speed)"
```

---

## Task 2: Exit detection

Find the exit instant: the deep accelerometer dropout that is followed by sustained high 3D speed.

**Files:**
- Create: `tandem/phases/detect.py`
- Test: `tests/phases/test_detect.py`

**Interfaces:**
- Consumes: `tandem.phases.signals.Signals`.
- Produces:
  - Module constants: `G = 9.80665`, `EXIT_DROPOUT_G = 0.35`, `FREEFALL_MIN_MS = 45.0`, `FREEFALL_MAX_MS = 60.0`, `ONEG_LOW_G = 0.6`, `ONEG_HIGH_G = 1.4`, `GROUND_SPEED_MS = 3.0`.
  - `Event` dataclass: `type: str`, `t_s: float`, `source: str`, `confidence: float`.
  - `detect_exit(sig: Signals) -> Event | None` — the exit event, or `None` if no dropout-then-fast pattern exists.

- [ ] **Step 1: Write the failing test**

`tests/phases/test_detect.py`:
```python
from tandem.phases.signals import Signals
from tandem.phases.detect import detect_exit, G


def _in_aircraft_then_freefall(fs=10.0):
    # 10 s ground+climb at ~1 g, low then plane speed; 2 s exit dropout;
    # 20 s freefall at ~1 g noisy, speed ~55.
    accel, speed = [], []
    for _ in range(int(10 * fs)):      # in aircraft
        accel.append(1.0 * G); speed.append(40.0)
    for _ in range(int(2 * fs)):       # exit: |a| ~0
        accel.append(0.05 * G); speed.append(50.0)
    for _ in range(int(20 * fs)):      # freefall
        accel.append(1.0 * G); speed.append(55.0)
    t = [i / fs for i in range(len(accel))]
    return Signals(t_s=t, accel_mag=accel, speed_3d=speed, fs=fs,
                   has_accel=True, has_gps=True)


def test_detect_exit_at_dropout():
    sig = _in_aircraft_then_freefall()
    ev = detect_exit(sig)
    assert ev is not None
    assert ev.type == "exit"
    assert ev.source == "telemetry"
    assert 9.5 <= ev.t_s <= 12.5     # within the dropout window (~10-12 s)
    assert ev.confidence > 0.5


def test_detect_exit_none_when_no_dropout():
    fs = 10.0
    accel = [1.0 * G] * int(30 * fs)   # steady 1 g, never drops
    speed = [40.0] * int(30 * fs)
    t = [i / fs for i in range(len(accel))]
    sig = Signals(t_s=t, accel_mag=accel, speed_3d=speed, fs=fs,
                  has_accel=True, has_gps=True)
    assert detect_exit(sig) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.phases.detect'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/phases/detect.py`:
```python
"""Detect physical phases from accelerometer magnitude and 3D speed.

Only phases up to freefall are emitted — the operator's canopy and
landing are on a different flight profile and never represent the
tandem (spec §5.1). All results carry source="telemetry".
"""
from __future__ import annotations

from dataclasses import dataclass

G = 9.80665
EXIT_DROPOUT_G = 0.35        # |a| below this (in g) marks free-fall onset
FREEFALL_MIN_MS = 45.0
FREEFALL_MAX_MS = 60.0
ONEG_LOW_G = 0.6
ONEG_HIGH_G = 1.4
GROUND_SPEED_MS = 3.0


@dataclass
class Event:
    type: str
    t_s: float
    source: str
    confidence: float


def _dropout_indices(sig) -> list[int]:
    thresh = EXIT_DROPOUT_G * G
    return [i for i, a in enumerate(sig.accel_mag) if a < thresh]


def detect_exit(sig):
    if not sig.has_accel or not sig.accel_mag:
        return None
    drops = _dropout_indices(sig)
    if not drops:
        return None
    # The exit dropout must be followed by sustained high speed.
    for i in drops:
        after = sig.speed_3d[i:]
        if after and max(after) >= FREEFALL_MIN_MS:
            depth = 1.0 - (sig.accel_mag[i] / G)      # how close to true 0 g
            confidence = max(0.0, min(1.0, depth))
            return Event(type="exit", t_s=sig.t_s[i], source="telemetry",
                         confidence=round(confidence, 3))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/detect.py tests/phases/test_detect.py
git commit -m "feat(phases): exit detection from accel dropout"
```

---

## Task 3: Freefall boundary detection

Find the freefall segment: the contiguous stretch after exit where 3D speed is in the freefall band and |a| sits near 1 g.

**Files:**
- Modify: `tandem/phases/detect.py`
- Test: `tests/phases/test_detect.py` (add cases)

**Interfaces:**
- Consumes: `Signals`, `Event` (exit), the constants from Task 2.
- Produces:
  - `Segment` dataclass: `type: str`, `start_s: float`, `end_s: float`, `source: str`, `confidence: float`.
  - `detect_freefall(sig: Signals, exit_event: Event | None) -> Segment | None` — freefall segment starting at/after exit, `None` if none.

- [ ] **Step 1: Write the failing test**

Add to `tests/phases/test_detect.py`:
```python
from tandem.phases.detect import detect_freefall, Segment


def test_detect_freefall_after_exit():
    sig = _in_aircraft_then_freefall()
    ev = detect_exit(sig)
    seg = detect_freefall(sig, ev)
    assert seg is not None
    assert seg.type == "freefall"
    assert seg.source == "telemetry"
    assert seg.start_s >= ev.t_s - 0.5
    assert seg.end_s > seg.start_s
    # freefall spans roughly the last 20 s (12..32)
    assert seg.end_s - seg.start_s >= 15.0


def test_detect_freefall_none_without_exit():
    sig = _in_aircraft_then_freefall()
    assert detect_freefall(sig, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_freefall'`.

- [ ] **Step 3: Write minimal implementation**

Append to `tandem/phases/detect.py`:
```python
@dataclass
class Segment:
    type: str
    start_s: float
    end_s: float
    source: str
    confidence: float


def _is_freefall_sample(a: float, v: float) -> bool:
    return (FREEFALL_MIN_MS <= v <= FREEFALL_MAX_MS
            and ONEG_LOW_G * G <= a <= ONEG_HIGH_G * G)


def detect_freefall(sig, exit_event):
    if exit_event is None or not sig.speed_3d:
        return None
    # Find the exit index, then the longest freefall run at/after it.
    start_idx = min(range(len(sig.t_s)), key=lambda i: abs(sig.t_s[i] - exit_event.t_s))
    best = None
    run_start = None
    for i in range(start_idx, len(sig.t_s)):
        if _is_freefall_sample(sig.accel_mag[i], sig.speed_3d[i]):
            if run_start is None:
                run_start = i
            run_end = i
            if best is None or (run_end - run_start) > (best[1] - best[0]):
                best = (run_start, run_end)
        else:
            run_start = None
    if best is None:
        return None
    lo, hi = best
    return Segment(type="freefall", start_s=sig.t_s[lo], end_s=sig.t_s[hi],
                   source="telemetry", confidence=0.9)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: PASS (4 tests total in the file).

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/detect.py tests/phases/test_detect.py
git commit -m "feat(phases): freefall boundary detection"
```

---

## Task 4: Ground and climb segmentation

Split the pre-exit stretch into ground (aircraft stationary, speed ≈ 0) and climb (aircraft moving), by 3D speed.

**Files:**
- Modify: `tandem/phases/detect.py`
- Test: `tests/phases/test_detect.py` (add cases)

**Interfaces:**
- Consumes: `Signals`, `Event` (exit), `Segment`, `GROUND_SPEED_MS`.
- Produces:
  - `detect_ground_climb(sig: Signals, exit_event: Event | None) -> list[Segment]` — up to two segments (`ground_pre`, `climb`) covering `[0, exit)`; empty list if no exit.

- [ ] **Step 1: Write the failing test**

Add to `tests/phases/test_detect.py`:
```python
from tandem.phases.detect import detect_ground_climb


def test_ground_then_climb_before_exit():
    fs = 10.0
    accel, speed = [], []
    for _ in range(int(5 * fs)):       # ground: stationary
        accel.append(1.0 * G); speed.append(0.5)
    for _ in range(int(5 * fs)):       # climb: plane moving
        accel.append(1.0 * G); speed.append(40.0)
    for _ in range(int(2 * fs)):       # exit dropout
        accel.append(0.05 * G); speed.append(50.0)
    for _ in range(int(10 * fs)):      # freefall
        accel.append(1.0 * G); speed.append(55.0)
    t = [i / fs for i in range(len(accel))]
    from tandem.phases.signals import Signals
    sig = Signals(t_s=t, accel_mag=accel, speed_3d=speed, fs=fs,
                  has_accel=True, has_gps=True)
    ev = detect_exit(sig)
    segs = detect_ground_climb(sig, ev)
    types = [s.type for s in segs]
    assert types == ["ground_pre", "climb"]
    assert segs[0].start_s == 0.0
    assert segs[0].end_s <= 5.5          # ground ends ~5 s
    assert segs[1].end_s <= ev.t_s + 0.2 # climb ends at exit


def test_ground_climb_empty_without_exit():
    sig = _in_aircraft_then_freefall()
    assert detect_ground_climb(sig, None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_ground_climb'`.

- [ ] **Step 3: Write minimal implementation**

Append to `tandem/phases/detect.py`:
```python
def detect_ground_climb(sig, exit_event):
    if exit_event is None or not sig.speed_3d:
        return []
    exit_idx = min(range(len(sig.t_s)), key=lambda i: abs(sig.t_s[i] - exit_event.t_s))
    if exit_idx <= 0:
        return []
    # First index where speed rises above the ground threshold = takeoff.
    takeoff = None
    for i in range(exit_idx):
        if sig.speed_3d[i] > GROUND_SPEED_MS:
            takeoff = i
            break
    segments = []
    if takeoff is None:
        # Never left the ground before exit (unusual) — whole span is ground.
        segments.append(Segment("ground_pre", sig.t_s[0], sig.t_s[exit_idx],
                                 "telemetry", 0.9))
        return segments
    if takeoff > 0:
        segments.append(Segment("ground_pre", sig.t_s[0], sig.t_s[takeoff],
                                 "telemetry", 0.95))
    segments.append(Segment("climb", sig.t_s[takeoff], sig.t_s[exit_idx],
                             "telemetry", 0.9))
    return segments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: PASS (6 tests total in the file).

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/detect.py tests/phases/test_detect.py
git commit -m "feat(phases): ground/climb segmentation by 3D speed"
```

---

## Task 5: Phase-order FSM

Validate that the detected phases occur in the physically-mandatory order. The FSM does not find events — it rejects nonsense (freefall before exit, climb after freefall).

**Files:**
- Create: `tandem/phases/fsm.py`
- Test: `tests/phases/test_fsm.py`

**Interfaces:**
- Consumes: `tandem.phases.detect.Segment`, `Event`.
- Produces:
  - `PHASE_ORDER = ["ground_pre", "climb", "freefall"]` (module constant).
  - `validate_order(segments: list[Segment]) -> str | None` — returns `None` when the segments' start times are non-decreasing and their types follow `PHASE_ORDER`; returns `"PHASE_ORDER_VIOLATION"` otherwise.

- [ ] **Step 1: Write the failing test**

`tests/phases/test_fsm.py`:
```python
from tandem.phases.detect import Segment
from tandem.phases.fsm import validate_order, PHASE_ORDER


def _seg(t, s, e):
    return Segment(t, s, e, "telemetry", 0.9)


def test_valid_order_passes():
    segs = [_seg("ground_pre", 0.0, 5.0), _seg("climb", 5.0, 12.0),
            _seg("freefall", 12.0, 30.0)]
    assert validate_order(segs) is None


def test_freefall_before_climb_violates():
    segs = [_seg("freefall", 0.0, 10.0), _seg("climb", 10.0, 20.0)]
    assert validate_order(segs) == "PHASE_ORDER_VIOLATION"


def test_out_of_time_order_violates():
    segs = [_seg("ground_pre", 5.0, 10.0), _seg("climb", 0.0, 4.0)]
    assert validate_order(segs) == "PHASE_ORDER_VIOLATION"


def test_empty_is_ok():
    assert validate_order([]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_fsm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.phases.fsm'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/phases/fsm.py`:
```python
"""Phase-order validation. Rejects physically-impossible orderings;
it does not detect anything. A violation means the telemetry (or the
detector) is untrustworthy for this recording."""
from __future__ import annotations

PHASE_ORDER = ["ground_pre", "climb", "freefall"]


def validate_order(segments) -> str | None:
    last_rank = -1
    last_start = float("-inf")
    for seg in segments:
        if seg.type not in PHASE_ORDER:
            continue
        rank = PHASE_ORDER.index(seg.type)
        if rank < last_rank or seg.start_s < last_start:
            return "PHASE_ORDER_VIOLATION"
        last_rank = rank
        last_start = seg.start_s
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_fsm.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/fsm.py tests/phases/test_fsm.py
git commit -m "feat(phases): phase-order FSM validation"
```

---

## Task 6: Assembly API

Combine the detectors and the FSM into one entry point that returns phases, events, and degradations from a `Signals` object — the physical detector's public surface.

**Files:**
- Create: `tandem/phases/api.py`
- Test: `tests/phases/test_api.py`

**Interfaces:**
- Consumes: `Signals`, `detect_exit`, `detect_freefall`, `detect_ground_climb`, `validate_order`.
- Produces:
  - `PhaseResult` dataclass: `phases: list[Segment]`, `events: list[Event]`, `degradations: list[str]`.
  - `detect_phases(sig: Signals) -> PhaseResult` — runs the pipeline. When `sig` has no usable signal (`not sig.has_accel and not sig.has_gps`, or empty arrays), returns an empty result carrying `"NO_TELEMETRY"`. When the FSM rejects the order, includes `"PHASE_ORDER_VIOLATION"` and still returns what was detected (phases marked unreliable via confidence, per the spec's "phases marked unreliable").

- [ ] **Step 1: Write the failing test**

`tests/phases/test_api.py`:
```python
from tandem.phases.signals import Signals
from tandem.phases.detect import G
from tandem.phases.api import detect_phases, PhaseResult


def _full_flight(fs=10.0):
    accel, speed = [], []
    for _ in range(int(5 * fs)):       # ground
        accel.append(1.0 * G); speed.append(0.5)
    for _ in range(int(5 * fs)):       # climb
        accel.append(1.0 * G); speed.append(40.0)
    for _ in range(int(2 * fs)):       # exit
        accel.append(0.05 * G); speed.append(50.0)
    for _ in range(int(20 * fs)):      # freefall
        accel.append(1.0 * G); speed.append(55.0)
    t = [i / fs for i in range(len(accel))]
    return Signals(t_s=t, accel_mag=accel, speed_3d=speed, fs=fs,
                   has_accel=True, has_gps=True)


def test_detect_phases_full_flight():
    res = detect_phases(_full_flight())
    phase_types = [p.type for p in res.phases]
    assert phase_types == ["ground_pre", "climb", "freefall"]
    assert [e.type for e in res.events] == ["exit"]
    assert res.degradations == []
    assert all(p.source == "telemetry" for p in res.phases)


def test_detect_phases_no_signal_is_no_telemetry():
    res = detect_phases(Signals())
    assert res.phases == [] and res.events == []
    assert res.degradations == ["NO_TELEMETRY"]


def test_detect_phases_flags_order_violation(monkeypatch):
    # Force the detectors to return physically-impossible, out-of-order
    # segments so the FSM branch in detect_phases is exercised directly.
    import tandem.phases.api as api
    from tandem.phases.detect import Segment, Event

    monkeypatch.setattr(api, "detect_exit",
                        lambda s: Event("exit", 12.0, "telemetry", 0.9))
    monkeypatch.setattr(api, "detect_ground_climb",
                        lambda s, e: [Segment("freefall", 0.0, 5.0, "telemetry", 0.9)])
    monkeypatch.setattr(api, "detect_freefall",
                        lambda s, e: Segment("climb", 5.0, 10.0, "telemetry", 0.9))

    res = detect_phases(_full_flight())
    assert "PHASE_ORDER_VIOLATION" in res.degradations
    # every phase is marked unreliable (confidence zeroed) on a violation
    assert res.phases and all(p.confidence == 0.0 for p in res.phases)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.phases.api'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/phases/api.py`:
```python
"""Public surface of the physical phase detector: Signals -> PhaseResult.

Emits phases up to freefall and the exit event, all source="telemetry".
Records NO_TELEMETRY when there is no usable signal, and
PHASE_ORDER_VIOLATION when the detected order is physically impossible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tandem.phases.detect import (
    Event, Segment, detect_exit, detect_freefall, detect_ground_climb,
)
from tandem.phases.fsm import validate_order


@dataclass
class PhaseResult:
    phases: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)


def detect_phases(sig) -> PhaseResult:
    usable = (sig.has_accel or sig.has_gps) and (sig.accel_mag or sig.speed_3d)
    if not usable:
        return PhaseResult(degradations=["NO_TELEMETRY"])

    exit_event = detect_exit(sig)
    ground_climb = detect_ground_climb(sig, exit_event)
    freefall = detect_freefall(sig, exit_event)

    phases: list[Segment] = list(ground_climb)
    if freefall is not None:
        phases.append(freefall)
    events: list[Event] = [exit_event] if exit_event is not None else []

    degradations: list[str] = []
    violation = validate_order(phases)
    if violation is not None:
        degradations.append(violation)
        # Mark every phase unreliable when the order did not converge.
        phases = [Segment(p.type, p.start_s, p.end_s, p.source, 0.0) for p in phases]

    return PhaseResult(phases=phases, events=events, degradations=degradations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_api.py -v`
Then the whole suite: `python -m pytest -q` — all green (recon tests plus the new phases tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/api.py tests/phases/test_api.py
git commit -m "feat(phases): assembly API with NO_TELEMETRY and order validation"
```

---

## Self-Review

**Spec coverage (§5.1 physical detector, primary-mode rows up to freefall):**
- Ground (speed ≈ 0, |a| ≈ 1 g) → Task 4 (`detect_ground_climb`, ground_pre).
- Climb (aircraft moving) → Task 4 (climb). Note: without altitude, climb is "pre-exit and moving," distinguished from ground by the `GROUND_SPEED_MS` threshold — consistent with the no-altitude constraint.
- Exit (|a| dropout to ~0, return to 1 g, speed accelerating to ~55) → Task 2 (`detect_exit`).
- Freefall (speed 45–60, |a| ≈ 1 g noisy) → Task 3 (`detect_freefall`).
- Phase-order FSM rejecting impossible orderings, → `PHASE_ORDER_VIOLATION` in `degradations` → Task 5 + Task 6.
- `NO_TELEMETRY` when GPMF is absent/unusable → Task 6.
- Every result carries `source="telemetry"` → Tasks 2–4, asserted in Task 6.
- Accelerometer signal added to the pipeline (recon only had 3D speed) → Task 1 (`build_signals` parses `ACCL`).

**Explicitly out of scope (become later plans or belong elsewhere):** operator canopy/landing detection (spec says these do not transfer to the tandem, so this detector stops at freefall); mapping recording-relative seconds to `t0_utc` (downstream assembly); precise GPMF per-payload timing (`TSMP`); writing `telemetry.parquet` (Stage-1 extraction plan); all visual detectors and the fusion into `scenes.json`.

**Placeholder scan:** No `TBD`/"handle edge cases" steps; every code step has real code and every test step a runnable command with expected output. Thresholds are named constants in `detect.py`, not inlined magic numbers.

**Type consistency:** `Signals` (Task 1) is consumed unchanged by Tasks 2–6. `Event` and `Segment` are defined in `detect.py` (Tasks 2–3) and imported by `fsm.py` (only for typing/attribute access) and `api.py`. `detect_exit`/`detect_freefall`/`detect_ground_climb`/`validate_order`/`detect_phases` keep the signatures declared in their Interfaces blocks wherever later tasks call them.

**Known real-data risks (surfaced, not hidden):**
- Thresholds (`EXIT_DROPOUT_G`, the freefall band, `GROUND_SPEED_MS`) are §5.1 defaults; they must be calibrated on real telemetry from the Phase-0 archive before this detector is trusted. The recon tool's assumption-1 finding (GPS-UTC present) and a first real ACCL trace are the inputs to that calibration.
- The uniform-per-stream timing assumption in `signals.py` is coarse; exit/freefall boundaries are robust to it at seconds scale, but if calibration shows drift, precise GPMF payload timing becomes a follow-up task.
- `detect_exit` picks the first dropout followed by sustained speed; a bounce or sensor glitch mid-climb that momentarily reads < 0.35 g while a later GPS sample happens to be fast could mis-fire. Real ACCL traces from Phase-0 will show whether a minimum-duration guard on the dropout is needed.
