# Accelerometer-First Phase Detection & Multi-Payload Telemetry Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make phase detection work on the real archive — read the *whole* GPMF stream (not just its first second), detect exit (via a jerk gate) and freefall/ground/climb from the accelerometer alone (GPS optional), trim the freefall window at the operator's opening shock, and reject the firmware-default GPS timestamp.

**Architecture:** Fixes and extends the existing `tandem/phases/` (signals, detect, api) and `tandem/recon/telemetry.py`. The redesign flips the dependency established in the first phase-detector plan: the accelerometer becomes the primary signal and GPS 3D speed becomes optional corroboration, because ~50% of the real archive has no GPS at all. All logic stays pure and unit-testable on synthetic multi-payload GPMF blobs; provisional thresholds are grounded in a real accelerometer trace (see Design notes) and remain named constants for later calibration.

**Tech Stack:** Python 3.10+, numpy (optional), pytest. Reuses `tandem/recon/gpmf.py`. No new third-party deps.

## Global Constraints

- **Python floor: 3.10**; `from __future__ import annotations`; no 3.11+ syntax.
- **No GPS altitude, ever.** Descent/phase logic uses accelerometer magnitude (primary) and GPS 3D speed (optional). GPS altitude is never read.
- **Accelerometer is the primary phase signal; GPS is optional.** Nothing in phase detection may *require* GPS to be present. Where GPS is absent (`has_gps == False`), exit/freefall/ground/climb must still be produced from `accel_mag` alone.
- **Read the whole GPMF stream.** Real GoPro telemetry is a sequence of DEVC payloads (~1 per second, ~200 ACCL Hz). Any parse that stops at the first payload is a defect.
- **`source="telemetry"`** on every emitted phase/event. Times are recording-relative seconds.
- **Degradations:** `NO_TELEMETRY` (no usable accel or gps), `PHASE_ORDER_VIOLATION`, and new `UTC_UNRELIABLE` (GPS UTC absent, or equal to the firmware pre-fix default).
- **Thresholds are provisional** §5.1 defaults grounded in one real trace; keep them named constants. Calibration across more jumps is a later step.

## Design notes (read before Task 1) — grounded in a real GPS-less jump

A real 202 s recording (Родионов `GX012475.MP4`, no GPS) yielded 201 ACCL payloads / 40353 samples at ~200 Hz. The current `build_signals` returned **5 samples (~1 s)** for the same file — it reads only the first DEVC payload. That is the first thing to fix; every downstream number depends on it.

Observed `|a|` envelope (in g): steady phases sit at ~1.0 g with std ~0.01–0.05; movement/free-fall shows std ~0.10–0.15; **exit/free-fall onset dips to ~0.1 g** (near true 0 g); ground camera-handling bumps also hit ~2.3–2.9 g but last <1 s. **Validated on a real accel-only jump** (Дмитрий `GX010015`, no GPS used): the deepest smoothed (~1 s) `|a|` dip = **0.39 g at t=39.1 s** (raw min 0.13 g) is the exit, and the **largest post-exit spike = 5.89 g at t=91.8 s** is the canopy opening — a ~39–92 s free-fall that matched an independent vision review exactly. So:
- **Exit** = detected by FUSING two cues (per the domain owner): (a) a rapid **jerk** — `|a|` in the min-pooled envelope drops from a steady ~1 g to below ~0.35 g *quickly* (the rate of the drop, not just the low level; a ground bump also reads low but does not come from a sustained ~1 g), and (b) a **frame-exposure jump** — the cabin is dark, outside is bright, so the keyframe brightens sharply at exit. This plan implements the telemetry jerk (came-from-~1g rapid drop); the exposure cue is a fusion input to add once keyframe luminance is available (recon already extracts keyframes, so per-frame brightness is cheap).
- **Operator freefall window** = from exit to the operator's **accelerometer opening-shock** (largest `|a|` peak in ≈15–90 s after exit; real openings ~2–6 g, here 5.89 g, tower over buffeting ~2–3 g). This bounds the *operator's* freefall for aiming frame sampling — it is **NOT** the tandem canopy event.
- **Tandem canopy opening is detected strictly VISUALLY** — a sharp growth of the object near the frame **centre** as the pair's canopy blooms — optionally corroborated by a head-jerk (the operator pitches up to keep the decelerating pair in frame). It is a **separate visual-detector plan**, because the operator is often a camera-flyer who opens their own canopy *later* than the tandem, so the operator's accel opening-shock is the operator's event, not the pair's.
- **Downsampling erases the exit transient.** A 5 s mean stays ~1 g even when the 1 s minimum is 0.1 g. So the resampled grid must **preserve the minimum envelope** for dropout detection (min-pooling), not linear-average it away.

## File Structure

```
tandem/phases/signals.py    # MODIFY: accumulate ACCL+GPS across ALL payloads; add min/max envelopes
tandem/phases/detect.py     # MODIFY: accel-first exit; opening-shock freefall end; GPS optional
tandem/phases/api.py        # MODIFY: NO_TELEMETRY only when neither accel nor gps usable
tandem/recon/telemetry.py   # MODIFY: GPSU-with-fix; quarantine firmware-default UTC
tests/phases/test_signals.py
tests/phases/test_detect.py
tests/phases/test_api.py
tests/recon/test_telemetry.py
```

---

## Task 1: Read the whole GPMF stream (multi-payload accumulation)

The critical fix. `build_signals` must accumulate ACCL and GPS across every DEVC payload, and preserve a minimum envelope so the exit dip survives resampling.

**Files:**
- Modify: `tandem/phases/signals.py`
- Test: `tests/phases/test_signals.py`

**Interfaces:**
- Consumes: `tandem.recon.gpmf.walk`, `iter_klv`, `decode_numbers`.
- Produces (extends `Signals`):
  - `Signals` gains `accel_min: list[float]` (min-pooled |a| per grid cell, m/s²) alongside the existing `accel_mag` (mean-pooled). Fields: `t_s`, `accel_mag`, `accel_min`, `speed_3d`, `fs`, `has_accel`, `has_gps`.
  - `build_signals(blob, fs=10.0)` accumulates ACCL magnitudes and GPS5 3D-speed across **all** STRM containers in the blob (not the first), then resamples: `accel_mag`/`speed_3d` by averaging, `accel_min` by min-pooling.
  - `pool_min(values: list[float], n_out: int) -> list[float]` (pure) — min over each output bin.

- [ ] **Step 1: Write the failing test**

Add to `tests/phases/test_signals.py` (keep existing tests):
```python
def test_build_signals_accumulates_across_payloads():
    # Two DEVC payloads, each a STRM with SCAL+ACCL. Payload 1 has a deep dip.
    def strm_accl(vals):  # vals: list of (x,y,z) int16 tuples
        scal = _klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 100))
        payload = b"".join(struct.pack(">3h", *v) for v in vals)
        accl = _klv(b"ACCL", b"s", 6, len(vals), payload)
        inner = scal + accl
        return _klv(b"STRM", b"\x00", 1, len(inner), inner)

    p1 = _klv(b"DEVC", b"\x00", 1, len(strm_accl([(10, 0, 0)] * 4)),
              strm_accl([(10, 0, 0)] * 4))          # |a| ~0.1 (raw 10/100=0.1)
    p2 = _klv(b"DEVC", b"\x00", 1, len(strm_accl([(981, 0, 0)] * 4)),
              strm_accl([(981, 0, 0)] * 4))         # |a| ~9.81 (~1 g)
    blob = p1 + p2

    sig = build_signals(blob, fs=10.0)
    assert sig.has_accel is True
    # both payloads contributed: min envelope reaches the deep dip, mean reaches ~1g
    assert min(sig.accel_min) < 1.0            # the 0.1 m/s^2 dip survived
    assert max(sig.accel_mag) > 5.0            # the ~9.81 payload survived
    assert len(sig.accel_min) == len(sig.accel_mag) == len(sig.t_s)


def test_pool_min_takes_bin_minimum():
    from tandem.phases.signals import pool_min
    assert pool_min([5.0, 1.0, 4.0, 3.0], 2) == [1.0, 3.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_signals.py::test_build_signals_accumulates_across_payloads -v`
Expected: FAIL — current `build_signals` reads only the first payload, so `max(sig.accel_mag)` stays ~0.1 and the assertion fails (and `pool_min`/`accel_min` do not exist).

- [ ] **Step 3: Write minimal implementation**

In `tandem/phases/signals.py`: add `accel_min` to the `Signals` dataclass; add `pool_min`; rewrite `build_signals` to accumulate across payloads.
```python
@dataclass
class Signals:
    t_s: list[float] = field(default_factory=list)
    accel_mag: list[float] = field(default_factory=list)
    accel_min: list[float] = field(default_factory=list)
    speed_3d: list[float] = field(default_factory=list)
    fs: float = 10.0
    has_accel: bool = False
    has_gps: bool = False


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


def build_signals(blob: bytes, fs: float = 10.0) -> Signals:
    accel_raw: list[float] = []
    speed_raw: list[float] = []
    saw_accel = False
    saw_gps = False
    for children in _stream_children(blob):
        a = _accel_magnitudes(children)
        if a is not None:
            saw_accel = True
            accel_raw.extend(a)
        s = _gps_speeds(children)
        if s is not None:
            saw_gps = True
            speed_raw.extend(s)
    sig = Signals(fs=fs, has_accel=saw_accel, has_gps=saw_gps)
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
    return sig
```
Note: `saw_accel`/`saw_gps` are set from whether any payload produced samples, so an empty-but-present stream does not falsely flag (addresses a prior deferred minor).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_signals.py -v`
Expected: PASS (existing tests plus the two new ones). The `test_build_signals_accel_only_pads_speed_with_zeros` test still holds.

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/signals.py tests/phases/test_signals.py
git commit -m "fix(phases): read the whole GPMF stream (all payloads) + min envelope"
```

---

## Task 2: GPSU-with-fix and firmware-default UTC quarantine

`recon/telemetry.py` takes the first GPSU regardless of GPS fix, so it reports the firmware pre-fix default `2021-03-07T00:00:02`. Take the first GPSU that comes with a valid fix, and flag the default as unreliable.

**Files:**
- Modify: `tandem/recon/telemetry.py`
- Test: `tests/recon/test_telemetry.py`

**Interfaces:**
- Produces (extends `Telemetry`): `utc_reliable: bool` (default `False`). `parse_telemetry_blob` sets `first_utc` from the first GPSU whose sibling `GPSF` (fix type) is >= 2, and `utc_reliable=True` only then; if the only UTC available equals the firmware default epoch (year 2021, `2021-03-07`), leave `utc_reliable=False`.
- Constant: `FIRMWARE_DEFAULT_YEAR = 2021`.

- [ ] **Step 1: Write the failing test**

Add to `tests/recon/test_telemetry.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_telemetry.py -v`
Expected: FAIL — `Telemetry` has no `utc_reliable`, and the current parser latches the first (unfixed) GPSU.

- [ ] **Step 3: Write minimal implementation**

In `tandem/recon/telemetry.py`: add `utc_reliable: bool = False` to `Telemetry`, `FIRMWARE_DEFAULT_YEAR = 2021`, and in the STRM loop read `GPSF` alongside `GPSU`/`SCAL`/`GPS5`:
```python
FIRMWARE_DEFAULT_YEAR = 2021
# inside the per-STRM child loop, also capture:
#     elif c.key == "GPSF": gpsf = decode_numbers(c)[0][0]
# after the loop, replace the has_utc handling with:
        if gpsu is not None:
            when = decode_utc(gpsu.payload)
            fixed = gpsf is not None and gpsf >= 2
            if fixed and not tel.utc_reliable:
                tel.first_utc = when
                tel.has_utc = True
                tel.utc_reliable = True
            elif tel.first_utc is None:
                tel.first_utc = when          # fallback, but unreliable
                tel.has_utc = True
                tel.utc_reliable = when.year != FIRMWARE_DEFAULT_YEAR
```
(Keep the existing GPS5/SCAL accumulation intact.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/recon/test_telemetry.py -v`
Expected: PASS (existing plus the two new tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/telemetry.py tests/recon/test_telemetry.py
git commit -m "fix(recon): take GPSU with a valid fix; quarantine firmware-default UTC"
```

---

## Task 3: Accelerometer-first exit detection

Redefine exit as a sustained near-0g accelerometer dip, requiring no GPS. GPS speed, when present, only raises confidence.

**Files:**
- Modify: `tandem/phases/detect.py`
- Test: `tests/phases/test_detect.py`

**Interfaces:**
- Consumes: `Signals` (now with `accel_min`).
- Produces: constants `EXIT_MIN_DURATION_S = 1.0`, `PRE_EXIT_G = 0.8`, `JERK_LOOKBACK_S = 1.5`, `OPENING_SHOCK_G = 2.5`, `FREEFALL_MIN_S = 15.0`, `FREEFALL_MAX_S = 90.0`. `detect_exit(sig)` finds a min-envelope dip below `EXIT_DROPOUT_G` lasting ≥ `EXIT_MIN_DURATION_S` that dropped from a steady ~1 g — a **jerk gate**: the level `JERK_LOOKBACK_S` before the dip must have been ≥ `PRE_EXIT_G`. It does **not** require GPS. Confidence rises when GPS speed after the dip exceeds `FREEFALL_MIN_MS`. (The visual exposure-jump cue is fused in a later plan.)

- [ ] **Step 1: Write the failing test**

Add to `tests/phases/test_detect.py`:
```python
def _accel_only_flight(fs=10.0):
    from tandem.phases.signals import Signals
    amin, amag = [], []
    for _ in range(int(10 * fs)):      # in aircraft ~1g
        amin.append(0.9 * G); amag.append(1.0 * G)
    for _ in range(int(3 * fs)):       # exit: deep dip, >1s
        amin.append(0.1 * G); amag.append(0.3 * G)
    for _ in range(int(40 * fs)):      # freefall ~1g, noisy
        amin.append(0.8 * G); amag.append(1.0 * G)
    for _ in range(int(3 * fs)):       # opening shock ~3.5g (towers over buffeting)
        amin.append(3.0 * G); amag.append(3.5 * G)
    t = [i / fs for i in range(len(amin))]
    speed = [0.0] * len(amin)          # NO GPS
    return Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=speed,
                   fs=fs, has_accel=True, has_gps=False)


def test_detect_exit_from_accel_alone_without_gps():
    sig = _accel_only_flight()
    ev = detect_exit(sig)
    assert ev is not None
    assert ev.type == "exit" and ev.source == "telemetry"
    assert 9.5 <= ev.t_s <= 12.0       # at the dip onset (~10s)


def test_brief_dip_is_not_an_exit():
    from tandem.phases.signals import Signals
    fs = 10.0
    amin = [0.9 * G] * int(20 * fs)
    for k in range(3):                  # 0.3s blip, shorter than EXIT_MIN_DURATION_S
        amin[int(10 * fs) + k] = 0.1 * G
    amag = [1.0 * G] * len(amin)
    t = [i / fs for i in range(len(amin))]
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=[0.0] * len(amin),
                  fs=fs, has_accel=True, has_gps=False)
    assert detect_exit(sig) is None


def test_gradual_low_g_without_prior_1g_is_not_exit():
    # A recording already in a low-g state (never a steady ~1g before): the jerk
    # gate must reject it — there was no rapid drop FROM 1g.
    from tandem.phases.signals import Signals
    fs = 10.0
    amin = [0.2 * G] * int(20 * fs)
    amag = [0.3 * G] * int(20 * fs)
    t = [i / fs for i in range(len(amin))]
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=[0.0] * len(amin),
                  fs=fs, has_accel=True, has_gps=False)
    assert detect_exit(sig) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_detect.py::test_detect_exit_from_accel_alone_without_gps -v`
Expected: FAIL — current `detect_exit` requires `speed_3d` to reach `FREEFALL_MIN_MS`, which is all-zero here, so it returns `None`.

- [ ] **Step 3: Write minimal implementation**

Replace `detect_exit` in `tandem/phases/detect.py`:
```python
EXIT_MIN_DURATION_S = 1.0
PRE_EXIT_G = 0.8          # the drop must come FROM a steady ~1g (jerk gate)
JERK_LOOKBACK_S = 1.5     # measure the pre-drop level this far before the dip
OPENING_SHOCK_G = 2.5      # a real canopy opening (~2-6g) towers over freefall buffeting (~2-3g)
FREEFALL_MIN_S = 15.0
FREEFALL_MAX_S = 90.0


def _pre_drop_level(sig, i):
    """Mean |a| over ~1 s ending JERK_LOOKBACK_S before index i (the in-aircraft level)."""
    back = int(round(JERK_LOOKBACK_S * sig.fs))
    lo = max(0, i - back)
    hi = min(len(sig.accel_mag), lo + max(1, int(round(1.0 * sig.fs))))
    seg = sig.accel_mag[lo:hi]
    return sum(seg) / len(seg) if seg else 0.0


def detect_exit(sig):
    if not sig.has_accel or not sig.accel_min:
        return None
    thresh = EXIT_DROPOUT_G * G
    min_samples = max(1, int(round(EXIT_MIN_DURATION_S * sig.fs)))
    run_start = None
    for i, a in enumerate(sig.accel_min):
        if a < thresh:
            if run_start is None:
                run_start = i
            if i - run_start + 1 >= min_samples:
                # jerk gate: the dip must have dropped rapidly FROM ~1g, not be a
                # gradual low-g stretch. (Duration already filters <1s ground bumps.)
                if _pre_drop_level(sig, run_start) < PRE_EXIT_G * G:
                    run_start = None
                    continue
                depth = 1.0 - (sig.accel_min[run_start] / G)
                conf = max(0.0, min(1.0, depth))
                if sig.has_gps and sig.speed_3d[run_start:] and \
                        max(sig.speed_3d[run_start:]) >= FREEFALL_MIN_MS:
                    conf = min(1.0, conf + 0.1)   # GPS corroboration
                return Event(type="exit", t_s=sig.t_s[run_start], source="telemetry",
                             confidence=round(conf, 3))
        else:
            run_start = None
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: PASS. The earlier GPS-based exit tests still pass (they have both a dip in `accel_min` — add `accel_min` to their `Signals` construction if missing — and high speed; when updating, mirror `accel_mag` into `accel_min` in those fixtures).

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/detect.py tests/phases/test_detect.py
git commit -m "feat(phases): accelerometer-first exit detection (GPS optional)"
```

---

## Task 4: Operator freefall-window bound (accelerometer opening-shock)

The operator's freefall ends at their accelerometer opening-shock — the largest `|a|` peak in the plausible window after exit. This bounds the *operator's* freefall for aiming frame sampling and trims the over-long windows the recon run produced. **NOTE:** this is the operator's freefall bound, **not** the tandem canopy event — the tandem canopy is detected strictly visually in a separate plan (the operator, often a camera-flyer, opens their own canopy later than the pair).

**Files:**
- Modify: `tandem/phases/detect.py`
- Test: `tests/phases/test_detect.py`

**Interfaces:**
- Produces: `detect_freefall(sig, exit_event) -> Segment | None` redefined — freefall spans from `exit_event.t_s` to the **largest `accel_mag` peak** within `[exit+FREEFALL_MIN_S, exit+FREEFALL_MAX_S]` (the canopy opening), provided that peak reaches `OPENING_SHOCK_G * G`; if no qualifying peak, the window is capped at `exit+FREEFALL_MAX_S`. Taking the max (not a first-crossing) prevents an in-air maneuver from truncating the window early.

- [ ] **Step 1: Write the failing test**

Add to `tests/phases/test_detect.py`:
```python
def test_freefall_ends_at_opening_shock():
    sig = _accel_only_flight()          # exit ~10s, freefall 13-53s, shock 53-56s
    ev = detect_exit(sig)
    seg = detect_freefall(sig, ev)
    assert seg is not None and seg.type == "freefall"
    assert seg.start_s <= ev.t_s + 0.5
    # freefall must END at the opening shock (~53s), not run to the recording end (~56s)
    assert 51.0 <= seg.end_s <= 54.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_detect.py::test_freefall_ends_at_opening_shock -v`
Expected: FAIL — the current `detect_freefall` uses the GPS speed band (all zero here) and would return `None` or an untrimmed span.

- [ ] **Step 3: Write minimal implementation**

Replace `detect_freefall` in `tandem/phases/detect.py`:
```python
def detect_freefall(sig, exit_event):
    if exit_event is None or not sig.accel_mag:
        return None
    start_idx = min(range(len(sig.t_s)), key=lambda i: abs(sig.t_s[i] - exit_event.t_s))
    lo = start_idx + int(round(FREEFALL_MIN_S * sig.fs))
    hi = min(len(sig.accel_mag), start_idx + int(round(FREEFALL_MAX_S * sig.fs)))
    end_idx = None
    if lo < hi:
        peak = max(range(lo, hi), key=lambda i: sig.accel_mag[i])
        if sig.accel_mag[peak] >= OPENING_SHOCK_G * G:   # the opening towers over buffeting
            end_idx = peak
    if end_idx is None:                                   # no clear opening -> cap at max plausible freefall
        end_idx = min(hi, len(sig.t_s) - 1)
    if end_idx <= start_idx:
        return None
    return Segment(type="freefall", start_s=sig.t_s[start_idx], end_s=sig.t_s[end_idx],
                   source="telemetry", confidence=0.85)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases/test_detect.py -v`
Expected: PASS. Update the older freefall fixtures to carry `accel_min` and an opening-shock tail so they remain meaningful.

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/detect.py tests/phases/test_detect.py
git commit -m "feat(phases): freefall window trimmed at the opening shock"
```

---

## Task 5: Ground/climb without GPS, and NO_TELEMETRY only when truly empty

Ground/climb currently split by GPS speed. Without GPS, split the pre-exit span by accelerometer stillness (ground = very low `|a|` variance; climb = the rest before exit). And `NO_TELEMETRY` must fire only when neither accel nor gps is usable.

**Files:**
- Modify: `tandem/phases/detect.py`, `tandem/phases/api.py`
- Test: `tests/phases/test_detect.py`, `tests/phases/test_api.py`

**Interfaces:**
- `detect_ground_climb(sig, exit_event)`: when GPS present, keep the speed split; when GPS absent, treat the whole pre-exit span as `climb` (aircraft) unless a low-variance stationary prefix is detectable, which becomes `ground_pre`. Returns `list[Segment]`.
- `api.detect_phases(sig)`: `usable = sig.has_accel or sig.has_gps`; `NO_TELEMETRY` only when not usable.

- [ ] **Step 1: Write the failing test**

Add to `tests/phases/test_api.py`:
```python
def test_full_flight_without_gps_still_detects_phases():
    from tandem.phases.detect import G
    from tests.phases.test_detect import _accel_only_flight  # reuse fixture
    res = detect_phases(_accel_only_flight())
    assert "exit" in [e.type for e in res.events]
    assert "freefall" in [p.type for p in res.phases]
    assert "NO_TELEMETRY" not in res.degradations
```
(If cross-module fixture import is awkward, inline the same accel-only `Signals` in the test.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/phases/test_api.py::test_full_flight_without_gps_still_detects_phases -v`
Expected: FAIL — without GPS the old pipeline finds no exit and no freefall.

- [ ] **Step 3: Write minimal implementation**

Update `detect_ground_climb` to branch on `sig.has_gps` (GPS split unchanged; accel branch emits a single `climb` for the pre-exit span, and a leading `ground_pre` if the first seconds are low-variance), and update `api.detect_phases`'s `usable` guard as specified. Keep the FSM call and the confidence-zeroing on violation.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/phases -v`
Then the whole suite: `python -m pytest -q`. All green.

- [ ] **Step 5: Commit**

```bash
git add tandem/phases/detect.py tandem/phases/api.py tests/phases/test_detect.py tests/phases/test_api.py
git commit -m "feat(phases): ground/climb without GPS; NO_TELEMETRY only when truly empty"
```

---

## Task 6: Point recon's frame sampler at the accelerometer window

The recon sampler currently windows via the crude GPS-speed `estimate_windows`, which mis-detected 2 of 3 real jumps. Have it use the phase detector's trimmed freefall window when accelerometer telemetry is available, falling back to the old estimate only when there is no accel.

**Files:**
- Modify: `tandem/recon/cli.py`
- Test: `tests/recon/test_cli.py`

**Interfaces:**
- `run()` builds `Signals` for the telemetried recording (via `tandem.phases.signals.build_signals` on its GPMF blob), runs `detect_phases`, and if a freefall `Segment` exists, samples frames across *that* trimmed window; otherwise falls back to `estimate_windows`. Import `build_signals` and `detect_phases` at module scope so the test can monkeypatch them.

- [ ] **Step 1: Write the failing test**

Add a `tests/recon/test_cli.py` test: monkeypatch `cli.detect_phases` to return a `PhaseResult` with a freefall `Segment` of `start_s=40, end_s=84`, monkeypatch `cli.extract_frames` to capture the shot times, and assert the sampled shot times fall within `[40, 84]` (proving the trimmed accel window drove sampling, not the 0-based GPS estimate).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/recon/test_cli.py -v`
Expected: FAIL — `cli` does not yet call `detect_phases`.

- [ ] **Step 3: Write minimal implementation**

Wire `run()` as described: prefer the accel-based freefall segment for `sample_plan`; keep `estimate_windows` as the no-accel fallback. Keep the report/probe assembly unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q` — whole suite green.

- [ ] **Step 5: Commit**

```bash
git add tandem/recon/cli.py tests/recon/test_cli.py
git commit -m "feat(recon): sample frames across the accelerometer freefall window"
```

---

## Self-Review

**Spec / findings coverage:**
- Multi-payload GPMF read (the real bug: 1 s vs 202 s) → Task 1.
- Accelerometer-first exit, GPS optional (finding A, rec 1) → Tasks 3, 5.
- Opening-shock freefall trim (finding D, rec 5) → Task 4, wired into recon at Task 6.
- GPSU-with-fix + firmware-default UTC quarantine (finding B, rec 3) → Task 2 (`UTC_UNRELIABLE`/`utc_reliable`).
- GPS-less flights still produce phases (finding A) → Task 5 (`usable` guard), verified by an accel-only end-to-end test.

**Out of scope (next plan — archive ingest):** broaden the filename parser + media glob for DJI/`.MOV`/7-digit GoPro names and never drop a file silently (rec 4); non-GPS jump grouping by file order/mtime (rec 2); cross-file jumps (freefall straddling recording boundaries, seen in the real trace). These are ingest concerns, separable from telemetry-signal work.

**Out of scope (next plan — visual detectors / fusion):** per the domain owner, the tandem **canopy opening** is detected strictly **visually** — a sharp growth of the object near the frame **centre** — optionally corroborated by a head-jerk; and the **exit** detection fuses this plan's telemetry jerk with a **frame-exposure jump** (dark cabin → bright sky). Both need extracted-keyframe pixels, so they belong with the visual / feature-extraction work. This plan's accel opening-shock is only a sampling-window bound, never the emitted tandem canopy event.

**Placeholder scan:** every code step carries real code; every test step a runnable command and expected result. Thresholds (`EXIT_DROPOUT_G`, `OPENING_SHOCK_G`, `EXIT_MIN_DURATION_S`, `FREEFALL_MIN_MS`) are named constants, grounded in the real trace (0.09 g dips, 2.3–2.9 g spikes), pending broader calibration.

**Type consistency:** `Signals` gains `accel_min` (Task 1) and every consumer that indexes it (Tasks 3, 4) sees equal-length `accel_mag`/`accel_min`/`speed_3d`/`t_s` by construction. `detect_exit`/`detect_freefall`/`detect_ground_climb`/`detect_phases` keep their signatures; older tests that build `Signals` by hand must add the `accel_min` field (called out in Tasks 3–4 Step 4).

**Known risk:** `signals.build_signals` (phases) and `telemetry.parse_telemetry_blob` (recon) both parse GPMF and now both must accumulate across payloads. They are separate implementations; a future refactor should share one GPMF-to-series reader. Until then, Task 1 fixes signals and Task 2 confirms recon already accumulates GPS5 across STRMs (it does) while fixing its GPSU handling.
