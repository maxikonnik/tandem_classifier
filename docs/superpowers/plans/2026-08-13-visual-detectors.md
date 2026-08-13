# Visual Detectors — Exit-by-Background & Canopy-by-Growth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a recording's 720p keyframes, produce two visual event cues the telemetry cannot give reliably — **exit** (the aircraft that fills the frame rapidly leaves) and the **tandem canopy opening** (the pair's canopy blooms, growing sharply near the frame centre) — as candidate times to be fused with telemetry later.

**Architecture:** A new `tandem/visual/` subpackage. `features.py` reduces one grayscale keyframe to a few scalar features (brightness, contrast-blob area, central-fill fraction, structure/edge density). `series.py` runs that over a recording's keyframes into a time-ordered series. `detect.py` turns the series into an exit cue (sharp drop in structure density = plane leaves frame) and a canopy cue (sharp rise in central fill = object grows at centre). Pure image/series logic (numpy + Pillow), unit-testable on synthetic images and series; provisional thresholds are calibrated against a real jump whose exit/canopy times are already known from the accelerometer work.

**Tech Stack:** Python 3.10+, numpy, Pillow (both already deps). Reuses the recon keyframe extraction (`-skip_frame nokey`, 720p). No new third-party deps, no ML model in this plan.

## Global Constraints

- **Python floor 3.10**; `from __future__ import annotations`; no 3.11+ syntax.
- **Consume 720p keyframes** at the ~1 s cadence the real archive confirmed (assumption 2: `keyframe_med_s ≈ 1.001`). Temporal resolution of every visual cue is therefore ~1 s.
- **Every emitted cue carries `source="visual"`.** Times are recording-relative seconds.
- **Operator ≠ tandem.** The canopy cue is the *tandem pair's* canopy seen in the operator's frame (object growing at centre) — never the operator's own accel opening-shock.
- **Scope: external operator camera only.** These cues (aircraft-leaving-frame exit, object-growth canopy) are tuned to the external camera, whose interview/freefall/landing are *separate recordings*. The instructor's hand-cam records the whole jump as one continuous file and needs within-file segmentation — a **separate plan**, not covered here.
- **Exit visual cue = background change, not brightness.** Per the domain owner: before exit the aircraft fills most of the frame; at exit it rapidly leaves. This is detected from frame **structure/edge density**, which works even when the operator has already climbed outside the plane (where an exposure/brightness jump fails). Brightness is kept only as a secondary corroborator.
- **Thresholds are provisional** and must be calibrated on real keyframes (Task 5) — keep them named constants.

## Design notes (read before Task 1)

- **Contrast-blob (spec §5.2):** in the sky the tandem pair is the single dark, contrasting object. So a threshold on darkness relative to the frame's own brightness isolates it — no object detector needed. `blob_area_frac` = fraction of contrasting-dark pixels; `center_fill_frac` = that fraction inside the central box.
- **Canopy opening = growth at centre:** as the pair's canopy deploys and blooms, the contrasting object grows sharply and sits near the frame centre. So a sharp **rise** in `center_fill_frac` marks it.
- **Exit = aircraft leaves the frame:** the aircraft (fuselage/wing, close up) is highly structured and fills the frame; open sky after exit is smooth. So a sharp **drop** in `structure_frac` (fraction of high-gradient pixels) marks exit — independent of brightness, so it survives the operator-already-outside case.
- **Known ground truth for calibration:** for Дмитрий `GX010015`, the accelerometer work established exit at t≈39 s and the canopy region near t≈88 s; Task 5 checks the visual cues fire there.

## File Structure

```
tandem/visual/__init__.py
tandem/visual/features.py   # FrameFeature; frame_features(gray); load_gray(path)
tandem/visual/series.py     # extract_keyframes(video, out) + build_series(frame_dir) -> list[FrameFeature]
tandem/visual/detect.py     # exit_by_background(series); canopy_by_growth(series)
tests/visual/__init__.py
tests/visual/test_features.py
tests/visual/test_detect.py
tests/visual/test_series.py
```

---

## Task 1: Per-frame features

Reduce one grayscale keyframe to scalar features.

**Files:**
- Create: `tandem/visual/__init__.py`, `tandem/visual/features.py`
- Create: `tests/visual/__init__.py`, `tests/visual/test_features.py`

**Interfaces:**
- Produces:
  - Constants `DARK_REL = 0.5` (a pixel is "contrasting-dark" below this × frame mean luma), `GRAD_THRESH = 25.0`, `CENTER_FRAC = 0.4`.
  - `FrameFeature` dataclass: `t_s: float`, `mean_luma: float`, `blob_area_frac: float`, `center_fill_frac: float`, `structure_frac: float`.
  - `frame_features(gray: "np.ndarray", t_s: float = 0.0) -> FrameFeature` (pure) — `gray` is a 2-D uint8/float array.

- [ ] **Step 1: Write the failing test**

`tests/visual/test_features.py`:
```python
import numpy as np
from tandem.visual.features import frame_features, FrameFeature


def test_bright_smooth_frame_has_low_blob_and_structure():
    gray = np.full((90, 160), 220, dtype=np.uint8)   # uniform bright sky
    f = frame_features(gray, t_s=1.0)
    assert f.t_s == 1.0
    assert f.blob_area_frac == 0.0          # nothing dark
    assert f.structure_frac == 0.0          # no edges
    assert f.mean_luma > 200


def test_central_dark_object_fills_centre():
    gray = np.full((100, 100), 220, dtype=np.uint8)
    gray[40:60, 40:60] = 10                 # dark 20x20 object at centre
    f = frame_features(gray)
    assert f.blob_area_frac > 0.0
    assert f.center_fill_frac > f.blob_area_frac   # concentrated at centre
    assert f.structure_frac > 0.0                  # its edges register


def test_structured_frame_has_high_structure_frac():
    checker = np.zeros((100, 100), dtype=np.uint8)
    checker[::2, :] = 255                    # alternating rows -> many edges
    f = frame_features(checker)
    assert f.structure_frac > 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/visual/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.visual.features'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/visual/features.py`:
```python
"""Reduce one grayscale keyframe to scalar visual features.

blob = the dark, contrasting object (the tandem pair against bright sky).
structure = fraction of high-gradient pixels (the close-up aircraft is
highly structured; open sky is smooth). No object detector, no ML.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DARK_REL = 0.5        # contrasting-dark: below DARK_REL * frame mean luma
GRAD_THRESH = 25.0    # a pixel is an "edge" if its gradient magnitude exceeds this
CENTER_FRAC = 0.4     # central box side, as a fraction of frame


@dataclass
class FrameFeature:
    t_s: float
    mean_luma: float
    blob_area_frac: float
    center_fill_frac: float
    structure_frac: float


def frame_features(gray, t_s: float = 0.0) -> FrameFeature:
    g = np.asarray(gray, dtype=np.float32)
    mean_luma = float(g.mean())
    dark = g < (mean_luma * DARK_REL)
    blob_area_frac = float(dark.mean())

    h, w = g.shape
    y0, y1 = int(h * (0.5 - CENTER_FRAC / 2)), int(h * (0.5 + CENTER_FRAC / 2))
    x0, x1 = int(w * (0.5 - CENTER_FRAC / 2)), int(w * (0.5 + CENTER_FRAC / 2))
    center = dark[y0:y1, x0:x1]
    center_fill_frac = float(center.mean()) if center.size else 0.0

    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    edge_frac = 0.5 * (float((gx > GRAD_THRESH).mean()) + float((gy > GRAD_THRESH).mean()))
    return FrameFeature(t_s=t_s, mean_luma=mean_luma, blob_area_frac=blob_area_frac,
                        center_fill_frac=center_fill_frac, structure_frac=edge_frac)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/visual/test_features.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/visual/__init__.py tandem/visual/features.py tests/visual/__init__.py tests/visual/test_features.py
git commit -m "feat(visual): per-frame features (blob, centre-fill, structure)"
```

---

## Task 2: Keyframe extraction and feature series

Extract all of a recording's keyframes at 720p and build a time-ordered `FrameFeature` series.

**Files:**
- Create: `tandem/visual/series.py`
- Test: `tests/visual/test_series.py`

**Interfaces:**
- Consumes: `features.frame_features`, `features.load_gray` (new — Pillow load → grayscale numpy).
- Produces:
  - `extract_keyframes(video: str, out_dir: str, fps_cap: float | None = None) -> list[str]` — `ffmpeg -skip_frame nokey -vf scale=-2:720`, one JPEG per keyframe named `kf_<index>.jpg`; returns written paths. Records the keyframe timestamps to `out_dir/timestamps.txt`.
  - `build_series(frame_dir: str) -> list[FrameFeature]` — load each `kf_*.jpg` in order, tag with its timestamp (from `timestamps.txt` if present, else index seconds), return the series.
  - `load_gray(path: str) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

`tests/visual/test_series.py`:
```python
import numpy as np
from PIL import Image
from tandem.visual.series import build_series, load_gray


def test_load_gray_returns_2d(tmp_path):
    p = tmp_path / "x.jpg"
    Image.fromarray(np.full((40, 60, 3), 128, np.uint8)).save(p)
    g = load_gray(str(p))
    assert g.ndim == 2 and g.shape == (40, 60)


def test_build_series_orders_by_index(tmp_path):
    for i, val in [(0, 220), (1, 10), (2, 220)]:
        Image.fromarray(np.full((50, 50, 3), val, np.uint8)).save(tmp_path / f"kf_{i}.jpg")
    series = build_series(str(tmp_path))
    assert [round(f.t_s) for f in series] == [0, 1, 2]
    assert series[1].blob_area_frac == 0.0   # a uniform (dark) frame has no *relative* dark pixels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/visual/test_series.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.visual.series'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/visual/series.py`:
```python
"""Extract 720p keyframes and build a FrameFeature series."""
from __future__ import annotations

import glob
import os
import subprocess

import numpy as np
from PIL import Image

from tandem.visual.features import FrameFeature, frame_features


def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def extract_keyframes(video: str, out_dir: str, fps_cap=None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-skip_frame", "nokey", "-i", video,
         "-vsync", "0", "-frame_pts", "1", "-vf", "scale=-2:720",
         os.path.join(out_dir, "kf_%06d.jpg")],
        check=True, capture_output=True,
    )
    return sorted(glob.glob(os.path.join(out_dir, "kf_*.jpg")))


def _index_of(path: str) -> int:
    base = os.path.basename(path)
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else 0


def build_series(frame_dir: str) -> list[FrameFeature]:
    ts_path = os.path.join(frame_dir, "timestamps.txt")
    stamps = {}
    if os.path.exists(ts_path):
        with open(ts_path) as fh:
            for line in fh:
                idx, t = line.split()
                stamps[int(idx)] = float(t)
    series = []
    for path in sorted(glob.glob(os.path.join(frame_dir, "kf_*.jpg")), key=_index_of):
        idx = _index_of(path)
        t_s = stamps.get(idx, float(idx))
        series.append(frame_features(load_gray(path), t_s=t_s))
    return series
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/visual/test_series.py -v`
Expected: PASS. (`extract_keyframes` shells out to ffmpeg and is exercised in Task 5, not unit-tested here.)

- [ ] **Step 5: Commit**

```bash
git add tandem/visual/series.py tests/visual/test_series.py
git commit -m "feat(visual): keyframe extraction and feature series"
```

---

## Task 3: Exit cue — aircraft leaves the frame

Detect exit as the sharp drop in `structure_frac`: the close-up aircraft (highly structured, fills the frame) gives way to open sky.

**Files:**
- Modify: `tandem/visual/detect.py` (create)
- Test: `tests/visual/test_detect.py`

**Interfaces:**
- Produces: constants `EXIT_STRUCTURE_HI = 0.25`, `EXIT_STRUCTURE_LO = 0.10`, `EXIT_DROP_WINDOW_S = 3.0`. `exit_by_background(series) -> VisualCue | None`, where `VisualCue` has `type: str`, `t_s: float`, `source: str = "visual"`, `confidence: float`. The exit is the first time `structure_frac` falls from above `EXIT_STRUCTURE_HI` to below `EXIT_STRUCTURE_LO` within `EXIT_DROP_WINDOW_S`.

- [ ] **Step 1: Write the failing test**

`tests/visual/test_detect.py`:
```python
from tandem.visual.features import FrameFeature
from tandem.visual.detect import exit_by_background, VisualCue


def _feat(t, structure=0.0, center=0.0, luma=200.0):
    return FrameFeature(t_s=t, mean_luma=luma, blob_area_frac=center,
                        center_fill_frac=center, structure_frac=structure)


def test_exit_at_structure_drop():
    # 10 s of structured aircraft interior, then open sky
    series = [_feat(t, structure=0.4) for t in range(10)]
    series += [_feat(t, structure=0.03) for t in range(10, 20)]
    cue = exit_by_background(series)
    assert cue is not None and cue.type == "exit" and cue.source == "visual"
    assert 9.0 <= cue.t_s <= 12.0


def test_no_exit_when_always_open():
    series = [_feat(t, structure=0.02) for t in range(20)]
    assert exit_by_background(series) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/visual/test_detect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tandem.visual.detect'`.

- [ ] **Step 3: Write minimal implementation**

`tandem/visual/detect.py`:
```python
"""Visual event cues from a FrameFeature series.

exit  = aircraft leaves the frame: structure_frac drops sharply.
canopy = tandem canopy blooms: center_fill_frac rises sharply.
Every cue is source="visual".
"""
from __future__ import annotations

from dataclasses import dataclass

EXIT_STRUCTURE_HI = 0.25
EXIT_STRUCTURE_LO = 0.10
EXIT_DROP_WINDOW_S = 3.0


@dataclass
class VisualCue:
    type: str
    t_s: float
    source: str
    confidence: float


def exit_by_background(series):
    if not series:
        return None
    for i in range(len(series)):
        if series[i].structure_frac < EXIT_STRUCTURE_HI:
            continue
        # look ahead within the drop window for a fall below LO
        for j in range(i + 1, len(series)):
            if series[j].t_s - series[i].t_s > EXIT_DROP_WINDOW_S:
                break
            if series[j].structure_frac < EXIT_STRUCTURE_LO:
                drop = series[i].structure_frac - series[j].structure_frac
                conf = max(0.0, min(1.0, drop / EXIT_STRUCTURE_HI))
                return VisualCue("exit", series[j].t_s, "visual", round(conf, 3))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/visual/test_detect.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tandem/visual/detect.py tests/visual/test_detect.py
git commit -m "feat(visual): exit cue from aircraft leaving the frame"
```

---

## Task 4: Canopy cue — object grows at the centre

Detect the tandem canopy opening as the sharp rise in `center_fill_frac`: the pair's canopy blooms and grows near the frame centre.

**Files:**
- Modify: `tandem/visual/detect.py`
- Test: `tests/visual/test_detect.py`

**Interfaces:**
- Produces: constants `CANOPY_RISE = 0.15`, `CANOPY_RISE_WINDOW_S = 3.0`. `canopy_by_growth(series, after_s: float = 0.0) -> VisualCue | None` — the first time (at or after `after_s`) `center_fill_frac` rises by at least `CANOPY_RISE` within `CANOPY_RISE_WINDOW_S`. `after_s` lets a caller restrict the search to after exit.

- [ ] **Step 1: Write the failing test**

Add to `tests/visual/test_detect.py`:
```python
from tandem.visual.detect import canopy_by_growth


def test_canopy_at_central_growth():
    series = [_feat(t, center=0.02) for t in range(10)]     # small distant pair
    series += [_feat(t, center=0.35) for t in range(10, 20)]  # canopy blooms at centre
    cue = canopy_by_growth(series)
    assert cue is not None and cue.type == "canopy_open" and cue.source == "visual"
    assert 9.0 <= cue.t_s <= 12.0


def test_after_s_restricts_search():
    series = [_feat(t, center=0.02) for t in range(10)]
    series += [_feat(t, center=0.35) for t in range(10, 20)]
    assert canopy_by_growth(series, after_s=15.0) is None   # growth is before 15 s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/visual/test_detect.py -v`
Expected: FAIL — `ImportError: cannot import name 'canopy_by_growth'`.

- [ ] **Step 3: Write minimal implementation**

Append to `tandem/visual/detect.py`:
```python
CANOPY_RISE = 0.15
CANOPY_RISE_WINDOW_S = 3.0


def canopy_by_growth(series, after_s: float = 0.0):
    for i in range(len(series)):
        if series[i].t_s < after_s:
            continue
        base = series[i].center_fill_frac
        for j in range(i + 1, len(series)):
            if series[j].t_s - series[i].t_s > CANOPY_RISE_WINDOW_S:
                break
            if series[j].center_fill_frac - base >= CANOPY_RISE:
                conf = max(0.0, min(1.0, (series[j].center_fill_frac - base) / (2 * CANOPY_RISE)))
                return VisualCue("canopy_open", series[j].t_s, "visual", round(conf, 3))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/visual/test_detect.py -v`
Then `python -m pytest -q` — whole suite green.

- [ ] **Step 5: Commit**

```bash
git add tandem/visual/detect.py tests/visual/test_detect.py
git commit -m "feat(visual): canopy cue from object growth at the frame centre"
```

---

## Task 5: Calibrate on a real jump (validation)

Not a unit-test task — the payoff. Run the pipeline on a recording whose exit/canopy are already known from the accelerometer, and confirm the visual cues fire there; tune the four thresholds if needed.

- [ ] **Step 1** — Extract all keyframes of Дмитрий `GX010015.MP4` (720p) with `extract_keyframes`, into a scratch dir. (ffmpeg must be on PATH.)
- [ ] **Step 2** — `build_series` over them; print `structure_frac` and `center_fill_frac` vs `t_s` (e.g. every ~2 s).
- [ ] **Step 3** — Check: does `structure_frac` drop sharply near **t≈39 s** (exit — aircraft leaves frame), and does `center_fill_frac` rise sharply near **t≈88 s** (tandem canopy)? Run `exit_by_background` and `canopy_by_growth(after_s=exit_t)` and confirm the returned times land near 39 s and 88 s.
- [ ] **Step 4** — If a cue is off, adjust the relevant constant (`EXIT_STRUCTURE_HI/LO`, `CANOPY_RISE`, `DARK_REL`, `GRAD_THRESH`) and re-run. Record the final values and the observed real numbers in a short note under `docs/superpowers/findings/`.
- [ ] **Step 5** — Commit the calibrated constants and the note.

```bash
git add tandem/visual/features.py tandem/visual/detect.py docs/superpowers/findings/
git commit -m "chore(visual): calibrate visual thresholds on a real jump"
```

---

## Self-Review

**Coverage of the directives / spec:**
- Exit by **background change** (aircraft leaves frame), robust to operator-already-outside → Task 3 (`structure_frac` drop). Brightness kept only as a secondary field (`mean_luma`), not the trigger.
- Tandem **canopy opening** by object **growth at centre** → Task 4 (`center_fill_frac` rise). Emits `source="visual"`, never the operator's accel opening-shock.
- Contrast-blob segmentation (§5.2) → Task 1 (`blob_area_frac`, `center_fill_frac`).
- Grounded/validated on a real jump with known event times → Task 5.

**Out of scope (later plans):**
- **Fusion & assembly:** combine this plan's visual exit cue with the telemetry jerk exit (accel-first plan), and corroborate the canopy cue with a **head-jerk** accelerometer bump. Fusion belongs with the scene-assembly layer once both signal sources exist.
- **Landing (§5.3):** semantic detection on cluttered ground background needs embeddings — a separate plan (the spec's acknowledged weakest point).
- **Interview (VAD), highlights + trained head, full `features.parquet`, ground-background fallback (`GROUND_BACKGROUND`)** — separate plans.
- **Ground-background degradation:** when the freefall background is earth rather than sky (assumption 3), the dark-relative blob threshold weakens; `GROUND_BACKGROUND` handling and an embedding fallback are deferred.

**Placeholder scan:** each code step carries real code; each test step a runnable command and expected result. Thresholds are named constants, calibrated in Task 5.

**Type consistency:** `FrameFeature` (Task 1) flows unchanged through `series.py` (Task 2) and `detect.py` (Tasks 3–4). `VisualCue` is defined once in `detect.py`. `exit_by_background`/`canopy_by_growth` keep the signatures declared in their Interfaces blocks.

**Known real-data risk:** the visual metrics are proxies (dark-relative blob, edge-density structure) and were reasoned, not yet measured on frames — Task 5 is where they meet reality, exactly as the accelerometer plan's thresholds did. If `structure_frac` does not separate aircraft-interior from open-sky on real keyframes, the fallback is a foreground-occupancy metric (largest non-sky connected region), noted here so the calibrator has a next move.
