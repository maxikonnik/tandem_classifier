# Telemetry Motion Features & Operator-Behaviour Detectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** From the operator camera's accelerometer, derive two motion features — per-axis acceleration and the rolling oscillation amplitude (rolling std of |a|) — and use them to (1) corroborate the freefall window, (2) detect the operator's fly-around of the pair ("облёт"), and (3) mark the moment the operator breaks off tracking and turns away.

**Architecture:** Extend `tandem/phases/signals.py` (per-axis + rolling-std) and add `tandem/phases/motion.py` with the three detectors. Pure logic; **validation is on the real `Samples/` videos, not synthetic fixtures** (per the project's real-video testing rule). Thresholds are grounded in a hand-verified real jump and stay named constants for calibration across jumps.

**Tech Stack:** Python 3.10+, numpy, pytest. Reuses `tandem/recon/gpmf.py` and `tandem/phases/`. No new deps.

## Global Constraints

- **Python 3.10**; `from __future__ import annotations`; no 3.11+ syntax.
- **Real videos are the source of truth.** Validate every detector by running it on real `Samples/` recordings and checking against known ground truth (accelerometer exit/freefall, and the hand-verified events in `GX012255`). Keep only minimal pure-helper unit tests; do not build elaborate synthetic fixtures to "prove" a detector — a synthetic pass means nothing here (the visual detectors passed synthetic tests and failed on real footage).
- **Operator-camera telemetry only.** These are the external operator's motion cues.
- **`source="telemetry"`**; recording-relative seconds.
- **Mounting-agnostic where possible.** The camera's axis orientation varies between mounts, so a detector must not hard-code "axis 2" — key off the axis that deviates most from its own freefall baseline (or, as a refinement, the gyroscope rotation rate).
- **Thresholds provisional**, grounded in `GX012255` (below); calibrate across jumps.

## Design notes — grounded in a hand-verified real jump (GX012255, 105 s, ~201 Hz)

Rolling std of |a| over a ~1 s window ("oscillation amplitude") cleanly separates the phases; per-axis means reveal reorientation:

| Phase | window | std\|a\| | per-axis |
|---|---|---|---|
| In aircraft | 7–36 s | ~0.02 | one axis ≈ −1 g (gravity), steady |
| Exit | 37 s | jumps 0.09→0.29, \|a\|→0.46 | — |
| Freefall | 47–89 s | **0.3–0.6** (sustained high) | noisy |
| **Облёт** (fly-around) | 65–70 s | **~0.13** (a dip within freefall) | one axis rises smoothly (0.16→0.72) |
| **Break-off** (turns away) | 90–95 s | — | one axis large excursion (−0.59→+1.42) |
| Operator canopy | 96–99 s | — | \|a\| 2.0–2.2 g (opening shock) |

So: **freefall = high rolling std; облёт = a low-std dip inside freefall (smooth coordinated arc); break-off = a large per-axis excursion after freefall, just before the operator's own opening shock.**

## File Structure

```
tandem/phases/signals.py   # MODIFY: add per-axis series (ax/ay/az) + rolling_std(|a|)
tandem/phases/motion.py    # NEW: freefall_std_ok, detect_orbit, detect_breakoff
tests/phases/test_motion.py
scripts/ (none) — validation is via real-video runs, recorded in a findings note
```

---

## Task 1: Per-axis acceleration + rolling-std feature

**Files:** Modify `tandem/phases/signals.py`; Test `tests/phases/test_signals.py` (append minimal).

**Interfaces:**
- `Signals` gains `ax: list[float]`, `ay: list[float]`, `az: list[float]` (per-axis, g, on the same grid as `accel_mag`), and `accel_std: list[float]` (rolling std of |a| over `STD_WINDOW_S`, same grid).
- `build_signals` accumulates the 3 ACCL axes across all payloads (SCAL-corrected, in g) and resamples them to the grid; computes `accel_std` from the resampled |a|.
- `rolling_std(values, win) -> list[float]` (pure).

- [ ] **Step 1: minimal pure test** — `rolling_std([1,1,1,5,5,5], 3)` has a low value in the flat region and a high value spanning the step; `build_signals` on a 2-payload synthetic blob yields `ax/ay/az/accel_std` all length == `len(accel_mag)`. (One small test; the real check is Task 4.)
- [ ] **Step 2: run, see fail.**
- [ ] **Step 3: implement.** Extend `_accel_*` parsing to return the 3 axes; add `STD_WINDOW_S = 1.0`; add `rolling_std`; populate the new fields (zero-filled length `n_out` when accel absent, preserving the alignment invariant).
- [ ] **Step 4: run, green** (`python -m pytest -q`).
- [ ] **Step 5: commit** `feat(phases): per-axis accel + rolling-std oscillation feature`.

---

## Task 2: Freefall-std corroboration & the orbit (облёт) detector

**Files:** Create `tandem/phases/motion.py`; Test `tests/phases/test_motion.py`.

**Interfaces:**
- Constants `FREEFALL_STD_MIN = 0.2` (freefall shows rolling std above this), `ORBIT_STD_MAX = 0.18`, `ORBIT_MIN_DURATION_S = 3.0`.
- `freefall_std_ok(sig, freefall: Segment) -> bool` — mean `accel_std` inside the freefall window exceeds `FREEFALL_STD_MIN` (corroborates the accel-first freefall).
- `detect_orbit(sig, freefall: Segment) -> list[Segment]` — within the freefall window, contiguous runs of `accel_std < ORBIT_STD_MAX` lasting ≥ `ORBIT_MIN_DURATION_S` become `type="orbit"` highlight-candidate segments (`source="telemetry"`).

- [ ] **Step 1: minimal pure test** on a hand-built `Signals` (freefall high-std with a low-std dip) → `freefall_std_ok` True; `detect_orbit` returns one segment spanning the dip.
- [ ] **Step 2–4:** fail → implement → green.
- [ ] **Step 5: commit** `feat(phases): freefall-std corroboration and orbit (облёт) detector`.

---

## Task 3: Operator break-off (end-of-tracking) detector

**Files:** Modify `tandem/phases/motion.py`; Test `tests/phases/test_motion.py`.

**Interfaces:**
- Constants `BREAKOFF_AXIS_EXCURSION_G = 0.8`, `BREAKOFF_BASELINE_S = 5.0`.
- `detect_breakoff(sig, freefall: Segment) -> Event | None` — after the freefall window, the first time ANY axis (ax/ay/az) deviates from its freefall-baseline mean by more than `BREAKOFF_AXIS_EXCURSION_G` (mounting-agnostic: pick the axis with the largest deviation). Emits `Event(type="operator_breakoff", t_s, "telemetry", conf)`. This marks the end of the useful pair-tracking footage.

- [ ] **Step 1: minimal pure test** — a `Signals` where after freefall one axis swings by ~1.5 g → break-off detected at the swing; no swing → None.
- [ ] **Step 2–4:** fail → implement → green.
- [ ] **Step 5: commit** `feat(phases): operator break-off (end-of-tracking) detector`.

---

## Task 4: Real-video validation (the actual proof)

Not a unit-test task — the real validation, per the project rule.

- [ ] **Step 1:** Run the pipeline on `Samples/08 02/Видео/GX012255.MP4`: build signals, `detect_phases` (exit/freefall), then `freefall_std_ok`, `detect_orbit`, `detect_breakoff`. Confirm against the hand-verified truth: exit ~37 s, freefall ~37–90 s, orbit ~65–70 s, break-off ~90–95 s.
- [ ] **Step 2:** Run on ≥4 more real recordings that contain freefall (e.g. Дмитрий GX010015, 18 05/GX010188, and two from other folders). For each, print exit/freefall/orbit/break-off times and sanity-check them (freefall std high; orbit = a low-std dip if present; break-off near the end before the operator's opening shock). Finding 2 ("verify on other videos") is discharged here.
- [ ] **Step 3:** Tune the constants (`FREEFALL_STD_MIN`, `ORBIT_STD_MAX`, `ORBIT_MIN_DURATION_S`, `BREAKOFF_AXIS_EXCURSION_G`) if a jump disagrees; record the final values and the per-jump results in `docs/superpowers/findings/2026-08-13-motion-detectors-validation.md`.
- [ ] **Step 4: commit** the calibrated constants + the validation note.

---

## Self-Review

**Findings coverage:**
- Finding 2 (freefall = high freq+amplitude) → Task 1 (`accel_std`) + Task 2 (`freefall_std_ok`), validated Task 4.
- Finding 3 (облёт = low amplitude) → Task 2 (`detect_orbit`), validated Task 4. Feeds the §5.5 "облёт пары оператором" highlight subcategory.
- Finding 1 (break-off = per-axis excursion) → Task 1 (per-axis) + Task 3 (`detect_breakoff`), validated Task 4.

**Out of scope (later):** the visual облёт confirmation (blob trajectory + background parallax, §5.2); using the **gyroscope** for a fully rotation-based break-off (a refinement over per-axis accel); wiring these events into the scenes.json contract and the highlights ranking; the still-open visual *exit* metric.

**Real-video-first:** per the project rule, synthetic unit tests here only sanity-check the pure math (`rolling_std`, alignment). The detectors are proven or rejected by Task 4's runs over real `Samples/` footage; a green synthetic suite is necessary but NOT sufficient.

**Type consistency:** `Signals` gains `ax/ay/az/accel_std` all equal-length with `accel_mag`/`t_s`; `motion.py` consumes `Segment`/`Event` from `detect.py`. Constants named, grounded in `GX012255`.
