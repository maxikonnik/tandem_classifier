# Visual Detectors — Real-Data Calibration Findings

*Calibration of the visual-detectors branch on a real jump (Дмитрий `GX010015`, 205 keyframes over 204 s; known ground truth from the accelerometer work: exit ≈ 39 s, canopy ≈ 88–112 s).*

## Runtime fix (committed)

`extract_keyframes` used `-vsync 0`, which **ffmpeg 9.0 removed**, so extraction crashed. Replaced with `-fps_mode passthrough` (verified: one image per keyframe, no duplication). The pipeline now runs end-to-end on real footage.

## The metrics do NOT separate the phases as designed

| Signal | In aircraft (t=0–36 s) | Freefall/sky (t=40–96 s) | Canopy (t≈108–112 s) | Verdict |
|---|---|---|---|---|
| `structure_frac` | 0.02–0.045 | 0.01–0.02 | 0.02–0.10 | **Fails** — uniformly low, never reaches `EXIT_STRUCTURE_HI=0.25`; cannot mark "aircraft fills frame". |
| `mean_luma` | 89–120 (dark cabin) | 140–160 (bright sky) | ~140 | **Separates well** — clean jump at exit (~t=38). |
| `blob_area_frac` | 0.16–0.40 (noisy) | 0.02–0.11 | rises to 0.31 | High cabin noise; drops in freefall. |
| `center_fill_frac` | 0.11–0.48 (noisy) | 0.0–0.05 | **0.14 → 0.52** (real bloom) | Real canopy rise is visible but drowned by cabin false positives. |

**Detector results on real data:**
- `exit_by_background` → **None** (structure_frac never crosses the HI threshold). Exit undetected.
- `canopy_by_growth` → **false positive at t=2.0 s** (cabin center-fill jitter), not the real canopy at ~108–112 s.

## Why, and what it means

- The "aircraft fills the frame ⇒ high edge density (`structure_frac`)" proxy does not hold as measured (GoPro fisheye at 720p, `GRAD_THRESH=25`). Threshold tuning **cannot** rescue it — the aircraft and the sky both sit at ~0.02–0.04.
- The signals that *do* separate are **`mean_luma`** (dark cabin → bright sky at exit) and, more weakly, **`blob_area_frac`** (cabin clutter → small distant pair). The real **canopy** is a genuine large central bloom (`center_fill_frac` 0.14→0.52) but needs gating to the post-freefall window and a much larger, sustained rise with hysteresis to reject cabin jitter.

This is a **design finding, not a constant tweak**: the exit and canopy detectors need a metric rethink. Candidate redesigns:
- **Exit:** fuse `mean_luma` jump (dark→bright) with a proper "foreground-occupancy" / aircraft-leaves-frame metric — keeping the domain owner's caveat that a plain luma jump fails when the operator is already hanging outside the plane before exit (a different, structure/occupancy-based cue is needed for that case).
- **Canopy:** restrict `canopy_by_growth` to the window after the (telemetry) exit/freefall phase, and require a large sustained central-fill rise (toward the observed ~0.5) with hysteresis, rejecting the noisy cabin segment.

The unit-tested detector *code* (Tasks 1–4) is sound on synthetic series; it is the choice of image metrics and thresholds that must change to work on real footage. That choice is a detection-design decision for the domain owner.

## Resolution (real-video validated)

**Telemetry exit detector was broken on real data — fixed.** Real-video validation exposed that the merged `detect_exit` never fired on real jumps: the free-fall-onset |a| dip is noisy and oscillates above/below the raw 0.35 g threshold, so no continuous sub-threshold run of ≥1 s ever forms (longest ≈0.6 s). Fixed by thresholding the **smoothed** |a| envelope (1 s window, <0.7 g, jerk gate ≥0.8 g). Validated (in `main`, commit fccc5cd):
- Дмитрий GX010015 → exit 36.4 s, freefall (36.4, 90.2 s).
- 18 05 / GX010188 → exit 48.4 s, freefall (48.4, 111.6 s).
- Родионов ground-footage file → exit None (correct, no false positive).

**Canopy now works with the 20 s gate.** Per the domain owner, the tandem canopy is searched no earlier than exit + 20 s (`CANOPY_MIN_AFTER_EXIT_S`). With telemetry exit feeding the gate, the full pipeline validates on Дмитрий: **exit 36.4 s → canopy 112.1 s (confidence 1.0)** — the real bloom, cabin false positives eliminated.

**Still open (lower priority now):** the *visual* exit cue (aircraft leaves frame) — `structure_frac` doesn't separate on real footage. It matters only for recordings without usable telemetry (rare, since the accelerometer is almost always present and telemetry exit is now reliable). Redesign candidates (mean-luma jump / foreground-occupancy) remain a domain-owner decision, deferred.
