# Telemetry Motion Detectors — Real-Data Validation

*Validated on real `Samples/` jumps. Ground truth for `GX012255` (hand-verified against video): exit ~37 s, freefall ~37–90 s, облёт ~65–70 s, operator turn-away ~90–95 s, operator canopy ~96–99 s.*

## Results (motion pipeline: detect_phases → freefall_std_ok, detect_orbit, detect_breakoff)

| Recording | exit | freefall | freefall_std_ok | orbit (облёт) | breakoff (turn-away) |
|---|---|---|---|---|---|
| GX012255 (ground truth) | 38 | 38–97 | True | **(67,71)** ✓ (truth 65–70) | **92** ✓ (truth 90–95) |
| Дмитрий GX010015 | 36 | 36–90 | True | none | none |
| 18 05 / GX010188 | 48 | 48–112 | True | none | 107 (before deploy 112) |
| Сергей GX012041 | — | no-freefall | — | — | — |
| Курносов GX012330 | 72 | 72–136 (no GPS) | True | — | 136 |

Дмитрий's operator was a close-tracking camera-flyer (no smooth orbit, no sharp turn-away) → orbit/breakoff `none`, which is correct.

## Bugs the real-video validation caught (that synthetic tests + code review did NOT)

1. **`accel_std` was in m/s², thresholds in g.** `freefall_std_ok` passed trivially and `detect_orbit` never fired (std ~0.84 m/s² never < 0.18). Fixed: `accel_std` computed on `|a|/G`. In g the phases separate cleanly (aircraft ~0.02, freefall 0.3–0.6, облёт ~0.09–0.13).
2. **Orbit false-positive at freefall onset.** The first ~13 s after exit have low std (|a| rising smoothly toward terminal velocity, not yet buffeting). Fixed: `ORBIT_MIN_AFTER_EXIT_S = 15` — search only established freefall.
3. **Break-off landed on the operator's opening shock, not the turn-away.** `detect_freefall` ends the window at the operator's own opening shock (~97), and the turn-away (~90) happens *inside* the window, so scanning "after freefall" missed it. Fixed: break-off is now a **sustained axis excursion in the late freefall window** (`freefall.end − BREAKOFF_SEARCH_LATE_S … freefall.end`), from an established-freefall baseline. Restricting to the late window also keeps a mid-freefall orbit (which swings an axis but returns) from firing.

## Calibrated constants (grounded in real footage)

- `STD_WINDOW_S = 1.0`; `accel_std` in g.
- `FREEFALL_STD_MIN = 0.2`, `ORBIT_STD_MAX = 0.18`, `ORBIT_MIN_DURATION_S = 3.0`, `ORBIT_MIN_AFTER_EXIT_S = 15.0`.
- `BREAKOFF_AXIS_EXCURSION_G = 0.8`, `BREAKOFF_BASELINE_S = 5.0`, `BREAKOFF_SEARCH_LATE_S = 20.0`, `BREAKOFF_SUSTAIN_S = 1.0`.

## Notes / follow-ups
- Thresholds are calibrated on a few jumps; widen the validation set as more hand-verified footage is available.
- Break-off is mounting-agnostic (largest-deviation axis). A gyroscope-based turn-away (rotation rate) is a future robustness refinement.
- Feeds the scene model: freefall window (with облёт highlight segments) and the break-off = end of useful pair-tracking footage.
