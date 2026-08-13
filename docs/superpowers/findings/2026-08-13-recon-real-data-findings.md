# Real-Data Reconnaissance Report — Tandem Scene Detection

*First run of the recon tool over a real archive of 8 jump folders (2026-08-13).*

**Context.** Archive: `Samples/` (17 folders, 8 with video). Tooling: `tandem/recon/` at
`main` (commit c713832), ffmpeg 9.0. Mechanical probes are from the tool; the visual
assessments (assumptions 3 and 6) come from vision agents that actually read the extracted
freefall keyframes of the three folders that produced frames. This report decides what gets
built next.

## 1. Verdict per assumption (1–8)

| # | Assumption | Real-data verdict |
|---|------------|-------------------|
| 1 | GPS UTC present | **FALSE for ~half the archive.** GPS present in only 3 of 8 folders (2, 4, 8), absent in 3 (5, 6, 7 — `has_gps:false`), undeterminable in 2 (1, 3, which produced 0 parsed recordings). Among parsed folders: **3 with GPS / 3 without = 50% GPS-less.** Where GPS *is* present, the UTC is not real (finding B). |
| 2 | Keyframe interval | **CONFIRMED and stable.** `keyframe_med_s = 1.001` in every folder that yielded recordings. ~1.0 s is a reliable fixed temporal resolution across cameras and dates. |
| 3 | Ground-background fraction in freefall | **Weakly measured, single data point.** Only jump 2 contained genuine freefall; there ground/snow occupies the lower part of frame (`ground_bg_fraction ≈ 0.3`), subject usually framed against sky. Jumps 4 and 8 report `1.0` but are **not freefall** (ground scenes) and must be excluded. Net: one usable jump, ~0.3, over only ~44 s of true freefall. |
| 4 | Canopy opening lands in operator frame | **NO FRAME EVIDENCE.** `canopy_open` was never detected; zero canopy frames sampled. The one incidental observation (jump 2) is a *counter-example*: after deployment only the jumper's own POV survives — **no companion-camera footage of the pair exists after the canopy opens**, so the cue may vanish exactly when the assumption needs it. |
| 5 | Naming / chunk convention | **FALSE — convention does not hold, and failures are silent.** Three violations: a 7-digit GoPro name `GX0120611.mp4` (folder 1, whole jump lost); a Panasonic `.MOV` invisible to the `*.MP4` glob (folder 3, whole jump lost, not even flagged); an unmatched DJI file `DJI_20260523143455_0009_D.MP4` (folder 5). |
| 6 | Operator distance to the pair | **Measured once; NOT constant.** In jump 2 the distance is bimodal within one window: the pair is ~10–15% of frame height at exit (t≈40–48 s), then a companion camera-flyer closes to fill 40–60%+ of frame (t≈64–88 s). A single "operator distance" prior is wrong. |
| 7 | Fraction of jumps with both cameras | **UNMEASURABLE this run.** Dual-camera pairing depends on GPS-UTC grouping (failed wherever GPS is absent) and the one clear second camera (DJI, folder 5) was left unmatched. No reliable fraction computable. |
| 8 | Tandem landing framing | **NO FRAME EVIDENCE.** No landing phase detected, no landing frames sampled. Cannot be assessed. |

## 2. Design-changing findings (ranked by impact)

**A. GPS is absent in ~50% of the archive, and its absence cascades into total failure, not degraded output.** Folders 5, 6, 7 have no GPS on the operator GoPros; all three produced `jumps:0, frames:0`. The chain: no GPS UTC → no cross-camera sync key → grouping cannot form a jump (`jumps=0`) → no freefall window → nothing sampled. Worse, the **physical phase detector itself depends on GPS**: exit is defined as an accelerometer dropout *followed by high GPS 3D speed*, so with no GPS-speed channel the detector's confirmation step cannot fire even though the accelerometer signal is present. GPS is currently a hard dependency of three independent subsystems (grouping, windowing, exit confirmation), and half the real archive doesn't satisfy it.

**B. Every GPS-bearing folder reports the identical firmware pre-fix default UTC of 2021-03-07.** Folders 2, 4, and 8 all report `first_utc` of `2021-03-07T00:00:02.xxxZ` — a firmware default, differing only in sub-second fractions. The "GPS UTC" grouping and t0 rely on is **not a real wall-clock anchor**: t0 is fictional, and any grouping keyed on absolute UTC would collapse footage from *different dates* (Mar 2021 default, May 2026, Oct 2024) onto the same nominal timestamp. Cross-camera sync via UTC is unsafe even where GPS "exists."

**C. Format/naming gaps silently delete whole jumps.** Two of eight folders (25%) yielded zero recordings purely from parser/glob narrowness: folder 1's only file has a 7-digit GoPro name and was unparsed; folder 3's only file is a Panasonic `.MOV`, invisible to a `*.MP4` glob — **not even flagged**. Folder 5's DJI second camera was dropped as unmatched. Silent loss is the dangerous property: a lost jump is indistinguishable from an empty folder.

**D. The freefall windows are frequently not freefall.** Of the three folders that produced frames, **only jump 2 contained any real freefall — and even there only ~44 s of a ~112 s window (~39%) was true freefall**, over-extended ~35–40 s into aircraft/door on the front and ~20–25 s into canopy/descent on the back. Jumps 4 and 8 were **100% mis-detected**: jump 4 is a person standing on the ground by a windsock; jump 8 is two men walking to the aircraft. So **2 of 3 sampled windows were entirely wrong, and the third was ~60% contaminated.** Any per-window image statistic computed this run is untrustworthy without window trimming. The mis-detections are consistent with a ground speed/accel signature being mistaken for an exit.

**E. Dual-camera fraction is unmeasurable**, blocking assumptions 4, 7, and 8. It is gated on both the GPS-UTC grouping (A/B) and the filename parser (C), both currently broken.

**F. The ~1 s keyframe interval is the one solid result** — and it fixes the achievable temporal resolution of everything downstream. Sub-second events (exit transient, opening shock) will need the accelerometer's native rate, not keyframes.

## 3. Recommended changes for the next plan(s) (prioritized)

1. **Make the accelerometer the primary phase signal; make GPS optional confirmation.** *(A, D.)* Re-detect exit/freefall/ground/climb from the accelerometer dropout + integration alone, treating GPS 3D speed as corroboration when present rather than a required gate. Single change that unblocks folders 5/6/7 (half the archive) and fixes the two fully-mis-detected windows.
2. **Add a non-GPS jump-grouping fallback.** *(A, B.)* When UTC is absent or the 2021-03-07 firmware default, group recordings by file mtime / recording order / chunk continuity. Detect and **quarantine the pre-fix default UTC** so it is never used as a real t0 or sync key.
3. **Fix GPSU-with-fix parsing and validate UTC sanity.** *(B.)* Parse real GPS time where a fix exists; flag any `first_utc` equal to the firmware default (2021-03-07) as invalid telemetry rather than a usable anchor.
4. **Broaden the filename parser and media glob; never drop a file silently.** *(C.)* Support variant GoPro names (7-digit), DJI names, and non-MP4 containers (`.MOV`, Panasonic `P*`); glob all video extensions. Any file that cannot be parsed or matched must be **emitted as an explicit `unmatched`/`unsupported` record**, never omitted.
5. **Calibrate freefall-window thresholds and trim windows.** *(D.)* Tighten the window to the confirmed freefall span (jump 2's true freefall was t≈40–84 s vs. the tool's 0–112 s). Front-trim (drop aircraft/door) and back-trim (drop canopy/descent) using the accelerometer opening-shock signature as the freefall-end boundary. Re-run assumptions 3 and 6 only on trimmed windows.
6. **Treat operator distance as bimodal, not constant.** *(6.)* Model exit-distance vs. approach-distance separately.
7. **Once grouping + parsing are fixed, add canopy/landing detection** to close assumptions 4 and 8, which currently have no frame evidence at all.

## 4. What is safe to build now vs. blocked

**Safe now:** the accelerometer-first phase detector (rec. 1), the broadened filename parser + all-extension glob with mandatory explicit flagging (4), the pre-fix-UTC quarantine guard (2/3), and window-trimming logic keyed to the accelerometer opening shock (5) — none depend on data the recon run failed to produce, and all are motivated by confirmed findings. **Blocked pending those fixes:** dual-camera fraction (7), canopy-in-operator-frame (4), and landing framing (8), because they require working non-GPS grouping and DJI/second-camera pairing before real frames exist to measure; and the ground-background-fraction and operator-distance priors (3, 6) should not be trusted until windows are re-detected and trimmed.

---

## Appendix A — mechanical results per folder

| # | Folder | Recordings | Unmatched | GPS | first_utc | keyframe s | jumps | frames |
|---|--------|-----------:|-----------|-----|-----------|-----------:|------:|-------:|
| 1 | 12 01 Ахмед | 0 | GX0120611.mp4 (7-digit) | NO_TELEMETRY | — | — | 0 | 0 |
| 2 | 16 03 24 Затяжной (Дмитрий) | 1 | — | ✅ | 2021-03-07 (pre-fix) | 1.001 | 1 | 57 |
| 3 | 17 03 24 Крис | 0 | *(none — .MOV invisible)* | NO_TELEMETRY | — | — | 0 | 0 |
| 4 | 18 05 | 9 | — | ✅ | 2021-03-07 (pre-fix) | 1.001 | 1 | 6 |
| 5 | 23 05 26 Лиза | 9 | DJI_…0009_D.MP4 | ❌ none | null | 1.001 | 0 | 0 |
| 6 | 23 05 26 Курносов | 5 | — | ❌ none | null | 1.001 | 0 | 0 |
| 7 | 25 07 Родионов | 12 | — | ❌ none | null | 1.001 | 0 | 0 |
| 8 | Сергей 6.10.24 | 8 | — | ✅ | 2021-03-07 (pre-fix) | 1.001 | 1 | 17 |

## Appendix B — vision assessment of sampled "freefall" windows

- **jump 2 (Дмитрий)** — `window_is_really_freefall: false`. Mixed: t=0–36 s aircraft interior; **t=40–84 s genuine freefall** (tandem pair + drogue visible); t=88 s main canopy already open; t=92–112 s under-canopy POV over a snow-covered village. Pair scale bimodal (small dot at exit → fills 40–60% as a camera-flyer closes in). After deployment only the jumper's own POV survives.
- **jump 4 (18 05)** — `window_is_really_freefall: false`. All frames: a person in harness standing still on a grass airfield by an inflated windsock; a distant open canopy (someone else) in the far sky. Ground scene, not freefall.
- **jump 8 (Сергей)** — `window_is_really_freefall: false`. All frames: two men walking to the aircraft on the ground (drop-zone tent, vehicles, fence, tower in background). Pre-jump ground footage, not freefall.
