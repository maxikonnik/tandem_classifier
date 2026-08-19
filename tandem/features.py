"""Per-jump feature table: every Signals channel on the 10 Hz grid.

One dense pass over a recording produces a flat table (one row per 10 Hz sample)
with all available telemetry and camera-metadata features. This is the Stage-0
foundation: the annotation tool and any downstream model read these tables instead
of re-parsing GPMF, so experiments run in seconds. Missing streams are written as
zeros; `has_*` flags on the Signals object say which were actually present.
"""
from __future__ import annotations

import csv
import os
import sys

# Column order of the exported table.
FEATURE_COLUMNS = [
    "t_s",
    "accel_mag", "accel_min", "accel_std", "ax", "ay", "az",
    "gyro_mag", "gx", "gy", "gz",
    "speed_3d",
    "iso", "shutter",
    "face_count", "smile", "blink", "face_area",
    "audio_level", "scene_indoor",
]


def signals_to_rows(sig) -> list[dict]:
    """Flatten a Signals object into one dict per 10 Hz sample."""
    n = len(sig.t_s)

    def col(name: str) -> list[float]:
        v = getattr(sig, name, None)
        return v if (v is not None and len(v) == n) else [0.0] * n

    cols = {c: col(c) for c in FEATURE_COLUMNS}
    return [{c: round(float(cols[c][i]), 5) for c in FEATURE_COLUMNS} for i in range(n)]


def write_features_csv(sig, path: str) -> int:
    """Write the feature table to CSV; returns the number of rows."""
    rows = signals_to_rows(sig)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


VIDEO_EXTS = (".mp4", ".mov")


def _safe_name(rel: str) -> str:
    """A filesystem-safe CSV stem for a recording's relative path (keeps it unique
    across folders that reuse GoPro file numbers)."""
    return rel.replace("\\", "__").replace("/", "__").replace(" ", "_")


def _collect_videos(path: str) -> list[tuple[str, str]]:
    """Return (abspath, rel-name) for each video under `path` (file or directory)."""
    if os.path.isfile(path):
        return [(path, os.path.basename(path))]
    out = []
    for dp, _dirs, fns in os.walk(path):
        for fn in fns:
            if fn.lower().endswith(VIDEO_EXTS):
                fp = os.path.join(dp, fn)
                out.append((fp, os.path.relpath(fp, path)))
    out.sort()
    return out


def main(argv=None) -> int:
    """CLI: regenerate the per-jump feature tables.

    Usage: python -m tandem.features <video-or-dir> [--out features] [--fps 10]

    Exports one CSV per recording that carries telemetry; recordings without GPMF
    are skipped. Re-run this whenever the feature-extraction code changes.
    """
    import argparse
    from tandem.phases.signals import build_signals_from_file

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Export per-recording feature tables (10 Hz).")
    ap.add_argument("path", help="video file or directory to walk")
    ap.add_argument("--out", default="features", help="output directory (default: features/)")
    ap.add_argument("--fps", type=float, default=10.0, help="resample grid, Hz (default: 10)")
    args = ap.parse_args(argv)

    videos = _collect_videos(args.path)
    if not videos:
        print(f"no videos found under {args.path}")
        return 1
    os.makedirs(args.out, exist_ok=True)
    print(f"exporting features for {len(videos)} recording(s) -> {args.out}")

    exported = skipped = failed = 0
    for i, (abspath, rel) in enumerate(videos):
        dst = os.path.join(args.out, _safe_name(rel) + ".csv")
        try:
            sig = build_signals_from_file(abspath, fs=args.fps)
            if sig is None:
                skipped += 1
                print(f"[{i+1}/{len(videos)}] no-telemetry {rel}")
                continue
            n = write_features_csv(sig, dst)
            exported += 1
            print(f"[{i+1}/{len(videos)}] {n} rows {rel}")
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            failed += 1
            print(f"[{i+1}/{len(videos)}] ERROR {rel}: {type(e).__name__}")
    print(f"done: {exported} exported, {skipped} no-telemetry, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
