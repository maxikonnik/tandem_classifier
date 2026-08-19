"""Per-jump feature table: every Signals channel on the 10 Hz grid.

One dense pass over a recording produces a flat table (one row per 10 Hz sample)
with all available telemetry and camera-metadata features. This is the Stage-0
foundation: the annotation tool and any downstream model read these tables instead
of re-parsing GPMF, so experiments run in seconds. Missing streams are written as
zeros; `has_*` flags on the Signals object say which were actually present.
"""
from __future__ import annotations

import csv

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
