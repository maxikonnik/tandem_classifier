"""Thin wrappers over ffprobe/ffmpeg. Pure helpers are separated from
subprocess calls so the logic is testable without media files."""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass


@dataclass
class IntervalStats:
    count: int
    median_s: float
    min_s: float
    max_s: float


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run_json(args: list[str]) -> dict:
    out = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def ffprobe_streams(path: str) -> list[dict]:
    data = _run_json(["ffprobe", "-v", "error", "-show_streams",
                      "-print_format", "json", path])
    return data.get("streams", [])


def _select_gpmf_index(streams: list[dict]) -> int | None:
    for s in streams:
        if s.get("codec_tag_string") == "gpmd":
            return int(s["index"])
        handler = s.get("tags", {}).get("handler_name", "")
        if "GoPro MET" in handler:
            return int(s["index"])
    return None


def find_gpmf_stream_index(path: str) -> int | None:
    return _select_gpmf_index(ffprobe_streams(path))


def keyframe_pts(path: str) -> list[float]:
    data = _run_json([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-skip_frame", "nokey", "-show_frames",
        "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time",
        "-print_format", "json", path,
    ])
    pts = []
    for frame in data.get("frames", []):
        raw = frame.get("best_effort_timestamp_time") or frame.get("pkt_pts_time")
        if raw is not None:
            pts.append(float(raw))
    return sorted(pts)


def keyframe_interval_stats(pts: list[float]) -> IntervalStats:
    if len(pts) < 2:
        return IntervalStats(count=len(pts), median_s=0.0, min_s=0.0, max_s=0.0)
    deltas = [b - a for a, b in zip(pts, pts[1:])]
    return IntervalStats(
        count=len(pts),
        median_s=statistics.median(deltas),
        min_s=min(deltas),
        max_s=max(deltas),
    )
