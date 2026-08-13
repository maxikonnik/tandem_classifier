"""Extract 720p keyframes and build a FrameFeature series."""
from __future__ import annotations

import glob
import os
import subprocess

import numpy as np
from PIL import Image

from tandem.recon.ffprobe import keyframe_pts
from tandem.visual.features import FrameFeature, frame_features


def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def extract_keyframes(video: str, out_dir: str, fps_cap=None) -> list[str]:
    if fps_cap is not None:
        raise NotImplementedError("fps_cap not yet supported")

    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-skip_frame", "nokey", "-i", video,
         "-vsync", "0", "-vf", "scale=-2:720",
         os.path.join(out_dir, "kf_%06d.jpg")],
        check=True, capture_output=True,
    )
    paths = sorted(glob.glob(os.path.join(out_dir, "kf_*.jpg")), key=_index_of)
    pts = keyframe_pts(video)
    with open(os.path.join(out_dir, "timestamps.txt"), "w") as fh:
        for path, t in zip(paths, pts):
            fh.write(f"{_index_of(path)} {t}\n")
    return paths


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
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                idx, t = parts
                stamps[int(idx)] = float(t)
    series = []
    for path in sorted(glob.glob(os.path.join(frame_dir, "kf_*.jpg")), key=_index_of):
        idx = _index_of(path)
        t_s = stamps.get(idx, float(idx))
        series.append(frame_features(load_gray(path), t_s=t_s))
    return series
