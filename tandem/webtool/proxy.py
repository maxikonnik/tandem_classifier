"""Lazily transcode an HEVC clip to a browser-friendly H.264 proxy.

Invoked only when the frontend reports the original will not decode. The
proxy is cached as a hidden sibling file so it is made once per clip.
"""
from __future__ import annotations

import os
import subprocess


def proxy_path(src: str) -> str:
    directory = os.path.dirname(src)
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(directory, f".{stem}.proxy.mp4")


def ensure_h264_proxy(src: str, runner=subprocess.run) -> str:
    out = proxy_path(src)
    if os.path.exists(out):
        return out
    runner(
        ["ffmpeg", "-y", "-i", src,
         "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", out],
        check=True,
    )
    return out
