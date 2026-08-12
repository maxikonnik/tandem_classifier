"""Sample keyframes across estimated windows and render an HTML sheet.

Extraction uses -skip_frame nokey and scales to 720p height per the
global constraints. The contact sheet embeds thumbnails as base64 so a
single file can be reviewed anywhere without the source video.
"""
from __future__ import annotations

import base64
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SampleShot:
    t_s: float
    label: str


def sample_plan(windows, every_s: float = 2.0) -> list[SampleShot]:
    shots: list[SampleShot] = []
    t = windows.freefall_start_s
    while t <= windows.freefall_end_s:
        shots.append(SampleShot(t_s=round(t, 3), label="freefall"))
        t += every_s
    if windows.canopy_open_s is not None:
        for k in range(3):  # a few frames right after canopy opens
            shots.append(SampleShot(t_s=round(windows.canopy_open_s + k * every_s, 3),
                                    label="canopy"))
    return shots


def extract_frames(video: str, shots: list[SampleShot], out_dir: str) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for shot in shots:
        name = f"{shot.label}_{shot.t_s:.3f}.jpg"
        dest = str(Path(out_dir) / name)
        subprocess.run(
            ["ffmpeg", "-y", "-skip_frame", "nokey", "-ss", f"{shot.t_s}",
             "-i", video, "-frames:v", "1",
             "-vf", "scale=-2:720", dest],
            check=True, capture_output=True,
        )
        # A shot timestamp past end-of-recording can make ffmpeg exit 0
        # without writing the JPEG; skip it rather than passing a
        # nonexistent path downstream to build_contact_sheet.
        if Path(dest).exists():
            paths.append(dest)
    return paths


def build_contact_sheet(image_paths: list[str], out_html: str, title: str) -> None:
    cells = []
    for path in image_paths:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        name = Path(path).name
        cells.append(
            f'<figure><img src="data:image/jpeg;base64,{b64}" alt="{name}">'
            f'<figcaption>{name}</figcaption></figure>'
        )
    html = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:sans-serif}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}"
        "img{width:100%;height:auto}figcaption{font-size:12px;color:#555}</style>"
        f"<h1>{title}</h1><div class='grid'>" + "".join(cells) + "</div>"
    )
    Path(out_html).write_text(html, encoding="utf-8")
