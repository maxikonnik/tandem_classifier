"""Reconnaissance CLI: scan an archive and write a findings report."""
from __future__ import annotations

import argparse
import glob
import os
from datetime import timedelta
from pathlib import Path

from tandem.recon.naming import group_recordings
from tandem.recon.telemetry import read_telemetry
from tandem.recon.ffprobe import keyframe_pts, keyframe_interval_stats
from tandem.recon.jumps import RecordingInfo, group_jumps
from tandem.recon.windows import Windows, estimate_windows
from tandem.recon.sampler import sample_plan, extract_frames, build_contact_sheet
from tandem.recon import probes as P
from tandem.recon.report import render_report
from tandem.phases.signals import build_signals_from_file
from tandem.phases.api import detect_phases


def _list_media(archive_dir: str) -> list[str]:
    files = []
    for ext in ("MP4", "mp4"):
        files.extend(glob.glob(os.path.join(archive_dir, f"*.{ext}")))
    return sorted(set(files))


def _accel_freefall_window(primary_path: str) -> Windows | None:
    """Trimmed freefall window from the accelerometer phase detector.

    Returns None when there is no accelerometer telemetry or the
    detector found no freefall segment, so the caller can fall back to
    the crude GPS-speed estimate (`estimate_windows`).
    """
    sig = build_signals_from_file(primary_path)
    if sig is None or not sig.has_accel:
        return None
    result = detect_phases(sig)
    freefall = next((p for p in result.phases if p.type == "freefall"), None)
    if freefall is None:
        return None
    return Windows(freefall_start_s=freefall.start_s, freefall_end_s=freefall.end_s,
                    canopy_open_s=None)


def run(archive_dir: str, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = _list_media(archive_dir)
    recordings, unmatched = group_recordings(paths)

    infos: list[RecordingInfo] = []
    first_tel = None
    first_tel_info = None
    first_stats = None
    for rec in recordings:
        primary = rec.chapters[0]
        tel = read_telemetry(primary)
        if first_stats is None:
            first_stats = keyframe_interval_stats(keyframe_pts(primary))
        start = tel.first_utc if tel else None
        end = (start + timedelta(seconds=len(tel.speed_3d_ms) / 18.0)) if (tel and start) else None
        info = RecordingInfo(recording=rec, telemetry=tel, start_utc=start, end_utc=end)
        infos.append(info)
        if first_tel is None and tel is not None:
            first_tel = tel
            first_tel_info = info

    jump_list = group_jumps(infos)

    results = [
        P.probe_gps_utc(first_tel),
        P.probe_keyframe_interval(first_stats if first_stats else keyframe_interval_stats([])),
        P.probe_naming(recordings, unmatched),
        P.probe_dual_camera(jump_list),
    ]

    # Visual assumptions: sample frames from the first telemetried recording.
    sheet = None
    if first_tel and first_tel_info:
        primary_path = first_tel_info.recording.chapters[0]
        windows = _accel_freefall_window(primary_path)
        if windows is None:
            windows = estimate_windows(first_tel)
        if windows:
            shots = sample_plan(windows)
            frame_dir = os.path.join(out_dir, "frames")
            frames = extract_frames(primary_path, shots, frame_dir)
            sheet = os.path.join(out_dir, "contact_sheet.html")
            build_contact_sheet(frames, sheet, title="recon-sample")
    for a, title in [
        (3, "Ground background fraction in freefall"),
        (4, "Canopy opening lands in operator frame"),
        (6, "How close operator gets to the pair"),
        (8, "How tandem landing is framed"),
    ]:
        results.append(P.probe_visual(a, title, sheet))

    report = render_report(results, archive=archive_dir)
    report_path = os.path.join(out_dir, "recon-findings.md")
    Path(report_path).write_text(report, encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tandem.recon",
                                     description="Verify section-9 assumptions on an archive.")
    parser.add_argument("archive", help="directory of camera files")
    parser.add_argument("-o", "--out", required=True, help="output directory")
    args = parser.parse_args(argv)
    path = run(args.archive, args.out)
    print(f"wrote {path}")
    return 0
