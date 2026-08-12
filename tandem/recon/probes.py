"""One probe per section-9 assumption. Mechanical probes measure;
visual probes hand back a contact sheet for human tally."""
from __future__ import annotations

from dataclasses import dataclass

from tandem.recon import jumps as jumps_mod
from tandem.recon.ffprobe import IntervalStats
from tandem.recon.telemetry import Telemetry


@dataclass
class ProbeResult:
    assumption: int
    title: str
    status: str
    value: dict
    degradation: str | None = None


def probe_gps_utc(tel: Telemetry | None) -> ProbeResult:
    if tel is None:
        return ProbeResult(1, "GPS UTC present", "blocked", {}, degradation="NO_TELEMETRY")
    return ProbeResult(
        1, "GPS UTC present", "measured",
        {"has_gps": tel.has_gps, "has_utc": tel.has_utc,
         "first_utc": tel.first_utc.isoformat() if tel.first_utc else None},
    )


def probe_keyframe_interval(stats: IntervalStats) -> ProbeResult:
    status = "measured" if stats.count >= 2 else "blocked"
    return ProbeResult(
        2, "Keyframe interval", status,
        {"count": stats.count, "median_s": stats.median_s,
         "min_s": stats.min_s, "max_s": stats.max_s},
    )


def probe_naming(recordings: list, unmatched: list[str]) -> ProbeResult:
    status = "measured" if not unmatched else "needs_review"
    return ProbeResult(
        5, "Naming and chunk convention", status,
        {"recordings": len(recordings), "unmatched": unmatched},
    )


def probe_dual_camera(jump_list: list[jumps_mod.Jump]) -> ProbeResult:
    frac = jumps_mod.dual_camera_fraction(jump_list)
    return ProbeResult(
        7, "Fraction of jumps with both cameras", "measured",
        {"jumps": len(jump_list), "fraction": frac},
        degradation=None if frac > 0 else "NO_INSTRUCTOR_CAM",
    )


def probe_visual(assumption: int, title: str, sheet_path: str | None) -> ProbeResult:
    return ProbeResult(
        assumption, title, "needs_review",
        {"contact_sheet": sheet_path},
    )
