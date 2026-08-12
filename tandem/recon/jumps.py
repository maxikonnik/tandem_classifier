"""Group recordings into jumps by overlapping UTC time and count cameras.

A single jump is filmed by one or two cameras at the same wall-clock
time. Two cameras therefore appear as two recordings whose UTC ranges
overlap but whose GPMF device ids differ. Recordings whose ranges are
separated by more than the gap tolerance belong to different jumps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class RecordingInfo:
    recording: object
    telemetry: object
    start_utc: datetime | None
    end_utc: datetime | None
    _device_id: int | None = None

    @property
    def device_id(self) -> int | None:
        if self._device_id is not None:
            return self._device_id
        return getattr(self.telemetry, "device_id", None)


@dataclass
class Jump:
    recordings: list[RecordingInfo] = field(default_factory=list)

    @property
    def distinct_devices(self) -> set[int]:
        return {r.device_id for r in self.recordings if r.device_id is not None}


def group_jumps(infos: list[RecordingInfo], gap_tolerance_s: float = 120.0) -> list[Jump]:
    timed = [i for i in infos if i.start_utc is not None and i.end_utc is not None]
    timed.sort(key=lambda i: i.start_utc)
    jumps: list[Jump] = []
    tol = timedelta(seconds=gap_tolerance_s)
    current: Jump | None = None
    current_end: datetime | None = None
    for info in timed:
        if current is None or info.start_utc > current_end + tol:
            current = Jump(recordings=[info])
            current_end = info.end_utc
            jumps.append(current)
        else:
            current.recordings.append(info)
            current_end = max(current_end, info.end_utc)
    return jumps


def dual_camera_fraction(jumps: list[Jump]) -> float:
    if not jumps:
        return 0.0
    dual = sum(1 for j in jumps if len(j.distinct_devices) >= 2)
    return dual / len(jumps)
