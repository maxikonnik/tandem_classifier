"""Assemble the telemetry detectors into one segmentation of a jump recording.

Runs the full operator-camera telemetry pipeline and returns a single
`Segmentation`: the physical phases, the events (exit, operator break-off),
the orbit (облёт) highlight segments, and the useful pair-tracking window
`[exit, break-off]` — the fragment of the jump where the operator is filming
the tandem pair. All results carry source="telemetry"; times are
recording-relative seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tandem.phases.api import detect_phases
from tandem.phases.detect import Event, Segment
from tandem.phases.motion import detect_breakoff, detect_orbit, freefall_std_ok
from tandem.phases.signals import build_signals_from_file


@dataclass
class Segmentation:
    phases: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    highlights: list[Segment] = field(default_factory=list)      # orbit (облёт) candidates
    tracking_window: tuple[float, float] | None = None           # [exit, break-off | freefall end]
    degradations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def seg(s):
            return {"type": s.type, "start": round(s.start_s, 2), "end": round(s.end_s, 2),
                    "source": s.source, "confidence": s.confidence}

        def ev(e):
            return {"type": e.type, "t": round(e.t_s, 2), "source": e.source,
                    "confidence": e.confidence}

        return {
            "phases": [seg(p) for p in self.phases],
            "events": [ev(e) for e in self.events],
            "highlights": [seg(h) for h in self.highlights],
            "tracking_window": ([round(self.tracking_window[0], 2), round(self.tracking_window[1], 2)]
                                if self.tracking_window else None),
            "degradations": list(self.degradations),
        }


def segment_signals(sig) -> Segmentation:
    res = detect_phases(sig)
    out = Segmentation(phases=list(res.phases), events=list(res.events),
                       degradations=list(res.degradations))

    freefall = next((p for p in res.phases if p.type == "freefall"), None)
    if freefall is not None and freefall_std_ok(sig, freefall):
        out.highlights = detect_orbit(sig, freefall)
        breakoff = detect_breakoff(sig, freefall)
        if breakoff is not None:
            out.events.append(breakoff)
        exit_event = next((e for e in res.events if e.type == "exit"), None)
        if exit_event is not None:
            end = breakoff.t_s if breakoff is not None else freefall.end_s
            out.tracking_window = (exit_event.t_s, end)
    return out


def segment_file(path: str, fs: float = 10.0) -> Segmentation | None:
    sig = build_signals_from_file(path, fs=fs)
    if sig is None:
        return None
    return segment_signals(sig)
