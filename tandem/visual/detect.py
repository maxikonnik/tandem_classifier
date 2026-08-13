"""Visual event cues from a FrameFeature series.

exit  = aircraft leaves the frame: structure_frac drops sharply.
canopy = tandem canopy blooms: center_fill_frac rises sharply.
Every cue is source="visual".
"""
from __future__ import annotations

from dataclasses import dataclass

EXIT_STRUCTURE_HI = 0.25
EXIT_STRUCTURE_LO = 0.10
EXIT_DROP_WINDOW_S = 3.0


@dataclass
class VisualCue:
    type: str
    t_s: float
    source: str
    confidence: float


def exit_by_background(series):
    if not series:
        return None
    for i in range(len(series)):
        if series[i].structure_frac < EXIT_STRUCTURE_HI:
            continue
        # look ahead within the drop window for a fall below LO
        for j in range(i + 1, len(series)):
            if series[j].t_s - series[i].t_s > EXIT_DROP_WINDOW_S:
                break
            if series[j].structure_frac < EXIT_STRUCTURE_LO:
                drop = series[i].structure_frac - series[j].structure_frac
                conf = max(0.0, min(1.0, drop / EXIT_STRUCTURE_HI))
                return VisualCue("exit", series[j].t_s, "visual", round(conf, 3))
    return None
