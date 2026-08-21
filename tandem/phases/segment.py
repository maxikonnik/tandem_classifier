"""Assemble the detectors into one segmentation of an operator jump recording.

The product target is three intervals on the jump — отделение (exit), свободное
падение (free-fall) and раскрытие (opening) — delimited by three boundaries: exit,
the tandem's d-bag deploy, and the operator's break-off (отворот). So free-fall runs
[exit -> d-bag deploy] and раскрытие runs [d-bag deploy -> break-off]. `Segmentation`
carries those boundaries as events + the canopy interval, and `.intervals()` returns
the three target intervals directly. Scene sub-segmentation inside free-fall
(emotions, general shots, облёт) is a separate backlog task, not emitted here. Times
are recording-relative seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tandem.phases.api import detect_phases
from tandem.phases.detect import Event, Segment
from tandem.phases.exposure import detect_exit_exposure
from tandem.phases.motion import detect_breakoff, detect_orbit, freefall_std_ok
from tandem.phases.signals import build_signals_from_file

# Accel and exposure exits should agree within this; a wider gap is flagged for review.
EXIT_AGREE_S = 3.0

# Physical floor on the drogue -> deploy span (свободное падение). A drogue-slowed
# tandem falls for tens of seconds before the d-bag; in the 97 hand-labelled jumps
# this span is never below 23 s (median 44 s). The frozen probe, run on unfamiliar
# operators/cameras, sometimes collapses the two boundaries onto one frame — a
# physically impossible near-zero free-fall. Reject such a prediction rather than
# ship a confidently-wrong boundary; 15 s sits safely under the real minimum.
FREEFALL_MIN_S = 15.0

# Empirical editing offsets for the product cut (tunable against the final videos):
# the отделение clip opens CABIN_LEAD_S before the detected exit so it starts IN THE
# CABIN, and раскрытие ends RASKRYTIE_TRIM_S before the break-off so the operator's
# turn-away (отворот) does not land in the final video.
CABIN_LEAD_S = 4.0
RASKRYTIE_TRIM_S = 1.5


@dataclass
class Segmentation:
    phases: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    highlights: list[Segment] = field(default_factory=list)      # orbit (облёт) candidates
    canopy: Segment | None = None                                # canopy deploy -> break-off (visual)
    drogue: Event | None = None                                  # drogue throw = free-fall start (visual)
    tracking_window: tuple[float, float] | None = None           # [exit, break-off | freefall end]
    degradations: list[str] = field(default_factory=list)

    def intervals(self) -> list[dict]:
        """The product target: the three intervals on the jump, from the detected
        boundaries (exit, d-bag deploy, break-off).

        - отделение: the exit (a moment; its extent into free-fall is not detected).
        - свободное падение: [exit -> d-bag deploy] (falls back to break-off if the
          deploy was not found).
        - раскрытие: [d-bag deploy -> break-off] (only when both are present).

        Scene sub-segmentation inside free-fall (emotions, general shots, облёт) is
        a separate, backlog task and is not emitted here.
        """
        exit_e = next((e for e in self.events if e.type == "exit"), None)
        if exit_e is None:
            return []
        breakoff = next((e for e in self.events if e.type == "operator_breakoff"), None)
        deploy = self.canopy.start_s if self.canopy is not None else None
        ff_end = deploy if deploy is not None else (breakoff.t_s if breakoff else None)
        # free-fall begins at the drogue throw when we detected it, else at exit.
        ff_start = self.drogue.t_s if self.drogue is not None else exit_e.t_s

        if self.drogue is not None:
            out: list[dict] = [{"type": "отделение", "kind": "span",
                                "start_s": round(exit_e.t_s, 2), "end_s": round(self.drogue.t_s, 2),
                                "source": self.drogue.source, "confidence": self.drogue.confidence}]
        else:
            out = [{"type": "отделение", "kind": "moment",
                    "t_s": round(exit_e.t_s, 2), "source": exit_e.source}]
        if ff_end is not None:
            out.append({"type": "свободное падение", "kind": "span",
                        "start_s": round(ff_start, 2), "end_s": round(ff_end, 2),
                        "source": "telemetry"})
        if deploy is not None and breakoff is not None:
            out.append({"type": "раскрытие", "kind": "span",
                        "start_s": round(self.canopy.start_s, 2),
                        "end_s": round(self.canopy.end_s, 2),
                        "source": self.canopy.source, "confidence": self.canopy.confidence})
        return out

    def edit_intervals(self, cabin_lead: float = CABIN_LEAD_S,
                       raskrytie_trim: float = RASKRYTIE_TRIM_S) -> list[dict]:
        """The product cut: the three intervals with empirical editing offsets applied.

        Unlike `intervals()` (the raw detected boundaries, used for annotation and
        evaluation), this shifts times for the final video: отделение opens
        `cabin_lead` s before exit (so it starts in the cabin) and раскрытие ends
        `raskrytie_trim` s before the break-off (so the отворот is trimmed out). The
        offsets are tunable per the editors' taste.
        """
        exit_e = next((e for e in self.events if e.type == "exit"), None)
        if exit_e is None:
            return []
        breakoff = next((e for e in self.events if e.type == "operator_breakoff"), None)
        deploy = self.canopy.start_s if self.canopy is not None else None
        ff_end = deploy if deploy is not None else (breakoff.t_s if breakoff else None)
        ff_start = self.drogue.t_s if self.drogue is not None else exit_e.t_s

        out = [{"type": "отделение", "kind": "span",
                "start_s": round(max(0.0, exit_e.t_s - cabin_lead), 2),
                "end_s": round(ff_start, 2)}]
        if ff_end is not None:
            out.append({"type": "свободное падение", "kind": "span",
                        "start_s": round(ff_start, 2), "end_s": round(ff_end, 2)})
        if deploy is not None and breakoff is not None:
            r_end = max(deploy + 0.1, self.canopy.end_s - raskrytie_trim)
            out.append({"type": "раскрытие", "kind": "span",
                        "start_s": round(deploy, 2), "end_s": round(r_end, 2),
                        "confidence": self.canopy.confidence})
        return out

    def to_dict(self) -> dict:
        def seg(s):
            return {"type": s.type, "start": round(s.start_s, 2), "end": round(s.end_s, 2),
                    "source": s.source, "confidence": s.confidence}

        def ev(e):
            return {"type": e.type, "t": round(e.t_s, 2), "source": e.source,
                    "confidence": e.confidence}

        return {
            "intervals": self.intervals(),
            "edit_intervals": self.edit_intervals(),
            "phases": [seg(p) for p in self.phases],
            "events": [ev(e) for e in self.events],
            "highlights": [seg(h) for h in self.highlights],
            "canopy": seg(self.canopy) if self.canopy else None,
            "drogue": ev(self.drogue) if self.drogue else None,
            "tracking_window": ([round(self.tracking_window[0], 2), round(self.tracking_window[1], 2)]
                                if self.tracking_window else None),
            "degradations": list(self.degradations),
        }


def segment_signals(sig) -> Segmentation:
    res = detect_phases(sig)
    out = Segmentation(phases=list(res.phases), events=list(res.events),
                       degradations=list(res.degradations))

    # Independent exit corroborator from camera exposure (daylight onset). Agreement
    # with the accel exit raises confidence; a wide gap is an annotation-priority flag.
    exp_exit = detect_exit_exposure(sig)
    if exp_exit is not None:
        out.events.append(exp_exit)
        accel_exit = next((e for e in out.events if e.type == "exit"), None)
        if accel_exit is not None and abs(exp_exit.t_s - accel_exit.t_s) > EXIT_AGREE_S:
            out.degradations.append("EXIT_DISAGREEMENT")

    freefall = next((p for p in res.phases if p.type == "freefall"), None)
    if freefall is not None:
        # Orbit (облёт) reads accel_std, so keep it behind the std corroboration.
        # Break-off and the tracking window come from the GYROSCOPE / exit and must
        # NOT be gated on accel_std: some camera models log lower accel buffeting
        # (mean accel_std < FREEFALL_STD_MIN) yet still show a clear gyro turn, and
        # gating here silently dropped break-off + tracking + canopy on all of them.
        if freefall_std_ok(sig, freefall):
            out.highlights = detect_orbit(sig, freefall)
        breakoff = detect_breakoff(sig, freefall)
        if breakoff is not None:
            out.events.append(breakoff)
        exit_event = next((e for e in res.events if e.type == "exit"), None)
        if exit_event is not None:
            end = breakoff.t_s if breakoff is not None else freefall.end_s
            out.tracking_window = (exit_event.t_s, end)
    return out


def segment_file(path: str, fs: float = 10.0, visual: bool = True,
                 probe: bool = False) -> Segmentation | None:
    sig = build_signals_from_file(path, fs=fs)
    if sig is None:
        return None
    out = segment_signals(sig)
    if visual:
        _add_canopy(out, path)
        _add_drogue(out, path)
    if probe:
        _apply_probe(out, path)
    return out


def _apply_probe(out: "Segmentation", path: str) -> None:
    """Override the drogue and canopy (deploy) boundaries with the frozen-backbone
    probe — deploy 91 % / drogue 85 % vs the heuristics' 59 % / 16 % (leave-one-
    operator-out, 97 jumps). Its window is [exit, break-off], so it needs both.
    Best-effort: if torch/transformers or the probe weights are missing it does
    nothing and the heuristic boundaries stand."""
    exit_e = next((e for e in out.events if e.type == "exit"), None)
    breakoff = next((e for e in out.events if e.type == "operator_breakoff"), None)
    if exit_e is None or breakoff is None:
        return
    try:
        from tandem.visual.probe import predict_boundaries
        pred = predict_boundaries(path, exit_e.t_s, breakoff.t_s)
    except Exception:
        return
    if not pred:
        return
    drogue_t, deploy_t = pred.get("drogue"), pred.get("deploy")
    # Sanity guard: if the probe put drogue and deploy within an impossibly short
    # free-fall (< FREEFALL_MIN_S), it has collapsed the two phases — a known
    # failure on out-of-distribution footage. Reject both and flag for review,
    # leaving the heuristic (or empty) boundaries in place rather than overriding
    # with a wrong prefill the annotator would have to notice and undo.
    if (drogue_t is not None and deploy_t is not None
            and deploy_t - drogue_t < FREEFALL_MIN_S):
        out.degradations.append("PROBE_FREEFALL_IMPLAUSIBLE")
        return
    if drogue_t is not None:
        out.drogue = Event(type="drogue", t_s=drogue_t,
                           source="visual-probe", confidence=0.85)
    if deploy_t is not None:
        end = max(deploy_t + 0.1, breakoff.t_s)
        out.canopy = Segment(type="canopy", start_s=deploy_t, end_s=end,
                             source="visual-probe", confidence=0.91)


def _add_drogue(out: "Segmentation", path: str) -> None:
    """Add the drogue-throw boundary (free-fall start) when the pair's stabilization
    under the drogue is visible. Visual-only and best-effort: any failure, or no
    clear steady column, just leaves drogue unset (отделение stays a moment)."""
    exit_event = next((e for e in out.events if e.type == "exit"), None)
    if exit_event is None:
        return
    try:
        from tandem.visual.deploy import detect_drogue
        found = detect_drogue(path, exit_event.t_s)
    except Exception:
        return
    if found is None:
        return
    t, conf = found
    out.drogue = Event(type="drogue", t_s=t, source="visual", confidence=conf)


def _add_canopy(out: "Segmentation", path: str) -> None:
    """Add the раскрытие (opening) interval — from the tandem's d-bag emergence to
    the operator's break-off. `detect_deploy` finds the d-bag moment visually; the
    interval then runs to the break-off (отворот). Strictly visual for the start, so
    it never blocks the telemetry segmentation: any failure just leaves canopy unset.
    """
    breakoff = next((e for e in out.events if e.type == "operator_breakoff"), None)
    if breakoff is None:
        return
    exit_event = next((e for e in out.events if e.type == "exit"), None)
    exit_t = exit_event.t_s if exit_event else 0.0
    try:
        from tandem.visual.deploy import detect_deploy
        found = detect_deploy(path, breakoff.t_s, exit_t=exit_t)
    except Exception:
        return
    if found is None:
        return
    onset_t, conf = found
    # раскрытие spans [d-bag deploy -> break-off]; guard against a deploy detected
    # at or after the break-off (keep at least a short interval).
    end = max(breakoff.t_s, onset_t + 0.1)
    out.canopy = Segment(type="canopy", start_s=onset_t, end_s=end,
                         source="visual", confidence=conf)
