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
from tandem.phases.signals import G, build_signals_from_file

# Accel and exposure exits should agree within this; a wider gap is flagged for review.
EXIT_AGREE_S = 3.0
# Visual vs gyroscope break-off gap above which the jump is flagged for review.
BREAKOFF_AGREE_S = 3.0

# Physical floor on the drogue -> deploy span (свободное падение). A drogue-slowed
# tandem falls for tens of seconds before the d-bag; in the 97 hand-labelled jumps
# this span is never below 23 s (median 44 s). The frozen probe, run on unfamiliar
# operators/cameras, sometimes collapses the two boundaries onto one frame — a
# physically impossible near-zero free-fall. Reject such a prediction rather than
# ship a confidently-wrong boundary; 15 s sits safely under the real minimum.
FREEFALL_MIN_S = 15.0

# Physical bounds on the exit -> deploy span (отделение + свободное падение, i.e. the
# whole descent from leaving the aircraft to the tandem's d-bag). Across the 128
# hand-labelled jumps this span is 28-57 s (median 49); 18-60 s is a safe envelope
# that admits low and high exits yet rejects a deploy landed impossibly close to the
# exit (a collapse) or far past it (a frame grabbed under the open canopy).
EXIT_DEPLOY_MIN_S = 18.0
EXIT_DEPLOY_MAX_S = 60.0

# Visual fallback when telemetry gives no exit. Anchored on an exposure exit, the
# probe window runs FALLBACK_PRE_S before it (the probe's cabin context) to
# FALLBACK_AFTER_S after (exit->break-off is ~55 s; generous for late telemetry).
# Without any telemetry the whole clip is searched, but only for clips that could hold
# a jump and are not so long that every ground clip in an archive burns minutes of CPU.
FALLBACK_PRE_S = 21.0
FALLBACK_AFTER_S = 150.0
FALLBACK_MIN_S = 40.0
FALLBACK_MAX_S = 600.0
# The exposure exit alone is not selective: going from indoors to daylight looks the
# same (it fires on ~37 % of non-jump clips — interviews, landings). A real exit is
# followed within seconds by a near-weightless dip even when the accel exit missed it
# (0.06-0.18 g on the missed jumps); only 9 % of non-jump exposure clips dip below
# 0.3 g. So the visual search is gated on that dip, sparing most of an archive.
FALLBACK_DIP_G = 0.3
FALLBACK_DIP_WINDOW_S = (-3.0, 15.0)

# For accel-only cameras (DJI, Insta360) with no exposure/GPS exit-corroborator, the
# exit detector can fire on a ground or handling jerk. A real jump enters free-fall,
# where the accelerometer drops toward weightlessness; a ground clip never does. So
# confirm a jump by requiring the accel magnitude to dip below this floor (~0.5 g)
# somewhere — a tandem in drogue free-fall reads ~0.2 g, well under it, while a held
# camera sits near 1 g (~9.8 m/s^2).
FREEFALL_CONFIRM_MS2 = 5.0


def _deploy_span_ok(exit_t: float, deploy_t: float) -> bool:
    """Whether a deploy time is physically plausible given the exit: the exit ->
    deploy span must fall within [EXIT_DEPLOY_MIN_S, EXIT_DEPLOY_MAX_S]."""
    return EXIT_DEPLOY_MIN_S <= (deploy_t - exit_t) <= EXIT_DEPLOY_MAX_S

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
    source: str = "gpmf"                                         # telemetry camera: gpmf / dji / insta360

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

    # Flag the telemetry provenance for any non-GoPro camera (DJI, Insta360, …): those
    # sources give accel + gyro but no GPS or exposure exit-corroborator, and their
    # break-off is a bare gyro turn whose meaning depends on who wears the camera, so
    # the annotator should sanity-check exit and break-off rather than trust them as
    # strongly as on the operator GoPro. The flag is "<SOURCE>_TELEMETRY".
    source = getattr(sig, "source", "gpmf")
    out.source = source
    if source != "gpmf":
        out.degradations.append(f"{source.upper()}_TELEMETRY")
        # Confirm a real jump: an accel-only camera's exit can fire on a ground jerk,
        # so require the accelerometer to actually enter free-fall somewhere. If it
        # never drops below the weightlessness floor, this is not a jump — drop the
        # boundary events (and free-fall phase) so no spurious prefill is emitted.
        if not (sig.accel_mag and min(sig.accel_mag) < FREEFALL_CONFIRM_MS2):
            out.events = [e for e in out.events
                          if e.type not in ("exit", "operator_breakoff")]
            out.phases = [p for p in out.phases if p.type != "freefall"]
            out.tracking_window = None
            return out

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
        # No telemetry at all (typically a re-exported/edited file with GPMF stripped):
        # with the probe on, look for a jump visually over the whole clip.
        if not probe:
            return None
        from tandem.recon.dji import probe_duration
        dur = probe_duration(path)
        if not (FALLBACK_MIN_S <= dur <= FALLBACK_MAX_S):
            return None
        out = Segmentation(source="none")
        _visual_fallback(out, path, 0.0, dur)
        return out if any(e.type == "exit" for e in out.events) else None
    out = segment_signals(sig)
    if visual and not probe:
        _add_canopy(out, path)
        _add_drogue(out, path)
    if probe:
        if any(e.type == "exit" for e in out.events):
            _apply_probe(out, path)
        else:
            # The accel exit can miss a short, shallow post-exit dip (at terminal speed
            # a GoPro reads ~1 g again within seconds); the exposure jump still marks
            # leaving the cabin, so anchor a visual search there.
            exp = next((e for e in out.events if e.type == "exit_exposure"), None)
            if exp is not None and _post_exit_dip_g(sig, exp.t_s) < FALLBACK_DIP_G:
                _visual_fallback(out, path, max(0.0, exp.t_s - FALLBACK_PRE_S),
                                 exp.t_s + FALLBACK_AFTER_S)
        # The CPU heuristics decode the whole jump and the probe overrides them; run one
        # only for a boundary the probe left empty (no weights, unreadable frames).
        if visual and any(e.type == "exit" for e in out.events):
            if out.canopy is None:
                _add_canopy(out, path)
            if out.drogue is None:
                _add_drogue(out, path)
    return out


def _post_exit_dip_g(sig, t: float) -> float:
    """Deepest specific force (in g) around a candidate exit — the weightless dip a
    real exit produces. Large (no dip) when the window holds no samples."""
    lo, hi = t + FALLBACK_DIP_WINDOW_S[0], t + FALLBACK_DIP_WINDOW_S[1]
    vals = [a for ts, a in zip(sig.t_s, sig.accel_min) if lo <= ts <= hi]
    return min(vals) / G if vals else float("inf")


def _visual_fallback(out: "Segmentation", path: str, lo: float, hi: float) -> None:
    """Find the jump purely visually in [lo, hi] when telemetry gave no exit: the
    5-class probe's HSMM decode places exit (5 fps refined), drogue, deploy and
    break-off, or finds no transition at all on a non-jump clip (then nothing is
    added). Flags VISUAL_FALLBACK so the annotator knows no telemetry backed it."""
    try:
        from tandem.visual.probe import predict_span
        pred = predict_span(path, lo, hi)
    except Exception:
        return
    if not pred or pred.get("exit") is None:
        return
    exit_e = Event(type="exit", t_s=pred["exit"], source="visual-probe", confidence=0.8)
    out.events.append(exit_e)
    out.degradations.append("VISUAL_FALLBACK")
    out.tracking_window = (exit_e.t_s, hi)
    _merge_probe(out, dict(pred, exit=None), exit_e, None, hi)


def _apply_probe(out: "Segmentation", path: str) -> None:
    """Refine the boundaries with the frozen-backbone probe (5 classes, HSMM decode;
    leave-one-session-out on 153 jumps, within 2 s: exit 99 %, drogue 96 %, deploy
    98 %, break-off 93 %). Its window is [exit, window_end]; window_end is the
    break-off when detected, else the free-fall end (tracking_window). The latter
    keeps the probe working on DJI accel-only telemetry, which has no gyro turn to
    give a break-off. Best-effort: if torch/transformers or the probe weights are
    missing it does nothing and the heuristic boundaries stand."""
    exit_e = next((e for e in out.events if e.type == "exit"), None)
    if exit_e is None:
        return
    breakoff = next((e for e in out.events if e.type == "operator_breakoff"), None)
    if breakoff is not None:
        window_end = breakoff.t_s
    elif out.tracking_window is not None:
        window_end = out.tracking_window[1]
    else:
        return
    try:
        from tandem.visual.probe import predict_boundaries
        pred = predict_boundaries(path, exit_e.t_s, window_end)
    except Exception:
        return
    if not pred:
        return
    if pred.get("jump_seen") is False:
        # The probe saw only the cabin around the telemetry exit — no frame of the jump.
        # Telemetry fired on something else (e.g. an in-cabin dip on a worn DJI in the
        # aircraft or at the interview), so this is not a jump: drop the draft.
        out.events = [e for e in out.events if e.type not in ("exit", "operator_breakoff")]
        out.drogue = None
        out.canopy = None
        out.tracking_window = None
        out.degradations.append("VISUAL_NO_JUMP")
        return
    if pred.get("exit") is None and pred.get("jump_seen") is not None and out.source != "gpmf":
        # An accel-only camera's exit is trusted only when the probe sees it: the DJI
        # accel exit fires in the cabin / 7-10 s early. Keep it for reference, emit no
        # exit boundary; the visual boundaries the clip does show still stand (it may
        # be one piece of a jump split across several DJI files).
        out.events = [e for e in out.events if e is not exit_e]
        out.events.append(Event(type="exit_telemetry", t_s=exit_e.t_s,
                                source=exit_e.source, confidence=exit_e.confidence))
        out.degradations.append("EXIT_NOT_SEEN")
    _merge_probe(out, pred, exit_e, breakoff, window_end)


def _merge_probe(out: "Segmentation", pred: dict, exit_e, breakoff, window_end: float) -> None:
    """Fold a probe prediction into the segmentation — visual exit and break-off
    events, drogue and canopy — each behind the physical-plausibility guards.
    ``exit_e`` is the exit the guards measure from; ``breakoff``/``window_end`` bound
    раскрытие until a visual break-off replaces them."""
    # The visual exit is the exit on every camera: against labels a human placed
    # themselves it lands within 1 s on 100 % of GoPro jumps (telemetry: 74 %, median
    # 0.20 vs 0.62 s), and on DJI the accel exit fires on in-cabin movement 7-10 s
    # early. Telemetry only located the window; it is kept as exit_telemetry, and a gap
    # over EXIT_AGREE_S is flagged for review.
    vis_exit = pred.get("exit")
    if vis_exit is not None:
        if abs(vis_exit - exit_e.t_s) > EXIT_AGREE_S:
            out.degradations.append("VISUAL_EXIT_DISAGREEMENT")
        out.events = [e for e in out.events if e is not exit_e]
        out.events.append(Event(type="exit_telemetry", t_s=exit_e.t_s,
                                source=exit_e.source, confidence=exit_e.confidence))
        exit_e = Event(type="exit", t_s=vis_exit, source="visual-probe", confidence=0.9)
        out.events.append(exit_e)
        if out.tracking_window is not None:
            out.tracking_window = (vis_exit, out.tracking_window[1])
    # Break-off (отворот) is the pair leaving the operator's frame — a visual event.
    # A probe trained on post-break-off frames finds it directly; it then replaces the
    # gyroscope turn (kept as breakoff_telemetry) and also fills jumps where the gyro
    # found none. The раскрытие interval ends at the final break-off.
    vis_bo = pred.get("breakoff")
    if vis_bo is not None:
        if breakoff is not None:
            if abs(vis_bo - breakoff.t_s) > BREAKOFF_AGREE_S:
                out.degradations.append("VISUAL_BREAKOFF_DISAGREEMENT")
            out.events = [e for e in out.events if e is not breakoff]
            out.events.append(Event(type="breakoff_telemetry", t_s=breakoff.t_s,
                                    source=breakoff.source, confidence=breakoff.confidence))
        breakoff = Event(type="operator_breakoff", t_s=vis_bo,
                         source="visual-probe", confidence=0.85)
        out.events.append(breakoff)
        window_end = vis_bo
        if out.tracking_window is not None:
            out.tracking_window = (out.tracking_window[0], vis_bo)
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
        if not _deploy_span_ok(exit_e.t_s, deploy_t):
            # deploy landed outside the physical exit -> deploy window: reject it
            # and flag rather than ship a canopy at the wrong depth of the jump.
            out.degradations.append("DEPLOY_SPAN_IMPLAUSIBLE")
            return
        end = max(deploy_t + 0.1, window_end)
        out.canopy = Segment(type="canopy", start_s=deploy_t, end_s=end,
                             source="visual-probe", confidence=0.91)
        # A span flag raised earlier by the heuristic path refers to the deploy just
        # replaced (often measured from a bad accel exit); the final one passed the guard.
        out.degradations = [d for d in out.degradations if d != "DEPLOY_SPAN_IMPLAUSIBLE"]


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
    # Reject a d-bag detected outside the physical exit -> deploy window (only when
    # we have an exit to measure from); a bright-line burst can otherwise fire on a
    # cabin door or an already-open canopy far from the real deploy.
    if exit_event is not None and not _deploy_span_ok(exit_t, onset_t):
        out.degradations.append("DEPLOY_SPAN_IMPLAUSIBLE")
        return
    # раскрытие spans [d-bag deploy -> break-off]; guard against a deploy detected
    # at or after the break-off (keep at least a short interval).
    end = max(breakoff.t_s, onset_t + 0.1)
    out.canopy = Segment(type="canopy", start_s=onset_t, end_s=end,
                         source="visual", confidence=conf)
