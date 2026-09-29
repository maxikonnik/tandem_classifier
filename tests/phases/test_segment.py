import tandem.visual.probe as probe_mod
import tandem.visual.deploy as deploy_mod
from tandem.phases.signals import Signals
from tandem.phases.detect import Event, Segment
from tandem.phases.segment import (Segmentation, segment_signals, _add_canopy,
                                   _apply_probe, _deploy_span_ok, _visual_fallback)


def _jump_with_exit_breakoff():
    return Segmentation(events=[Event("exit", 38.0, "telemetry", 0.9),
                                Event("operator_breakoff", 95.0, "telemetry", 0.8)])


def test_apply_probe_sets_boundaries_when_freefall_plausible(monkeypatch):
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"drogue": 42.0, "deploy": 84.0})
    out = _jump_with_exit_breakoff()
    _apply_probe(out, "dummy.mp4")
    assert out.drogue is not None and out.drogue.t_s == 42.0
    assert out.canopy is not None and out.canopy.start_s == 84.0
    assert "PROBE_FREEFALL_IMPLAUSIBLE" not in out.degradations


def test_apply_probe_rejects_collapsed_freefall(monkeypatch):
    # drogue and deploy within an impossible <15 s free-fall -> guard rejects both.
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"drogue": 42.0, "deploy": 42.0})
    out = _jump_with_exit_breakoff()
    _apply_probe(out, "dummy.mp4")
    assert out.drogue is None and out.canopy is None
    assert "PROBE_FREEFALL_IMPLAUSIBLE" in out.degradations


def test_apply_probe_records_visual_exit_and_flags_disagreement(monkeypatch):
    # telemetry exit 38 s; the probe sees the cabin end at 50 s -> 12 s apart.
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"exit": 50.0, "drogue": 55.0, "deploy": 90.0})
    out = _jump_with_exit_breakoff()
    _apply_probe(out, "dummy.mp4")
    vis = [e for e in out.events if e.type == "exit_visual"]
    assert len(vis) == 1 and vis[0].t_s == 50.0
    assert "VISUAL_EXIT_DISAGREEMENT" in out.degradations
    # the telemetry exit stays the exit boundary
    assert next(e for e in out.events if e.type == "exit").t_s == 38.0


def test_apply_probe_visual_exit_agreeing_is_not_flagged(monkeypatch):
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"exit": 39.0, "drogue": 42.0, "deploy": 84.0})
    out = _jump_with_exit_breakoff()
    _apply_probe(out, "dummy.mp4")
    assert "VISUAL_EXIT_DISAGREEMENT" not in out.degradations


def test_apply_probe_visual_exit_is_primary_for_accel_only_camera(monkeypatch):
    # DJI: the accel exit fired early in the cabin; the visual exit takes over and the
    # telemetry one is kept as exit_telemetry. drogue 4 s after the visual exit.
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"exit": 55.0, "drogue": 59.0, "deploy": 90.0})
    out = _jump_with_exit_breakoff()          # telemetry exit 38, break-off 95
    out.source = "dji"
    out.tracking_window = (38.0, 95.0)
    _apply_probe(out, "dummy.mp4")
    assert next(e for e in out.events if e.type == "exit").t_s == 55.0
    assert next(e for e in out.events if e.type == "exit_telemetry").t_s == 38.0
    assert out.tracking_window == (55.0, 95.0)
    assert out.canopy is not None and out.canopy.start_s == 90.0   # 35 s after exit: ok


def test_apply_probe_clears_stale_heuristic_span_flag(monkeypatch):
    # the heuristic deploy was rejected (flag set) but the probe places a valid one.
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"drogue": 42.0, "deploy": 84.0})
    out = _jump_with_exit_breakoff()
    out.degradations.append("DEPLOY_SPAN_IMPLAUSIBLE")
    _apply_probe(out, "dummy.mp4")
    assert out.canopy is not None
    assert "DEPLOY_SPAN_IMPLAUSIBLE" not in out.degradations


def test_apply_probe_visual_breakoff_replaces_gyro_and_ends_raskrytie(monkeypatch):
    # gyro break-off 95 s; the pair leaves the frame at 88 s.
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"drogue": 42.0, "deploy": 84.0, "breakoff": 88.0})
    out = _jump_with_exit_breakoff()
    out.tracking_window = (38.0, 95.0)
    _apply_probe(out, "dummy.mp4")
    assert next(e for e in out.events if e.type == "operator_breakoff").t_s == 88.0
    assert next(e for e in out.events if e.type == "breakoff_telemetry").t_s == 95.0
    assert "VISUAL_BREAKOFF_DISAGREEMENT" in out.degradations
    assert out.canopy.end_s == 88.0 and out.tracking_window == (38.0, 88.0)


def test_apply_probe_visual_breakoff_fills_a_missing_gyro_breakoff(monkeypatch):
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"drogue": 42.0, "deploy": 84.0, "breakoff": 88.5})
    out = Segmentation(events=[Event("exit", 38.0, "telemetry", 0.9)],
                       tracking_window=(38.0, 100.0))          # gyro found no break-off
    _apply_probe(out, "dummy.mp4")
    assert next(e for e in out.events if e.type == "operator_breakoff").t_s == 88.5
    assert "VISUAL_BREAKOFF_DISAGREEMENT" not in out.degradations
    assert len(out.intervals()) == 3                         # раскрытие now closes


def test_visual_fallback_builds_the_jump_when_telemetry_has_no_exit(monkeypatch):
    monkeypatch.setattr(probe_mod, "predict_span",
                        lambda *a, **k: {"exit": 22.5, "drogue": 26.0, "deploy": 70.5,
                                         "breakoff": 74.5})
    out = Segmentation()                                  # telemetry found no exit
    _visual_fallback(out, "dummy.mp4", 0.0, 85.0)
    assert next(e for e in out.events if e.type == "exit").t_s == 22.5
    assert next(e for e in out.events if e.type == "operator_breakoff").t_s == 74.5
    assert out.drogue.t_s == 26.0 and out.canopy.start_s == 70.5
    assert "VISUAL_FALLBACK" in out.degradations
    assert len(out.intervals()) == 3


def test_visual_fallback_adds_nothing_on_a_non_jump_clip(monkeypatch):
    monkeypatch.setattr(probe_mod, "predict_span",
                        lambda *a, **k: {"exit": None, "drogue": None, "deploy": None})
    out = Segmentation()
    _visual_fallback(out, "dummy.mp4", 0.0, 60.0)
    assert out.events == [] and out.degradations == [] and out.canopy is None


def test_deploy_span_ok_bounds():
    # exit at 40: deploy plausible only within [40+18, 40+60] = [58, 100].
    assert _deploy_span_ok(40.0, 90.0) is True
    assert _deploy_span_ok(40.0, 56.0) is False    # 16 s: too close to exit
    assert _deploy_span_ok(40.0, 101.0) is False   # 61 s: too far past exit


def test_apply_probe_rejects_deploy_past_exit_window(monkeypatch):
    # drogue plausible, deploy 61 s after exit (>60) -> keep drogue, reject deploy.
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"drogue": 42.0, "deploy": 99.0})
    out = _jump_with_exit_breakoff()          # exit 38, breakoff 95
    _apply_probe(out, "dummy.mp4")
    assert out.drogue is not None and out.drogue.t_s == 42.0
    assert out.canopy is None
    assert "DEPLOY_SPAN_IMPLAUSIBLE" in out.degradations


def test_apply_probe_rejects_deploy_too_close_to_exit(monkeypatch):
    # free-fall span ok (deploy-drogue=15) but exit->deploy=17 (<18) -> reject deploy.
    monkeypatch.setattr(probe_mod, "predict_boundaries",
                        lambda *a, **k: {"drogue": 40.0, "deploy": 55.0})
    out = _jump_with_exit_breakoff()
    _apply_probe(out, "dummy.mp4")
    assert out.canopy is None
    assert "DEPLOY_SPAN_IMPLAUSIBLE" in out.degradations


def test_add_canopy_rejects_deploy_outside_exit_window(monkeypatch):
    monkeypatch.setattr(deploy_mod, "detect_deploy",
                        lambda *a, **k: (99.0, 0.5))   # 61 s after exit
    out = _jump_with_exit_breakoff()
    _add_canopy(out, "dummy.mp4")
    assert out.canopy is None
    assert "DEPLOY_SPAN_IMPLAUSIBLE" in out.degradations


def test_dji_exit_dropped_without_real_freefall(monkeypatch):
    # An accel-only (DJI) clip whose accel never enters free-fall is not a jump: a
    # ground/handling jerk can still make detect_phases emit an exit, so the guard
    # must drop it. Fake a detect_phases that returns an exit + free-fall phase.
    import tandem.phases.segment as segmod
    from tandem.phases.signals import Signals

    class _Res:
        phases = [Segment("freefall", 5.0, 35.0, "telemetry", 0.5)]
        events = [Event("exit", 5.0, "accel", 0.5)]
        degradations = []

    monkeypatch.setattr(segmod, "detect_phases", lambda sig: _Res())
    sig = Signals(source="dji")
    sig.accel_mag = [9.5] * 50            # stays near 1 g — never free-fall
    out = segmod.segment_signals(sig)
    assert not any(e.type == "exit" for e in out.events)
    assert out.tracking_window is None
    assert "DJI_TELEMETRY" in out.degradations


def test_add_canopy_accepts_deploy_inside_exit_window(monkeypatch):
    monkeypatch.setattr(deploy_mod, "detect_deploy",
                        lambda *a, **k: (84.0, 0.5))   # 46 s after exit
    out = _jump_with_exit_breakoff()
    _add_canopy(out, "dummy.mp4")
    assert out.canopy is not None and out.canopy.start_s == 84.0
    assert "DEPLOY_SPAN_IMPLAUSIBLE" not in out.degradations


def test_to_dict_shape():
    s = Segmentation(
        phases=[Segment("freefall", 10.0, 60.0, "telemetry", 0.85)],
        events=[Event("exit", 10.0, "telemetry", 0.9)],
        highlights=[Segment("orbit", 30.0, 34.0, "telemetry", 0.8)],
        tracking_window=(10.0, 55.0),
        degradations=[],
    )
    d = s.to_dict()
    assert d["phases"][0]["type"] == "freefall" and d["phases"][0]["start"] == 10.0
    assert d["events"][0]["t"] == 10.0
    assert d["highlights"][0]["type"] == "orbit"
    assert d["tracking_window"] == [10.0, 55.0]
    assert d["degradations"] == []


def test_no_telemetry_signals_segment():
    # Empty signals -> detect_phases returns NO_TELEMETRY, no freefall, no window.
    out = segment_signals(Signals())
    assert out.degradations == ["NO_TELEMETRY"]
    assert out.tracking_window is None
    assert out.highlights == []
    assert out.canopy is None


def test_to_dict_includes_canopy():
    s = Segmentation(canopy=Segment("canopy", 80.0, 85.0, "visual", 0.5))
    d = s.to_dict()
    assert d["canopy"]["type"] == "canopy"
    assert d["canopy"]["start"] == 80.0 and d["canopy"]["end"] == 85.0
    assert d["canopy"]["source"] == "visual"


def test_to_dict_canopy_none_by_default():
    assert Segmentation().to_dict()["canopy"] is None


def test_add_canopy_noop_without_breakoff():
    # No break-off event -> _add_canopy returns before any frame I/O.
    out = Segmentation(events=[Event("exit", 10.0, "telemetry", 0.9)])
    _add_canopy(out, "does-not-exist.mp4")
    assert out.canopy is None


def test_intervals_three_target_intervals():
    s = Segmentation(
        events=[Event("exit", 38.2, "telemetry", 0.5),
                Event("operator_breakoff", 91.6, "telemetry", 0.8)],
        canopy=Segment("canopy", 88.8, 91.6, "visual", 0.6),
    )
    ivs = s.intervals()
    assert [i["type"] for i in ivs] == ["отделение", "свободное падение", "раскрытие"]
    assert ivs[0]["kind"] == "moment" and ivs[0]["t_s"] == 38.2
    assert ivs[1]["start_s"] == 38.2 and ivs[1]["end_s"] == 88.8   # free-fall ends at the deploy
    assert ivs[2]["start_s"] == 88.8 and ivs[2]["end_s"] == 91.6   # раскрытие = deploy -> break-off


def test_intervals_freefall_runs_to_breakoff_without_deploy():
    s = Segmentation(events=[Event("exit", 38.2, "telemetry", 0.5),
                             Event("operator_breakoff", 91.6, "telemetry", 0.8)])
    ivs = s.intervals()
    assert [i["type"] for i in ivs] == ["отделение", "свободное падение"]
    assert ivs[1]["end_s"] == 91.6   # no deploy -> free-fall runs to break-off, no раскрытие


def test_intervals_empty_without_exit():
    assert Segmentation().intervals() == []


def test_edit_intervals_apply_cabin_lead_and_raskrytie_trim():
    s = Segmentation(
        events=[Event("exit", 38.2, "telemetry", 0.5),
                Event("operator_breakoff", 91.6, "telemetry", 0.8)],
        canopy=Segment("canopy", 88.8, 91.6, "visual", 0.6),
        drogue=Event("drogue", 42.5, "visual", 0.5),
    )
    ivs = s.edit_intervals(cabin_lead=4.0, raskrytie_trim=1.5)
    otd, ff, rask = ivs
    assert otd["start_s"] == 34.2 and otd["end_s"] == 42.5   # opens in the cabin, 4s before exit
    assert rask["start_s"] == 88.8 and rask["end_s"] == 90.1  # trimmed 1.5s before break-off (91.6)


def test_edit_intervals_cabin_lead_clamps_at_zero():
    s = Segmentation(events=[Event("exit", 2.0, "telemetry", 0.5)])
    assert s.edit_intervals()[0]["start_s"] == 0.0


def test_intervals_with_drogue_makes_exit_a_span_and_starts_freefall_at_drogue():
    s = Segmentation(
        events=[Event("exit", 38.2, "telemetry", 0.5),
                Event("operator_breakoff", 91.6, "telemetry", 0.8)],
        canopy=Segment("canopy", 88.8, 91.6, "visual", 0.6),
        drogue=Event("drogue", 42.5, "visual", 0.5),
    )
    ivs = s.intervals()
    assert [i["type"] for i in ivs] == ["отделение", "свободное падение", "раскрытие"]
    assert ivs[0]["kind"] == "span" and ivs[0]["start_s"] == 38.2 and ivs[0]["end_s"] == 42.5
    assert ivs[1]["start_s"] == 42.5 and ivs[1]["end_s"] == 88.8   # free-fall from drogue to deploy
