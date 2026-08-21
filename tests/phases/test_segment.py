import tandem.visual.probe as probe_mod
import tandem.visual.deploy as deploy_mod
from tandem.phases.signals import Signals
from tandem.phases.detect import Event, Segment
from tandem.phases.segment import (Segmentation, segment_signals, _add_canopy,
                                   _apply_probe, _deploy_span_ok)


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
