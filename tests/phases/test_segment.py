from tandem.phases.signals import Signals
from tandem.phases.detect import Event, Segment
from tandem.phases.segment import Segmentation, segment_signals, _add_canopy


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
