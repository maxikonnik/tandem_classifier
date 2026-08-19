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
