from tandem.phases.signals import Signals
from tandem.phases.detect import Event, Segment
from tandem.phases.segment import Segmentation, segment_signals


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
