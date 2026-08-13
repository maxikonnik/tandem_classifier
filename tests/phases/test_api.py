from tandem.phases.signals import Signals
from tandem.phases.detect import G
from tandem.phases.api import detect_phases, PhaseResult


def _full_flight(fs=10.0):
    accel, amin, speed = [], [], []
    for _ in range(int(5 * fs)):       # ground
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(0.5)
    for _ in range(int(5 * fs)):       # climb
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(40.0)
    for _ in range(int(2 * fs)):       # exit
        accel.append(0.05 * G); amin.append(0.05 * G); speed.append(50.0)
    for _ in range(int(20 * fs)):      # freefall
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(55.0)
    t = [i / fs for i in range(len(accel))]
    return Signals(t_s=t, accel_mag=accel, accel_min=amin, speed_3d=speed, fs=fs,
                   has_accel=True, has_gps=True)


def test_detect_phases_full_flight():
    res = detect_phases(_full_flight())
    phase_types = [p.type for p in res.phases]
    assert phase_types == ["ground_pre", "climb", "freefall"]
    assert [e.type for e in res.events] == ["exit"]
    assert res.degradations == []
    assert all(p.source == "telemetry" for p in res.phases)


def test_detect_phases_no_signal_is_no_telemetry():
    res = detect_phases(Signals())
    assert res.phases == [] and res.events == []
    assert res.degradations == ["NO_TELEMETRY"]


def test_full_flight_without_gps_still_detects_phases():
    from tests.phases.test_detect import _accel_only_flight  # reuse fixture
    res = detect_phases(_accel_only_flight())
    assert "exit" in [e.type for e in res.events]
    assert "freefall" in [p.type for p in res.phases]
    assert "NO_TELEMETRY" not in res.degradations


def test_detect_phases_flags_order_violation(monkeypatch):
    # Force the detectors to return physically-impossible, out-of-order
    # segments so the FSM branch in detect_phases is exercised directly.
    import tandem.phases.api as api
    from tandem.phases.detect import Segment, Event

    monkeypatch.setattr(api, "detect_exit",
                        lambda s: Event("exit", 12.0, "telemetry", 0.9))
    monkeypatch.setattr(api, "detect_ground_climb",
                        lambda s, e: [Segment("freefall", 0.0, 5.0, "telemetry", 0.9)])
    monkeypatch.setattr(api, "detect_freefall",
                        lambda s, e: Segment("climb", 5.0, 10.0, "telemetry", 0.9))

    res = detect_phases(_full_flight())
    assert "PHASE_ORDER_VIOLATION" in res.degradations
    # every phase is marked unreliable (confidence zeroed) on a violation
    assert res.phases and all(p.confidence == 0.0 for p in res.phases)
