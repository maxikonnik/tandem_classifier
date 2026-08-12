from tandem.phases.signals import Signals
from tandem.phases.detect import detect_exit, detect_freefall, Segment, G


def _in_aircraft_then_freefall(fs=10.0):
    # 10 s ground+climb at ~1 g, low then plane speed; 2 s exit dropout;
    # 20 s freefall at ~1 g noisy, speed ~55.
    accel, speed = [], []
    for _ in range(int(10 * fs)):      # in aircraft
        accel.append(1.0 * G); speed.append(40.0)
    for _ in range(int(2 * fs)):       # exit: |a| ~0
        accel.append(0.05 * G); speed.append(50.0)
    for _ in range(int(20 * fs)):      # freefall
        accel.append(1.0 * G); speed.append(55.0)
    t = [i / fs for i in range(len(accel))]
    return Signals(t_s=t, accel_mag=accel, speed_3d=speed, fs=fs,
                   has_accel=True, has_gps=True)


def test_detect_exit_at_dropout():
    sig = _in_aircraft_then_freefall()
    ev = detect_exit(sig)
    assert ev is not None
    assert ev.type == "exit"
    assert ev.source == "telemetry"
    assert 9.5 <= ev.t_s <= 12.5     # within the dropout window (~10-12 s)
    assert ev.confidence > 0.5


def test_detect_exit_none_when_no_dropout():
    fs = 10.0
    accel = [1.0 * G] * int(30 * fs)   # steady 1 g, never drops
    speed = [40.0] * int(30 * fs)
    t = [i / fs for i in range(len(accel))]
    sig = Signals(t_s=t, accel_mag=accel, speed_3d=speed, fs=fs,
                  has_accel=True, has_gps=True)
    assert detect_exit(sig) is None


def test_detect_freefall_after_exit():
    sig = _in_aircraft_then_freefall()
    ev = detect_exit(sig)
    seg = detect_freefall(sig, ev)
    assert seg is not None
    assert seg.type == "freefall"
    assert seg.source == "telemetry"
    assert seg.start_s >= ev.t_s - 0.5
    assert seg.end_s > seg.start_s
    # freefall spans roughly the last 20 s (12..32)
    assert seg.end_s - seg.start_s >= 15.0


def test_detect_freefall_none_without_exit():
    sig = _in_aircraft_then_freefall()
    assert detect_freefall(sig, None) is None
