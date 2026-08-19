from tandem.phases.signals import Signals
from tandem.phases.detect import Segment
from tandem.phases.motion import freefall_std_ok, detect_orbit, detect_breakoff


def _freefall_with_orbit_dip(fs=10.0):
    # 40 s freefall window: accel_std ~0.4 (buffeting) except a 25..29 s
    # low-std dip (~0.1) where the operator orbits smoothly around the pair.
    # The dip is past the ORBIT_MIN_AFTER_EXIT_S onset margin (established freefall).
    n = int(40 * fs)
    t = [i / fs for i in range(n)]
    accel_std = [0.1 if 25.0 <= ti < 29.0 else 0.4 for ti in t]
    sig = Signals(t_s=t, accel_std=accel_std, fs=fs, has_accel=True)
    freefall = Segment(type="freefall", start_s=t[0], end_s=t[-1],
                        source="telemetry", confidence=0.85)
    return sig, freefall


def test_freefall_std_ok_true_for_high_std_window():
    sig, freefall = _freefall_with_orbit_dip()
    assert freefall_std_ok(sig, freefall) is True


def test_detect_orbit_finds_one_segment_spanning_the_dip():
    sig, freefall = _freefall_with_orbit_dip()
    segments = detect_orbit(sig, freefall)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.type == "orbit"
    assert seg.source == "telemetry"
    assert 24.5 <= seg.start_s <= 25.5
    assert 28.0 <= seg.end_s <= 29.5
    assert seg.end_s - seg.start_s >= 3.0


def _freefall_with_late_turn(fs=10.0, turn_rate=0.0):
    # 60 s freefall window. Gyro axes carry oscillatory buffeting (net ~0). A
    # sustained one-direction rotation of `turn_rate` rad/s on gz spans t=48..51 s
    # — the operator turning away in the LATE window, before their own deploy
    # (freefall end). The search window is [end-18, end-3.5] = [~42, ~56], so the
    # turn sits inside it.
    n = int(60 * fs)
    t = [i / fs for i in range(n)]
    buffet = [0.6 if (i % 2 == 0) else -0.6 for i in range(n)]  # oscillatory, net ~0
    gx = list(buffet)
    gy = list(buffet)
    gz = list(buffet)
    for i in range(int(48 * fs), int(51 * fs)):
        gz[i] = turn_rate   # sustained one-direction rotation
    mag = [max(abs(x), abs(y), abs(z)) for x, y, z in zip(gx, gy, gz)]
    sig = Signals(t_s=t, gx=gx, gy=gy, gz=gz, gyro_mag=mag, fs=fs, has_gyro=True)
    freefall = Segment(type="freefall", start_s=t[0], end_s=t[-1],
                        source="telemetry", confidence=0.85)
    return sig, freefall


def test_detect_breakoff_finds_late_freefall_gyro_turn():
    sig, freefall = _freefall_with_late_turn(turn_rate=2.0)
    event = detect_breakoff(sig, freefall)
    assert event is not None
    assert event.type == "operator_breakoff"
    assert event.source == "telemetry"
    assert 47.5 <= event.t_s <= 51.5   # lands within the sustained turn


def test_detect_breakoff_none_without_sustained_turn():
    # Only oscillatory buffeting (net ~0) — no sustained turn-away.
    sig, freefall = _freefall_with_late_turn(turn_rate=0.6)
    assert detect_breakoff(sig, freefall) is None


def test_detect_breakoff_none_without_gyro():
    sig, freefall = _freefall_with_late_turn(turn_rate=2.0)
    sig.has_gyro = False
    assert detect_breakoff(sig, freefall) is None
