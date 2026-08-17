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


def _freefall_with_late_swing(fs=10.0, swing_g=0.0):
    # 60 s freefall window (axes near 0 = steady tracking orientation). `swing_g`
    # sustains a bump on one axis (az) from t=48 s to the end of freefall — the
    # operator turning away in the LATE part of freefall, before their own deploy.
    n = int(60 * fs)
    t = [i / fs for i in range(n)]
    ax = [0.0] * n
    ay = [0.0] * n
    az = [0.0] * n
    swing_start = int(48 * fs)
    for i in range(swing_start, n):
        az[i] = swing_g
    sig = Signals(t_s=t, ax=ax, ay=ay, az=az, fs=fs, has_accel=True)
    freefall = Segment(type="freefall", start_s=t[0], end_s=t[-1],
                        source="telemetry", confidence=0.85)
    return sig, freefall, t[swing_start]


def test_detect_breakoff_finds_late_freefall_axis_swing():
    sig, freefall, swing_t = _freefall_with_late_swing(swing_g=1.5)
    event = detect_breakoff(sig, freefall)
    assert event is not None
    assert event.type == "operator_breakoff"
    assert event.source == "telemetry"
    assert abs(event.t_s - swing_t) < 0.3


def test_detect_breakoff_none_when_axes_stay_near_baseline():
    sig, freefall, _ = _freefall_with_late_swing(swing_g=0.1)
    assert detect_breakoff(sig, freefall) is None
