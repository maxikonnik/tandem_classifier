from tandem.phases.signals import Signals
from tandem.phases.detect import Segment
from tandem.phases.motion import freefall_std_ok, detect_orbit, detect_breakoff


def _freefall_with_orbit_dip(fs=10.0):
    # 20 s freefall window: accel_std ~0.4 (buffeting) except an 8..12 s
    # low-std dip (~0.1) where the operator orbits smoothly around the pair.
    n = int(20 * fs)
    t = [i / fs for i in range(n)]
    accel_std = [0.1 if 8.0 <= ti < 12.0 else 0.4 for ti in t]
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
    assert 7.5 <= seg.start_s <= 8.5
    assert 11.0 <= seg.end_s <= 12.5
    assert seg.end_s - seg.start_s >= 3.0


def _post_freefall_signals(fs=10.0, swing_g=0.0):
    # 20 s freefall window (baseline axis values near 0), followed by 5 s of
    # post-freefall samples. `swing_g` optionally bumps one axis (az) by that
    # much starting 2 s after freefall ends, mimicking the operator turning
    # away from the pair.
    freefall_n = int(20 * fs)
    post_n = int(5 * fs)
    n = freefall_n + post_n
    t = [i / fs for i in range(n)]
    ax = [0.0] * n
    ay = [0.0] * n
    az = [0.0] * n
    swing_start = freefall_n + int(2 * fs)
    for i in range(swing_start, n):
        az[i] = swing_g
    sig = Signals(t_s=t, ax=ax, ay=ay, az=az, fs=fs, has_accel=True)
    freefall = Segment(type="freefall", start_s=t[0], end_s=t[freefall_n - 1],
                        source="telemetry", confidence=0.85)
    return sig, freefall, t[swing_start]


def test_detect_breakoff_finds_axis_swing_after_freefall():
    sig, freefall, swing_t = _post_freefall_signals(swing_g=1.5)
    event = detect_breakoff(sig, freefall)
    assert event is not None
    assert event.type == "operator_breakoff"
    assert event.source == "telemetry"
    assert abs(event.t_s - swing_t) < 0.2


def test_detect_breakoff_none_when_axes_stay_near_baseline():
    sig, freefall, _ = _post_freefall_signals(swing_g=0.1)
    assert detect_breakoff(sig, freefall) is None
