from tandem.phases.signals import Signals
from tandem.phases.detect import Segment
from tandem.phases.motion import freefall_std_ok, detect_orbit


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
