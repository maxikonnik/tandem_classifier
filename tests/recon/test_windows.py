from tandem.recon.telemetry import Telemetry
from tandem.recon.windows import smooth, estimate_windows


def _synthetic_speed():
    speed, t = [], []
    time = 0.0
    for _ in range(50):      # freefall: ~55 m/s
        speed.append(55.0); t.append(time); time += 1.0
    for _ in range(60):      # under canopy: ~6 m/s
        speed.append(6.0); t.append(time); time += 1.0
    return speed, t


def test_smooth_reduces_to_neighbourhood_mean():
    assert smooth([10.0, 10.0, 10.0])[1] == 10.0
    out = smooth([0.0, 0.0, 30.0, 0.0, 0.0], k=3)
    assert out[2] == 10.0  # (0 + 30 + 0) / 3


def test_estimate_windows_finds_freefall_and_canopy():
    speed, t = _synthetic_speed()
    tel = Telemetry(has_gps=True, speed_3d_ms=speed, t_s=t)
    w = estimate_windows(tel)
    assert w is not None
    assert w.freefall_start_s < w.freefall_end_s
    assert 45.0 <= w.canopy_open_s <= 55.0


def test_estimate_windows_none_without_speed():
    assert estimate_windows(Telemetry()) is None
