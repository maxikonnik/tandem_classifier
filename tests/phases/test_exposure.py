from tandem.phases.signals import Signals
from tandem.phases.exposure import detect_exit_exposure


def _sig_with_exposure(fs=10.0, exit_at=20.0, daylight=True):
    # 40 s: cabin (ISO 800, shutter 1/220) until exit_at, then daylight
    # (ISO 100, shutter 1/2000) if `daylight`.
    n = int(40 * fs)
    t = [i / fs for i in range(n)]
    iso, shut = [], []
    for ti in t:
        if daylight and ti >= exit_at:
            iso.append(100.0); shut.append(1.0 / 2000)
        else:
            iso.append(800.0); shut.append(1.0 / 220)
    return Signals(t_s=t, iso=iso, shutter=shut, fs=fs, has_exposure=True)


def test_exposure_exit_found_at_daylight_onset():
    sig = _sig_with_exposure(exit_at=20.0)
    ev = detect_exit_exposure(sig)
    assert ev is not None
    assert ev.type == "exit_exposure"
    assert ev.source == "telemetry"
    assert abs(ev.t_s - 20.0) < 0.3


def test_exposure_exit_none_without_daylight():
    assert detect_exit_exposure(_sig_with_exposure(daylight=False)) is None


def test_exposure_exit_none_without_exposure_stream():
    sig = _sig_with_exposure(exit_at=20.0)
    sig.has_exposure = False
    assert detect_exit_exposure(sig) is None
