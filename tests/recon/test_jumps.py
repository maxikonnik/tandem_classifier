from datetime import datetime, timezone

from tandem.recon.jumps import RecordingInfo, group_jumps, dual_camera_fraction


def _info(dev, start, end):
    return RecordingInfo(
        recording=None,
        telemetry=None,
        start_utc=datetime(2026, 8, 12, 9, start, 0, tzinfo=timezone.utc),
        end_utc=datetime(2026, 8, 12, 9, end, 0, tzinfo=timezone.utc),
        _device_id=dev,
    )


def test_overlapping_recordings_form_one_jump():
    a = _info(dev=1, start=0, end=8)   # operator
    b = _info(dev=2, start=1, end=6)   # instructor hand
    jumps = group_jumps([a, b])
    assert len(jumps) == 1
    assert jumps[0].distinct_devices == {1, 2}


def test_separated_recordings_form_two_jumps():
    a = _info(dev=1, start=0, end=8)
    b = _info(dev=1, start=20, end=28)
    jumps = group_jumps([a, b])
    assert len(jumps) == 2


def test_dual_camera_fraction():
    a = _info(dev=1, start=0, end=8)
    b = _info(dev=2, start=1, end=6)
    c = _info(dev=1, start=30, end=38)
    jumps = group_jumps([a, b, c])
    assert dual_camera_fraction(jumps) == 0.5
