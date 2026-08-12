from tandem.recon.telemetry import Telemetry
from tandem.recon.ffprobe import IntervalStats
from tandem.recon.jumps import Jump, RecordingInfo
from tandem.recon.probes import (
    probe_gps_utc, probe_keyframe_interval, probe_dual_camera, probe_visual,
)


def test_probe_gps_utc_measured_when_present():
    tel = Telemetry(has_gps=True, has_utc=True)
    r = probe_gps_utc(tel)
    assert r.assumption == 1
    assert r.status == "measured"
    assert r.value["has_utc"] is True
    assert r.degradation is None


def test_probe_gps_utc_blocked_without_telemetry():
    r = probe_gps_utc(None)
    assert r.status == "blocked"
    assert r.degradation == "NO_TELEMETRY"


def test_probe_keyframe_interval_reports_median():
    r = probe_keyframe_interval(IntervalStats(count=100, median_s=0.5, min_s=0.5, max_s=0.5))
    assert r.assumption == 2
    assert r.value["median_s"] == 0.5


def test_probe_dual_camera_fraction():
    j1 = Jump(recordings=[RecordingInfo(None, None, None, None, 1),
                          RecordingInfo(None, None, None, None, 2)])
    j2 = Jump(recordings=[RecordingInfo(None, None, None, None, 1)])
    r = probe_dual_camera([j1, j2])
    assert r.assumption == 7
    assert r.value["fraction"] == 0.5


def test_probe_visual_always_needs_review():
    r = probe_visual(3, "Ground background fraction", "/out/sheet.html")
    assert r.status == "needs_review"
    assert r.value["contact_sheet"] == "/out/sheet.html"
