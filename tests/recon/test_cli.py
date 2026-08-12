from pathlib import Path

from tandem.recon import cli
from tandem.recon.telemetry import Telemetry
from tandem.recon.ffprobe import IntervalStats


def test_run_writes_report(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "GX010123.MP4").write_bytes(b"stub")

    from datetime import datetime, timezone
    tel = Telemetry(device_id=1, has_gps=True, has_utc=True,
                    first_utc=datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc),
                    speed_3d_ms=[55.0, 6.0], t_s=[0.0, 1.0])

    monkeypatch.setattr(cli, "read_telemetry", lambda p: tel)
    monkeypatch.setattr(cli, "keyframe_pts", lambda p: [0.0, 0.5, 1.0])
    monkeypatch.setattr(cli, "extract_frames", lambda v, s, o: [])
    monkeypatch.setattr(cli, "build_contact_sheet", lambda paths, out, title: Path(out).write_text("x"))

    out = tmp_path / "out"
    report_path = cli.run(str(archive), str(out))

    text = Path(report_path).read_text(encoding="utf-8")
    assert "Reconnaissance findings" in text
    for a in (1, 2, 5, 7):
        assert f"Assumption {a}" in text


def test_main_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run", lambda a, o: str(tmp_path / "r.md"))
    (tmp_path / "r.md").write_text("ok")
    assert cli.main([str(tmp_path), "-o", str(tmp_path / "out")]) == 0
