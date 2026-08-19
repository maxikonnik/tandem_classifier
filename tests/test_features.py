from tandem.phases.signals import Signals
from tandem.features import (signals_to_rows, write_features_csv, FEATURE_COLUMNS,
                             _safe_name, main)


def _sig():
    return Signals(t_s=[0.0, 0.1, 0.2], accel_mag=[9.8, 9.8, 9.8],
                   gyro_mag=[0.1, 0.2, 0.3], smile=[0.0, 50.0, 90.0], fs=10.0)


def test_signals_to_rows_shape_and_defaults():
    rows = signals_to_rows(_sig())
    assert len(rows) == 3
    assert set(rows[0].keys()) == set(FEATURE_COLUMNS)
    assert rows[2]["smile"] == 90.0
    assert rows[0]["accel_min"] == 0.0   # absent channel -> zeros


def test_write_features_csv(tmp_path):
    p = tmp_path / "f.csv"
    n = write_features_csv(_sig(), str(p))
    assert n == 3
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[0] == "t_s"
    assert len(lines) == 4   # header + 3 rows


def test_safe_name_sanitizes_separators_and_spaces():
    assert _safe_name("08 02\\Video\\GX012255.MP4") == "08_02__Video__GX012255.MP4"
    assert _safe_name("a/b c.mp4") == "a__b_c.mp4"


def test_cli_reports_when_no_videos(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1   # empty dir -> exit code 1
    assert "no videos" in capsys.readouterr().out
