import json

import numpy as np

from tandem.visual.probe import _first_ge, _first_run_ge, _smooth, _PROBE_JSON


def test_probe_weights_present_and_shaped():
    with open(_PROBE_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["dim"] == 384
    assert len(cfg["mu"]) == 384 and len(cfg["sd"]) == 384
    n = len(cfg["phases"])                                   # 4, or 5 with pre-exit
    assert len(cfg["W"]) == n and len(cfg["W"][0]) == 384
    assert len(cfg["b"]) == n
    # phases are time-ordered; the boundary lookups rely on these names existing
    assert {"отделение", "свободное падение", "раскрытие"} <= set(cfg["phases"])
    assert cfg["phases"].index("отделение") < cfg["phases"].index("раскрытие")


def test_first_ge_finds_first_sustained_crossing():
    ts = np.array([10, 11, 12, 13, 14, 15, 16], float)
    cls = np.array([0, 0, 1, 1, 2, 2, 3])
    assert _first_ge(cls, ts, 1) == 12.0   # свободное падение opens
    assert _first_ge(cls, ts, 2) == 14.0   # раскрытие opens
    assert _first_ge(np.array([0, 0, 0]), ts[:3], 2) is None


def test_first_run_ge_needs_a_sustained_run():
    # 5 fps refinement: a lone door-frame flicker (index 2) must not count as exit.
    ts = np.arange(10) * 0.2
    cls = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
    assert abs(_first_run_ge(cls, ts, 1, 3) - 1.0) < 1e-9   # run starts at index 5
    assert _first_run_ge(np.zeros(10, int), ts, 1, 3) is None


def test_smooth_majority_removes_single_frame_noise():
    cls = np.array([0, 0, 1, 0, 0, 2, 2, 2])
    assert list(_smooth(cls, 5, 3)) == [0, 0, 0, 0, 0, 2, 2, 2]


def test_boundary_unknown_when_span_opens_past_it():
    # a DJI clip that begins under canopy: every frame is already past exit/drogue,
    # so those transitions happened before the clip and must not snap to frame 0.
    from tandem.visual.probe import _boundary
    ts = np.arange(6, dtype=float)
    past = np.array([3, 3, 3, 4, 4, 4])
    assert _boundary(past, ts, 1) is None
    assert _boundary(past, ts, 4) == 3.0          # this one is seen in-span
    assert _boundary(np.array([0, 0, 1, 1, 2, 2]), ts, 1) == 2.0
