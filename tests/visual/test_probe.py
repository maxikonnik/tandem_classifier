import json

import numpy as np

from tandem.visual.probe import _first_ge, _PROBE_JSON


def test_probe_weights_present_and_shaped():
    with open(_PROBE_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["dim"] == 384
    assert len(cfg["mu"]) == 384 and len(cfg["sd"]) == 384
    assert len(cfg["W"]) == 4 and len(cfg["W"][0]) == 384   # 4 phases x 384 dims
    assert len(cfg["b"]) == 4
    assert cfg["phases"][2] == "раскрытие"


def test_first_ge_finds_first_sustained_crossing():
    ts = np.array([10, 11, 12, 13, 14, 15, 16], float)
    cls = np.array([0, 0, 1, 1, 2, 2, 3])
    assert _first_ge(cls, ts, 1) == 12.0   # свободное падение opens
    assert _first_ge(cls, ts, 2) == 14.0   # раскрытие opens
    assert _first_ge(np.array([0, 0, 0]), ts[:3], 2) is None
