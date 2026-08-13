import numpy as np
from PIL import Image
from tandem.visual.series import build_series, load_gray


def test_load_gray_returns_2d(tmp_path):
    p = tmp_path / "x.jpg"
    Image.fromarray(np.full((40, 60, 3), 128, np.uint8)).save(p)
    g = load_gray(str(p))
    assert g.ndim == 2 and g.shape == (40, 60)


def test_build_series_orders_by_index(tmp_path):
    for i, val in [(0, 220), (1, 10), (2, 220)]:
        Image.fromarray(np.full((50, 50, 3), val, np.uint8)).save(tmp_path / f"kf_{i}.jpg")
    series = build_series(str(tmp_path))
    assert [round(f.t_s) for f in series] == [0, 1, 2]
    assert series[1].blob_area_frac == 0.0   # a uniform (dark) frame has no *relative* dark pixels
