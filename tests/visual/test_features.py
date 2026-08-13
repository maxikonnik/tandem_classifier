import numpy as np
from tandem.visual.features import frame_features, FrameFeature


def test_bright_smooth_frame_has_low_blob_and_structure():
    gray = np.full((90, 160), 220, dtype=np.uint8)   # uniform bright sky
    f = frame_features(gray, t_s=1.0)
    assert f.t_s == 1.0
    assert f.blob_area_frac == 0.0          # nothing dark
    assert f.structure_frac == 0.0          # no edges
    assert f.mean_luma > 200


def test_central_dark_object_fills_centre():
    gray = np.full((100, 100), 220, dtype=np.uint8)
    gray[40:60, 40:60] = 10                 # dark 20x20 object at centre
    f = frame_features(gray)
    assert f.blob_area_frac > 0.0
    assert f.center_fill_frac > f.blob_area_frac   # concentrated at centre
    assert f.structure_frac > 0.0                  # its edges register


def test_structured_frame_has_high_structure_frac():
    checker = np.zeros((100, 100), dtype=np.uint8)
    checker[::2, :] = 255                    # alternating rows -> many edges
    f = frame_features(checker)
    assert f.structure_frac > 0.3
