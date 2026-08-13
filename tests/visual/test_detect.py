from tandem.visual.features import FrameFeature
from tandem.visual.detect import exit_by_background, VisualCue


def _feat(t, structure=0.0, center=0.0, luma=200.0):
    return FrameFeature(t_s=t, mean_luma=luma, blob_area_frac=center,
                        center_fill_frac=center, structure_frac=structure)


def test_exit_at_structure_drop():
    # 10 s of structured aircraft interior, then open sky
    series = [_feat(t, structure=0.4) for t in range(10)]
    series += [_feat(t, structure=0.03) for t in range(10, 20)]
    cue = exit_by_background(series)
    assert cue is not None and cue.type == "exit" and cue.source == "visual"
    assert 9.0 <= cue.t_s <= 12.0


def test_no_exit_when_always_open():
    series = [_feat(t, structure=0.02) for t in range(20)]
    assert exit_by_background(series) is None
