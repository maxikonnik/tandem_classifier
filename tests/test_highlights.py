import math

from tandem import highlights as hl
from tandem.highlights import Highlight


def test_shot_scale_bands():
    assert hl.shot_scale(0.0) == "none"
    assert hl.shot_scale(0.05) == "wide"
    assert hl.shot_scale(0.2) == "medium"
    assert hl.shot_scale(0.35) == "close"


def test_scale_moments_finds_close_face_and_wide_runs():
    t = list(range(100, 140))
    frac = [0.05] * 10 + [0.35] * 5 + [0.15] * 5 + [0.55] * 4 + [0.15] * 16
    smile = [0.0] * 20 + [90.0] * 4 + [0.0] * 16
    got = hl.scale_moments(t, frac, 100, 140, smile)
    kinds = {h.kind for h in got}
    assert {"face", "close", "wide"} <= kinds
    face = next(h for h in got if h.kind == "face")
    assert 120 <= face.start_s and face.end_s <= 124
    wide = next(h for h in got if h.kind == "wide")
    assert wide.end_s - wide.start_s <= hl.CLIP_S["wide"] + 1e-9


def test_short_runs_are_ignored():
    t = list(range(0, 20))
    frac = [0.15] * 9 + [0.4] + [0.15] * 10            # one-second close-up
    assert not [h for h in hl.scale_moments(t, frac, 0, 20) if h.kind == "close"]


def test_net_rotation_accumulates_a_turn_and_cancels_bobbing():
    fs = 10.0
    turn = [(math.radians(45), 0.0, 0.0)] * 120          # 45 deg/s for 12 s
    bob = [(math.radians(90) * math.sin(i / 3), 0.0, 0.0) for i in range(120)]
    assert max(hl.net_rotation_deg(turn, fs)) > 350      # 8 s window -> 360 deg
    assert max(hl.net_rotation_deg(bob, fs)) < 90


def test_orbit_needs_rotation_and_the_pair_in_frame():
    fs = 10.0
    rates = [(0.0, 0.0, 0.0)] * 100 + [(math.radians(45), 0.0, 0.0)] * 120 + [(0.0, 0.0, 0.0)] * 100
    rot = hl.net_rotation_deg(rates, fs)
    t = [i / fs for i in range(len(rot))]
    seen = hl.orbit_moments(t, rot, 0, 40, lambda a, b: 1.0, "gyro")
    assert len(seen) == 1 and seen[0].kind == "orbit" and seen[0].score >= hl.ORBIT_MIN_DEG
    assert hl.orbit_moments(t, rot, 0, 40, lambda a, b: 0.3, "gyro") == []   # pair lost: a turn, not an orbit


def test_flow_rotation_converts_pixels_through_the_field_of_view():
    dx = [2.0] * 60                                      # 2 px/frame at 5 fps, 192 px, 120 deg FOV
    deg = hl.flow_rotation_deg(dx, 5.0, 192, 120.0)
    assert abs(deg[0] - 2.0 * 40 * 120 / 192) < 1e-6


def test_select_respects_priority_quota_and_gaps():
    cands = [Highlight("close", 10, 14, 5, "detector"), Highlight("close", 30, 34, 4, "detector"),
             Highlight("close", 50, 54, 3, "detector"), Highlight("face", 12, 15, 90, "detector"),
             Highlight("orbit", 20, 28, 300, "gyro"), Highlight("wide", 27, 31, 5, "detector")]
    got = hl.select(cands)
    kinds = [h.kind for h in got]
    assert kinds.count("close") == 2 and "face" in kinds and "orbit" in kinds
    assert not any(h.kind == "close" and h.start_s == 10 for h in got)   # overlaps the face clip
    assert not any(h.kind == "wide" for h in got)                        # overlaps the orbit clip
    assert [h.start_s for h in got] == sorted(h.start_s for h in got)


def test_from_signals_combines_scale_orbit_and_smile():
    from types import SimpleNamespace
    fs = 10.0
    n = 400                                               # 40 s of free fall from t=100
    sig = SimpleNamespace(fs=fs, t_s=[100 + i / fs for i in range(n)],
                          gx=[math.radians(45) if 150 <= i < 270 else 0.0 for i in range(n)],
                          gy=[0.0] * n, gz=[0.0] * n,
                          smile=[95.0 if 300 <= i < 340 else 0.0 for i in range(n)])
    ts = list(range(100, 140))
    frac = [0.15] * 30 + [0.6] * 4 + [0.15] * 6           # a 4-s face-scale run at 130-134
    got = hl.from_signals(100, 140, ts, frac, sig)
    kinds = {h.kind for h in got}
    assert "orbit" in kinds and "face" in kinds
    orbit = next(h for h in got if h.kind == "orbit")
    assert orbit.source == "gyro" and 113 <= orbit.start_s <= 120
