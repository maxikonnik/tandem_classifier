from tandem import video_io as V


def test_gpu_chain_scales_before_rotating_and_converts_to_8_bit():
    c = V._scale_chain(320, 180, True, None, 90)
    assert c.startswith("scale_cuda=180:320") and "format=nv12" in c          # swapped, then turned
    assert "transpose=clock" in c and c.index("hwdownload") < c.index("transpose")
    assert "hflip,vflip" in V._scale_chain(320, 180, True, 2, 180)
    assert V._scale_chain(320, 180, True, 2, 0).startswith("fps=2,scale_cuda=320:180")


def test_cpu_chain_relies_on_ffmpeg_autorotate():
    c = V._scale_chain(320, 180, False, 1, 0)
    assert c.startswith("fps=1,scale=320:180") and "transpose" not in c and c.endswith("showinfo")


def test_rotation_maps_display_matrix_to_clockwise_turn(monkeypatch):
    class R:
        def __init__(self, out): self.stdout = out
    for raw, want in (("-90\n", 90), ("90\n", 270), ("-180\n", 180), ("", 0)):
        V.rotation.cache_clear()
        monkeypatch.setattr(V.subprocess, "run", lambda *a, **k: R(raw))
        assert V.rotation("x.mp4") == want
