import os

from tandem.webtool.proxy import ensure_h264_proxy, proxy_path


def test_proxy_path_is_hidden_sibling():
    p = proxy_path(os.path.join("videos", "GX012253.MP4"))
    assert os.path.basename(p) == ".GX012253.proxy.mp4"
    assert os.path.dirname(p) == "videos"


def test_ensure_returns_existing_without_transcoding(tmp_path):
    src = tmp_path / "clip.MP4"
    src.write_bytes(b"x")
    existing = tmp_path / ".clip.proxy.mp4"
    existing.write_bytes(b"proxy")
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)

    result = ensure_h264_proxy(str(src), runner=runner)
    assert result == str(existing)
    assert calls == []  # no transcode when proxy already present


def test_ensure_transcodes_when_missing(tmp_path):
    src = tmp_path / "clip.MP4"
    src.write_bytes(b"x")
    called = {}

    def runner(args, **kwargs):
        called["args"] = args
        open(proxy_path(str(src)), "wb").write(b"proxy")

    result = ensure_h264_proxy(str(src), runner=runner)
    assert result == proxy_path(str(src))
    assert called["args"][0] == "ffmpeg"
    assert "libx264" in called["args"]
