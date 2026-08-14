import shutil

import pytest

from tandem.recon.ffprobe import keyframe_interval_stats, find_gpmf_stream_index, has_ffmpeg, _parse_packet_times


def test_parse_packet_times_extracts_pairs_and_skips_incomplete():
    data = {
        "packets": [
            {"pts_time": "0.000000", "duration_time": "1.001000"},
            {"pts_time": "1.001000", "duration_time": "1.001000"},
            {"pts_time": "2.002000"},  # missing duration -> skipped
        ]
    }
    assert _parse_packet_times(data) == [(0.0, 1.001), (1.001, 1.001)]


def test_interval_stats_from_pts():
    stats = keyframe_interval_stats([0.0, 0.5, 1.0, 1.5])
    assert stats.count == 4
    assert stats.median_s == pytest.approx(0.5)
    assert stats.min_s == pytest.approx(0.5)
    assert stats.max_s == pytest.approx(0.5)


def test_interval_stats_empty_is_safe():
    stats = keyframe_interval_stats([])
    assert stats.count == 0
    assert stats.median_s == 0.0


def test_find_gpmf_stream_index_from_streams():
    # exercise the pure selection helper via a monkeypatched streams source
    from tandem.recon import ffprobe

    fake = [
        {"index": 0, "codec_type": "video", "codec_tag_string": "hvc1"},
        {"index": 3, "codec_type": "data", "codec_tag_string": "gpmd"},
    ]
    assert ffprobe._select_gpmf_index(fake) == 3


@pytest.mark.integration
@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg/ffprobe not installed")
def test_streams_on_generated_file(tmp_path):
    import subprocess

    sample = tmp_path / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
         str(sample)],
        check=True, capture_output=True,
    )
    assert find_gpmf_stream_index(str(sample)) is None  # generated file has no GPMF
