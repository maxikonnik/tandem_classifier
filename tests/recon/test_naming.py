from tandem.recon.naming import parse_gopro_name, group_recordings


def test_parse_valid_gopro_name():
    n = parse_gopro_name("GX010123.MP4")
    assert n is not None
    assert n.encoding == "GX"
    assert n.chapter == 1
    assert n.file_number == 123
    assert n.ext == "MP4"


def test_parse_rejects_non_gopro():
    assert parse_gopro_name("DJI_0001.MP4") is None
    assert parse_gopro_name("notes.txt") is None


def test_group_orders_chapters_and_flags_unmatched():
    paths = [
        "/a/GX020123.MP4",
        "/a/GX010123.MP4",
        "/a/GX010124.MP4",
        "/a/random.mov",
    ]
    recordings, unmatched = group_recordings(paths)
    by_num = {r.file_number: r for r in recordings}
    assert [p.rsplit("/", 1)[-1] for p in by_num[123].chapters] == [
        "GX010123.MP4",
        "GX020123.MP4",
    ]
    assert by_num[124].chapters == ["/a/GX010124.MP4"]
    assert unmatched == ["/a/random.mov"]
