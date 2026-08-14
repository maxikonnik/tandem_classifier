from tandem.webtool.server import parse_range


def test_parse_range_none_when_absent():
    assert parse_range(None, 1000) is None


def test_parse_range_open_ended_clamps_to_size():
    assert parse_range("bytes=500-", 1000) == (500, 999)


def test_parse_range_explicit_end_is_inclusive():
    assert parse_range("bytes=0-99", 1000) == (0, 99)


def test_parse_range_end_beyond_size_is_clamped():
    assert parse_range("bytes=900-5000", 1000) == (900, 999)
