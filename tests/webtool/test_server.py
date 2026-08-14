from tandem.webtool.server import parse_range, safe_static_path


def test_parse_range_none_when_absent():
    assert parse_range(None, 1000) is None


def test_parse_range_open_ended_clamps_to_size():
    assert parse_range("bytes=500-", 1000) == (500, 999)


def test_parse_range_explicit_end_is_inclusive():
    assert parse_range("bytes=0-99", 1000) == (0, 99)


def test_parse_range_end_beyond_size_is_clamped():
    assert parse_range("bytes=900-5000", 1000) == (900, 999)


def test_safe_static_allows_normal_file():
    p = safe_static_path("app.js")
    assert p is not None and p.endswith("app.js")


def test_safe_static_rejects_traversal():
    assert safe_static_path("../server.py") is None
    assert safe_static_path("../../pyproject.toml") is None
