import tandem


def test_version_is_a_string():
    assert isinstance(tandem.__version__, str)
    assert tandem.__version__
