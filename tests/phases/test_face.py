import struct

from tandem.recon.gpmf import KLV
from tandem.phases.signals import _face_values


def _face_klv(faces):
    """Build a FACE KLV from (blink, smile, w, h) tuples."""
    payload = b"".join(
        struct.pack(">BBHHHHHBB", 4, 99, 0, 100, 200, w, h, blink, smile)
        for (blink, smile, w, h) in faces
    )
    return KLV(key="FACE", type="?", sample_size=14, repeat=len(faces), payload=payload)


def test_face_values_decodes_count_smile_blink():
    face = _face_klv([(10, 90, 300, 400)])
    n, smile, blink, area = _face_values([face])
    assert n == 1.0
    assert smile == 90.0
    assert blink == 10.0
    assert 0.0 < area < 1.0


def test_face_values_takes_max_over_faces():
    face = _face_klv([(5, 30, 100, 100), (40, 88, 500, 500)])
    n, smile, blink, area = _face_values([face])
    assert n == 2.0
    assert smile == 88.0   # max smile across faces
    assert blink == 40.0   # max blink across faces


def test_face_values_none_without_face_stream():
    assert _face_values([KLV(key="ACCL", type="s", sample_size=6, repeat=1, payload=b"\x00" * 6)]) is None


def test_face_values_empty_when_no_faces():
    face = KLV(key="FACE", type="?", sample_size=14, repeat=0, payload=b"")
    n, smile, blink, area = _face_values([face])
    assert n == 0.0 and smile == 0.0 and blink == 0.0 and area == 0.0
