import struct

import pytest

from tandem.recon.dji import _vec3, _find_vec, _ACCEL_TAG, _GYRO_TAG


def _f(field, val):
    """A protobuf wire-type-5 (32-bit) float field."""
    return bytes([field << 3 | 5]) + struct.pack("<f", val)


def _submsg(tag, payload):
    """A length-delimited sub-message with a one-byte length."""
    return bytes([tag, len(payload)]) + payload


def test_vec3_reads_fields_1_2_3():
    sub = _f(1, 0.5) + _f(2, -0.2) + _f(3, 0.8)
    x, y, z = _vec3(sub)
    assert (x, y, z) == pytest.approx((0.5, -0.2, 0.8), abs=1e-6)


def test_vec3_defaults_omitted_component_to_zero():
    # protobuf omits a zero component; field 2 missing must read back as 0.0.
    sub = _f(1, 1.0) + _f(3, 2.0)
    assert _vec3(sub) == pytest.approx((1.0, 0.0, 2.0), abs=1e-6)


def test_find_vec_locates_accel_and_gyro_submessages():
    frame = (b"\x00\x00"
             + _submsg(_ACCEL_TAG, _f(1, 0.5) + _f(2, -0.2) + _f(3, 0.8))
             + _submsg(_GYRO_TAG, _f(1, 1.0) + _f(3, 2.0)))
    assert _find_vec(frame, _ACCEL_TAG) == pytest.approx((0.5, -0.2, 0.8), abs=1e-6)
    assert _find_vec(frame, _GYRO_TAG) == pytest.approx((1.0, 0.0, 2.0), abs=1e-6)


def test_find_vec_returns_none_when_tag_absent():
    frame = _submsg(_ACCEL_TAG, _f(1, 0.1) + _f(2, 0.1) + _f(3, 0.1))
    assert _find_vec(frame, _GYRO_TAG) is None
