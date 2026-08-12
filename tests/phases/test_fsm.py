from tandem.phases.detect import Segment
from tandem.phases.fsm import validate_order, PHASE_ORDER


def _seg(t, s, e):
    return Segment(t, s, e, "telemetry", 0.9)


def test_valid_order_passes():
    segs = [_seg("ground_pre", 0.0, 5.0), _seg("climb", 5.0, 12.0),
            _seg("freefall", 12.0, 30.0)]
    assert validate_order(segs) is None


def test_freefall_before_climb_violates():
    segs = [_seg("freefall", 0.0, 10.0), _seg("climb", 10.0, 20.0)]
    assert validate_order(segs) == "PHASE_ORDER_VIOLATION"


def test_out_of_time_order_violates():
    segs = [_seg("ground_pre", 5.0, 10.0), _seg("climb", 0.0, 4.0)]
    assert validate_order(segs) == "PHASE_ORDER_VIOLATION"


def test_empty_is_ok():
    assert validate_order([]) is None
