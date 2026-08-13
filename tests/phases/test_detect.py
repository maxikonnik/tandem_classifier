from tandem.phases.signals import Signals
from tandem.phases.detect import detect_exit, detect_freefall, detect_ground_climb, Segment, G


def _in_aircraft_then_freefall(fs=10.0):
    # 10 s ground+climb at ~1 g, low then plane speed; 2 s exit dropout;
    # 20 s freefall at ~1 g noisy, speed ~55.
    accel, amin, speed = [], [], []
    for _ in range(int(10 * fs)):      # in aircraft
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(40.0)
    for _ in range(int(2 * fs)):       # exit: |a| ~0
        accel.append(0.05 * G); amin.append(0.05 * G); speed.append(50.0)
    for _ in range(int(20 * fs)):      # freefall
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(55.0)
    t = [i / fs for i in range(len(accel))]
    return Signals(t_s=t, accel_mag=accel, accel_min=amin, speed_3d=speed, fs=fs,
                   has_accel=True, has_gps=True)


def test_detect_exit_at_dropout():
    sig = _in_aircraft_then_freefall()
    ev = detect_exit(sig)
    assert ev is not None
    assert ev.type == "exit"
    assert ev.source == "telemetry"
    assert 9.5 <= ev.t_s <= 12.5     # within the dropout window (~10-12 s)
    assert ev.confidence > 0.5


def test_detect_exit_none_when_no_dropout():
    fs = 10.0
    accel = [1.0 * G] * int(30 * fs)   # steady 1 g, never drops
    amin = [0.9 * G] * int(30 * fs)
    speed = [40.0] * int(30 * fs)
    t = [i / fs for i in range(len(accel))]
    sig = Signals(t_s=t, accel_mag=accel, accel_min=amin, speed_3d=speed, fs=fs,
                  has_accel=True, has_gps=True)
    assert detect_exit(sig) is None


def test_detect_freefall_after_exit():
    sig = _in_aircraft_then_freefall()
    ev = detect_exit(sig)
    seg = detect_freefall(sig, ev)
    assert seg is not None
    assert seg.type == "freefall"
    assert seg.source == "telemetry"
    assert seg.start_s >= ev.t_s - 0.5
    assert seg.end_s > seg.start_s
    # freefall spans roughly the last 20 s (12..32)
    assert seg.end_s - seg.start_s >= 15.0


def test_detect_freefall_none_without_exit():
    sig = _in_aircraft_then_freefall()
    assert detect_freefall(sig, None) is None


def test_ground_then_climb_before_exit():
    fs = 10.0
    accel, amin, speed = [], [], []
    for _ in range(int(5 * fs)):       # ground: stationary
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(0.5)
    for _ in range(int(5 * fs)):       # climb: plane moving
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(40.0)
    for _ in range(int(2 * fs)):       # exit dropout
        accel.append(0.05 * G); amin.append(0.05 * G); speed.append(50.0)
    for _ in range(int(10 * fs)):      # freefall
        accel.append(1.0 * G); amin.append(0.9 * G); speed.append(55.0)
    t = [i / fs for i in range(len(accel))]
    sig = Signals(t_s=t, accel_mag=accel, accel_min=amin, speed_3d=speed, fs=fs,
                  has_accel=True, has_gps=True)
    ev = detect_exit(sig)
    segs = detect_ground_climb(sig, ev)
    types = [s.type for s in segs]
    assert types == ["ground_pre", "climb"]
    assert segs[0].start_s == 0.0
    assert segs[0].end_s <= 5.5          # ground ends ~5 s
    assert segs[1].end_s <= ev.t_s + 0.2 # climb ends at exit


def test_ground_climb_empty_without_exit():
    sig = _in_aircraft_then_freefall()
    assert detect_ground_climb(sig, None) == []


def _accel_only_flight(fs=10.0):
    amin, amag = [], []
    for _ in range(int(10 * fs)):      # in aircraft ~1g
        amin.append(0.9 * G); amag.append(1.0 * G)
    for _ in range(int(3 * fs)):       # exit: deep dip, >1s
        amin.append(0.1 * G); amag.append(0.3 * G)
    for _ in range(int(40 * fs)):      # freefall ~1g, noisy
        amin.append(0.8 * G); amag.append(1.0 * G)
    for _ in range(int(3 * fs)):       # opening shock ~3.5g (towers over buffeting)
        amin.append(3.0 * G); amag.append(3.5 * G)
    t = [i / fs for i in range(len(amin))]
    speed = [0.0] * len(amin)          # NO GPS
    return Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=speed,
                   fs=fs, has_accel=True, has_gps=False)


def test_ground_climb_without_gps_uses_accel_stillness():
    # No GPS: split the pre-exit span by accelerometer stillness instead of
    # speed. A steady, low-variance prefix (parked, engine off) becomes
    # ground_pre; the noisier taxi/climb vibration after it is climb.
    fs = 10.0
    amag, amin = [], []
    for i in range(int(5 * fs)):        # ground: rock-steady ~1g (tiny ripple)
        amag.append(G + (0.005 * G if i % 2 == 0 else -0.005 * G))
        amin.append(0.95 * G)
    for i in range(int(5 * fs)):        # climb: noisy engine/airframe vibration
        amag.append(G + (0.35 * G if i % 2 == 0 else -0.35 * G))
        amin.append(0.6 * G)
    for _ in range(int(2 * fs)):        # exit dropout
        amag.append(0.3 * G); amin.append(0.1 * G)
    for _ in range(int(15 * fs)):       # freefall
        amag.append(1.0 * G); amin.append(0.8 * G)
    t = [i / fs for i in range(len(amag))]
    speed = [0.0] * len(amag)           # NO GPS
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=speed,
                  fs=fs, has_accel=True, has_gps=False)
    ev = detect_exit(sig)
    assert ev is not None
    segs = detect_ground_climb(sig, ev)
    types = [s.type for s in segs]
    assert types == ["ground_pre", "climb"]
    assert segs[0].start_s == 0.0
    assert segs[0].end_s <= 5.5           # ground ends ~5 s
    assert segs[1].end_s <= ev.t_s + 0.2  # climb ends at exit
    assert all(s.source == "telemetry" for s in segs)


def test_ground_climb_without_gps_no_stationary_prefix_is_all_climb():
    # No GPS and no detectable stationary prefix (noisy the whole way):
    # the whole pre-exit span is emitted as a single climb segment.
    fs = 10.0
    amag, amin = [], []
    for i in range(int(8 * fs)):        # noisy the whole way, never settles
        amag.append(G + (0.35 * G if i % 2 == 0 else -0.35 * G))
        amin.append(0.6 * G)
    for _ in range(int(2 * fs)):        # exit dropout
        amag.append(0.3 * G); amin.append(0.1 * G)
    for _ in range(int(15 * fs)):       # freefall
        amag.append(1.0 * G); amin.append(0.8 * G)
    t = [i / fs for i in range(len(amag))]
    speed = [0.0] * len(amag)           # NO GPS
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=speed,
                  fs=fs, has_accel=True, has_gps=False)
    ev = detect_exit(sig)
    assert ev is not None
    segs = detect_ground_climb(sig, ev)
    types = [s.type for s in segs]
    assert types == ["climb"]
    assert segs[0].start_s == 0.0
    assert segs[0].end_s <= ev.t_s + 0.2


def test_ground_climb_without_gps_flat_low_g_prefix_is_not_ground():
    # A flat, low-variance prefix that sits nowhere near 1g (e.g. a stuck or
    # saturated sensor, or a miscalibrated offset) must NOT be classified as
    # ground_pre — stillness alone (low std) isn't enough; the level must
    # also be near 1g. Followed by a genuinely steady climb so the exit
    # jerk-gate still fires correctly at the real dropout.
    fs = 10.0
    amag, amin = [], []
    for _ in range(int(3 * fs)):        # flat but at ~0.1g, not ~1g
        amag.append(0.1 * G); amin.append(0.1 * G)
    for i in range(int(5 * fs)):        # genuine noisy ~1g climb before exit
        amag.append(G + (0.35 * G if i % 2 == 0 else -0.35 * G))
        amin.append(0.6 * G)
    for _ in range(int(2 * fs)):        # exit dropout
        amag.append(0.3 * G); amin.append(0.1 * G)
    for _ in range(int(15 * fs)):       # freefall
        amag.append(1.0 * G); amin.append(0.8 * G)
    t = [i / fs for i in range(len(amag))]
    speed = [0.0] * len(amag)           # NO GPS
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=speed,
                  fs=fs, has_accel=True, has_gps=False)
    ev = detect_exit(sig)
    assert ev is not None
    segs = detect_ground_climb(sig, ev)
    types = [s.type for s in segs]
    assert "ground_pre" not in types    # the low-g flat prefix must not be "ground"
    assert types == ["climb"]
    assert segs[0].start_s == 0.0
    assert segs[0].end_s <= ev.t_s + 0.2


def test_detect_exit_from_accel_alone_without_gps():
    sig = _accel_only_flight()
    ev = detect_exit(sig)
    assert ev is not None
    assert ev.type == "exit" and ev.source == "telemetry"
    assert 9.5 <= ev.t_s <= 12.0       # at the dip onset (~10s)


def test_brief_dip_is_not_an_exit():
    fs = 10.0
    amin = [0.9 * G] * int(20 * fs)
    for k in range(3):                  # 0.3s blip, shorter than EXIT_MIN_DURATION_S
        amin[int(10 * fs) + k] = 0.1 * G
    amag = [1.0 * G] * len(amin)
    t = [i / fs for i in range(len(amin))]
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=[0.0] * len(amin),
                  fs=fs, has_accel=True, has_gps=False)
    assert detect_exit(sig) is None


def test_gradual_low_g_without_prior_1g_is_not_exit():
    # A recording already in a low-g state (never a steady ~1g before): the jerk
    # gate must reject it — there was no rapid drop FROM 1g.
    fs = 10.0
    amin = [0.2 * G] * int(20 * fs)
    amag = [0.3 * G] * int(20 * fs)
    t = [i / fs for i in range(len(amin))]
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=[0.0] * len(amin),
                  fs=fs, has_accel=True, has_gps=False)
    assert detect_exit(sig) is None


def test_freefall_ends_at_opening_shock():
    sig = _accel_only_flight()          # exit ~10s, freefall 13-53s, shock 53-56s
    ev = detect_exit(sig)
    seg = detect_freefall(sig, ev)
    assert seg is not None and seg.type == "freefall"
    assert seg.start_s <= ev.t_s + 0.5
    # freefall must END at the opening shock (~53s), not run to the recording end (~56s)
    assert 51.0 <= seg.end_s <= 54.0


def test_freefall_argmax_ignores_earlier_smaller_spike():
    # Prove that argmax chooses the LARGEST peak, not the first spike above threshold.
    # Fixture: exit ~10s, freefall with an in-air maneuver spike (~2.6g) at ~30s,
    # then continued freefall, then a real opening shock (~3.5g) at ~55s.
    fs = 10.0
    amin, amag = [], []
    # 0-10s: in aircraft ~1g, with real dip at exit
    for _ in range(int(10 * fs)):
        amin.append(0.9 * G); amag.append(1.0 * G)
    # 10-13s: exit dropout, >1s dip
    for _ in range(int(3 * fs)):
        amin.append(0.1 * G); amag.append(0.3 * G)
    # 13-30s: freefall at ~1g
    for _ in range(int(17 * fs)):
        amin.append(0.8 * G); amag.append(1.0 * G)
    # 30-31s: in-air maneuver spike ~2.6g (above threshold but brief and smaller)
    for _ in range(int(1 * fs)):
        amin.append(2.3 * G); amag.append(2.6 * G)
    # 31-55s: continued freefall at ~1g
    for _ in range(int(24 * fs)):
        amin.append(0.8 * G); amag.append(1.0 * G)
    # 55-58s: real opening shock ~3.5g (larger than the maneuver spike)
    for _ in range(int(3 * fs)):
        amin.append(3.0 * G); amag.append(3.5 * G)
    t = [i / fs for i in range(len(amin))]
    speed = [0.0] * len(amin)  # NO GPS
    sig = Signals(t_s=t, accel_mag=amag, accel_min=amin, speed_3d=speed,
                  fs=fs, has_accel=True, has_gps=False)
    ev = detect_exit(sig)
    seg = detect_freefall(sig, ev)
    assert seg is not None and seg.type == "freefall"
    assert seg.start_s <= ev.t_s + 0.5
    # freefall must END at the REAL shock (~55s), NOT the earlier maneuver spike (~30s)
    assert 53.0 <= seg.end_s <= 57.0
