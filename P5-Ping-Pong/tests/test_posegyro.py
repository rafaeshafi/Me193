"""PoseGyro: the camera as a virtual gyro, the swing source when the hub's IMU is too slow.

The second half runs a scripted hand through the VisionWorker's own filter (One-Euro), PoseGyro and the
real SwingDetector with the camera-source parameters, at the frame rates a webcam really delivers.
"""

import random

import pytest

from pingpong import fakerig, posegyro
from pingpong.calibration import SwingCalibration
from pingpong.events import PaddlePose
from pingpong.oneeuro import OneEuro2D
from pingpong.swing import SwingDetector

S = 1_000_000_000


def pose(t_s, u, v=0.0):
    return PaddlePose(t_scene_ns=round(t_s * S), u=u, v=v, conf=0.9, hand="right")


def test_one_shoulder_width_per_second_reads_as_a_hundred_degrees_per_second():
    pg = posegyro.PoseGyro()
    samples = [pg.feed(pose(k / 30, k / 30)) for k in range(5)]          # u = t: 1 shoulder width per second
    last = samples[-1]
    assert last.g[0] / posegyro.GYRO_PER_DPS == pytest.approx(100.0, abs=0.3)
    assert last.g[1] == 0 and last.g[2] == 0
    assert last.a == (0, 0, round(posegyro.ACCEL_PER_G)) and last.src == "pose"


def test_vertical_hand_motion_lands_on_the_second_axis_and_direction_is_kept():
    pg = posegyro.PoseGyro()
    samples = [pg.feed(pose(k / 30, 0.0, -k / 30)) for k in range(4)]    # moving down
    assert samples[-1].g[0] == 0 and samples[-1].g[1] < 0


def test_the_first_pose_and_a_pose_after_a_long_gap_read_as_standing_still():
    pg = posegyro.PoseGyro()
    assert pg.feed(pose(1.0, 0.0)).g == (0, 0, 0)
    pg.feed(pose(1.033, 0.02))
    after_gap = pg.feed(pose(1.8, 0.9))                                 # tracking was lost: a jump, not a speed
    assert after_gap.g == (0, 0, 0)


def test_a_pose_that_is_not_newer_is_skipped():
    pg = posegyro.PoseGyro()
    pg.feed(pose(1.0, 0.0))
    assert pg.feed(pose(1.0, 0.5)) is None and pg.feed(pose(0.9, 0.5)) is None


def test_samples_are_stamped_a_little_before_their_pose_because_the_filtered_speed_peaks_late():
    pg = posegyro.PoseGyro()
    sample = pg.feed(pose(2.0, 0.0))
    assert sample.t_ns == 2 * S - round(posegyro.DELAY_S * S)


def test_a_speed_beyond_the_raw_range_is_clipped_not_wrapped():
    pg = posegyro.PoseGyro()
    pg.feed(pose(1.0, 0.0))
    sample = pg.feed(pose(1.033, 50.0))                                  # 1500 shoulder widths per second
    assert sample.g[0] == posegyro.FS_RAW


# --- the whole chain on a scripted hand ---------------------------------------------------------------------------------
def calibration():
    return SwingCalibration(u_fwd=(1.0, 0.0, 0.0), omega_lo=200.0, omega_hi=700.0, source="pose")


def detect(swings=(), *, fps=30.0, seconds=9.0, glides=(), jump_at=None, seed=1):
    """IMPACT events for a hand that swings at the given (t0, duration, peak speed[, backswing ratio]) tuples."""
    rng, filt, pg = random.Random(seed), OneEuro2D(), posegyro.PoseGyro()
    params = calibration().swing_params(posegyro.GYRO_PER_DPS, posegyro.ACCEL_PER_G, posegyro.FS_RAW)
    detector, events = SwingDetector(params), []
    for k in range(int(seconds * fps)):
        t = k / fps
        u = sum(fakerig.swing_offset(t, *sw[:3], back=sw[3] if len(sw) > 3 else 0.5) for sw in swings)
        u += sum(d * fakerig._smoothstep((t - a) / (b - a)) for a, b, d in glides)
        if jump_at is not None and abs(t - jump_at) < 0.5 / fps:
            u += 0.6                                                     # one frame of a landmark gone astray
        fu, fv = filt((u + rng.gauss(0, 0.004), rng.gauss(0, 0.004)), 1.0 + t)
        sample = pg.feed(PaddlePose(t_scene_ns=round((1.0 + t) * S), u=fu, v=fv, conf=0.9, hand="right"))
        events += [e for e in detector.feed(sample) if e.kind == "IMPACT"]
    return events


def test_a_deliberate_swing_is_found_once_close_to_its_real_peak():
    events = detect([(3.0, 0.15, 6.0)])
    assert len(events) == 1
    e = events[0]
    assert abs(e.t_ns / S - 1.0 - 3.075) < 0.06                          # the real peak is mid-stroke
    assert 300.0 < e.w_pk < 900.0 and e.src == "pose" and e.n_reversals <= 1


@pytest.mark.parametrize("duration", (0.10, 0.15, 0.25))
@pytest.mark.parametrize("peak_speed", (3.0, 4.0, 6.0, 9.0))
def test_swings_of_every_realistic_speed_and_length_are_found_at_30_fps(duration, peak_speed):
    assert len(detect([(3.0, duration, peak_speed)])) == 1


@pytest.mark.parametrize("duration", (0.15, 0.25))
@pytest.mark.parametrize("peak_speed", (4.0, 6.0, 9.0))
def test_and_still_found_when_a_dim_room_drops_the_camera_to_20_fps(duration, peak_speed):
    assert len(detect([(3.0, duration, peak_speed)], fps=20.0)) == 1


def test_a_run_of_swings_gives_one_event_per_swing_in_order():
    events = detect([(2.0, 0.15, 5.0), (4.0, 0.15, 7.0), (6.0, 0.20, 4.0)])
    times = [e.t_ns / S - 1.0 for e in events]
    assert len(events) == 3 and all(abs(t - want) < 0.1 for t, want in zip(times, (2.075, 4.075, 6.1)))


def test_a_backswing_alone_never_fires():
    assert detect([(3.0, 0.15, -5.0, 0.0)]) == []                        # a stroke against the swing axis only


def test_a_resting_hand_never_fires():
    assert detect([]) == []


def test_slow_repositioning_is_not_a_swing():
    assert detect(glides=[(2.0, 2.8, 1.2), (5.0, 5.5, -1.2)]) == []     # a 0.8 s glide across the body


def test_a_single_frame_landmark_jump_is_not_a_swing():
    assert detect(jump_at=3.0) == []


def test_the_recovery_after_a_swing_does_not_fire_a_second_time():
    assert len(detect([(3.0, 0.15, 8.0)], seconds=7.0)) == 1
