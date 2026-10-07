"""Where the time goes: every stage between a hand moving and the player seeing, hearing or feeling the answer.

The hub stamps nothing, so its samples are stamped when they ARRIVE (the BLE path puts the stamp ~imu_s after the hand
did it; the camera's poses are aligned to that clock); the screen shows a frame display_s after it was drawn; and so on.
Compensating means placing every effect where it will be PERCEIVED: the ball is drawn where it will be when the light
reaches the eye, the hand is drawn where it will be then, the swing is dated by when the player did it, and the thump and
the sound are sent early so they arrive with the picture.
"""

import dataclasses

import pytest

import config
from pingpong import latency

S = 1_000_000_000


def test_the_defaults_come_from_the_config_so_a_measurement_in_config_local_json_wins():
    lat = latency.Latency.from_config()
    assert lat.imu_s == config.LAT_IMU_S and lat.stroke_s == config.LAT_STROKE_S
    assert lat.display_s == config.LAT_DISPLAY_S and lat.audio_s == config.LAT_AUDIO_S
    assert lat.haptic_s == config.LAT_HAPTIC_S


def test_the_contact_lag_is_the_strokes_length_less_the_time_the_hub_took_to_tell_us():
    lat = latency.Latency(imu_s=0.04, stroke_s=0.14)
    assert lat.contact_lag_s == pytest.approx(0.10)
    assert dataclasses.replace(lat, imu_s=0.0).contact_lag_s == pytest.approx(0.14)


def test_the_screen_is_drawn_ahead_by_the_display_delay_and_half_a_frame():
    lat = latency.Latency(display_s=0.05, loop_s=1 / 60)
    assert lat.view_ahead_s == pytest.approx(0.05 + 1 / 120)


def test_the_defaults_are_plausible_for_this_hardware():
    lat = latency.Latency.from_config()
    assert 0.01 <= lat.imu_s <= 0.10 and 0.08 <= lat.stroke_s <= 0.25 and 0.02 <= lat.display_s <= 0.12
    assert lat.contact_lag_s == pytest.approx(0.16, abs=0.03)           # the stroke the live recordings measured, less the hub's delay


def test_a_negative_delay_is_refused():
    with pytest.raises(ValueError):
        latency.Latency(display_s=-0.01)


# --- the hand, where it will be when the frame is seen -----------------------------------------------------------------
class P:
    def __init__(self, t_s, u, v=0.0, conf=0.9):
        self.t_scene_ns, self.u, self.v, self.conf = round(t_s * S), u, v, conf


LAT = latency.Latency(imu_s=0.04, display_s=0.05, loop_s=0.0)


def moving(speed, n=6, hz=30.0, t_end=1.0, u0=0.0):
    return [P(t_end - (n - 1 - k) / hz, u0 + speed * (k / hz)) for k in range(n)]


def test_a_hand_at_rest_is_drawn_where_it_is():
    poses = [P(1.0 - k / 30, 0.4, -0.2) for k in range(6)][::-1]
    assert latency.predict_hand(poses, round(1.05 * S), LAT) == pytest.approx((0.4, -0.2))


def test_a_moving_hand_is_drawn_ahead_by_its_speed_times_the_whole_delay():
    poses = moving(2.0)                                              # 2 shoulder widths a second to the right
    now = round(1.06 * S)                                            # the newest reading is 60 ms old
    u, v = latency.predict_hand(poses, now, LAT, gain=1.0)
    lead = 0.06 + 0.04 + 0.05                                        # its age + the hub's delay + the display's
    assert u == pytest.approx(poses[-1].u + 2.0 * lead, abs=0.01) and v == pytest.approx(0.0)


def test_the_default_gain_leads_by_less_than_the_full_amount_so_a_turn_does_not_overshoot():
    poses = moving(2.0)
    now = round(1.06 * S)
    full = latency.predict_hand(poses, now, LAT, gain=1.0)[0] - poses[-1].u
    less = latency.predict_hand(poses, now, LAT)[0] - poses[-1].u
    assert 0.5 * full < less < full


def test_the_lead_and_the_speed_are_capped_so_a_glitch_cannot_fling_the_paddle():
    fast = moving(40.0)
    u, _ = latency.predict_hand(fast, round(1.06 * S), LAT)
    assert u - fast[-1].u <= latency.MAX_SPEED_SW_S * latency.MAX_LEAD_S + 1e-9
    stale = moving(2.0)
    far_future = latency.predict_hand(stale, round(3.0 * S), LAT)             # nothing for 2 s: no extrapolation at all
    assert far_future == pytest.approx((stale[-1].u, 0.0))


def test_unsure_readings_are_ignored_and_no_hand_gives_none():
    assert latency.predict_hand([], S, LAT) is None
    assert latency.predict_hand([P(1.0, 0.5, conf=0.2)], S, LAT) is None
    mixed = moving(1.0) + [P(1.1, 9.0, conf=0.1)]
    u, _ = latency.predict_hand(mixed, round(1.15 * S), LAT)
    assert u < 2.0                                                    # the low-confidence outlier was not used
