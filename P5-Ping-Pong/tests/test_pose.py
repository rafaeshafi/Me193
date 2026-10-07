"""Paddle point from MediaPipe landmarks (shoulder-width units, player-right positive)."""

from types import SimpleNamespace

import pytest

from pingpong import pose

W, H = 1280, 720
L_SH, R_SH, L_PINKY, R_PINKY, L_WRIST, R_WRIST, L_INDEX, R_INDEX = 11, 12, 17, 18, 15, 16, 19, 20


def landmarks(sh_x=(0.6, 0.4), sh_y=0.4, vis=0.95, **points):
    """33 landmarks; shoulders at sh_x (left, right) as seen in the image; extra points by index."""
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    lm[L_SH] = SimpleNamespace(x=sh_x[0], y=sh_y, visibility=vis)
    lm[R_SH] = SimpleNamespace(x=sh_x[1], y=sh_y, visibility=vis)
    for idx, (x, y, v) in points.items():
        lm[int(idx.lstrip("p"))] = SimpleNamespace(x=x, y=y, visibility=v)
    return lm


def hand(lm, idx, x, y, vis=0.9):
    lm[idx] = SimpleNamespace(x=x, y=y, visibility=vis)


def test_right_hand_to_the_players_right_is_positive_u_and_raising_it_is_positive_v():
    lm = landmarks()
    hand(lm, R_WRIST, 0.30, 0.40)               # image-left of the right shoulder = player's right
    u, v, conf = pose.paddle_uv(lm, "right", W, H)
    assert u == pytest.approx(1.0) and v == pytest.approx(0.0)
    hand(lm, R_WRIST, 0.30, 0.20)
    _, v2, _ = pose.paddle_uv(lm, "right", W, H)
    assert v2 == pytest.approx((0.4 - 0.2) * H / (0.2 * W))


def test_the_paddle_point_blends_wrist_index_and_pinky():
    lm = landmarks()
    hand(lm, R_WRIST, 0.30, 0.40)
    hand(lm, R_INDEX, 0.26, 0.40)
    hand(lm, R_PINKY, 0.26, 0.40)
    u, _, _ = pose.paddle_uv(lm, "right", W, H)
    expected_x = 0.5 * 0.30 + 0.25 * 0.26 + 0.25 * 0.26
    assert u == pytest.approx((0.5 - expected_x) * W / (0.2 * W))


def test_the_wrist_alone_is_enough_when_the_fingers_are_hidden():
    lm = landmarks()
    hand(lm, R_WRIST, 0.30, 0.40)
    hand(lm, R_INDEX, 0.10, 0.10, vis=0.1)
    assert pose.paddle_uv(lm, "right", W, H)[0] == pytest.approx(1.0)


def test_left_hand_uses_the_left_landmarks():
    lm = landmarks()
    hand(lm, L_WRIST, 0.70, 0.40)               # player's left
    u, _, _ = pose.paddle_uv(lm, "left", W, H)
    assert u == pytest.approx(-1.0)


def test_confidence_is_the_weakest_of_the_wrist_and_both_shoulders():
    lm = landmarks(vis=0.8)
    hand(lm, R_WRIST, 0.3, 0.4, vis=0.65)
    assert pose.paddle_uv(lm, "right", W, H)[2] == pytest.approx(0.65)


def test_a_hidden_shoulder_or_wrist_means_no_reading():
    lm = landmarks()
    hand(lm, R_WRIST, 0.3, 0.4, vis=0.2)
    assert pose.paddle_uv(lm, "right", W, H) is None
    lm2 = landmarks(vis=0.2)
    hand(lm2, R_WRIST, 0.3, 0.4)
    assert pose.paddle_uv(lm2, "right", W, H) is None


def test_shoulders_too_close_together_are_rejected_instead_of_amplifying_noise():
    lm = landmarks(sh_x=(0.51, 0.49))           # side-on or far away: 2% of the frame width
    hand(lm, R_WRIST, 0.3, 0.4)
    assert pose.paddle_uv(lm, "right", W, H) is None


def test_shoulder_width_is_reported_for_the_one_player_lock():
    lm = landmarks()
    assert pose.shoulder_width_norm(lm, W, H) == pytest.approx(0.2)
    assert pose.shoulder_width_norm(landmarks(vis=0.1), W, H) is None


def test_the_lock_accepts_a_turned_or_nearer_player_and_rejects_a_stranger():
    # a stroke turns the torso and the shoulders look narrower: a symmetric +-25% lock threw away 32% of the player's
    # own frames in the first live game (pose gaps of up to 1.5 s, the ball clock paused nine times in 78 s)
    lock = pose.PoseLock()
    assert lock.accepts(0.50) is True            # not calibrated yet: nothing to compare with
    lock.calibrate(0.20)
    assert lock.accepts(0.20) and lock.accepts(0.25) and lock.accepts(0.13) and lock.accepts(0.11)
    assert not lock.accepts(0.27) and not lock.accepts(0.09)
    assert lock.accepts(None) is False and lock.unit == 0.20


def test_a_calibrated_unit_keeps_a_turned_torso_from_stretching_the_hand_coordinates():
    lm = landmarks(sh_x=(0.55, 0.45))            # shoulders at half the calibrated 0.20 width: the torso is turned
    hand(lm, R_WRIST, 0.30, 0.40)
    u_now, _, _ = pose.paddle_uv(lm, "right", W, H)
    u_fixed, _, _ = pose.paddle_uv(lm, "right", W, H, unit=0.20)
    assert u_now == pytest.approx(2.0) and u_fixed == pytest.approx(1.0)       # (0.5 - 0.3) / 0.10 against / 0.20


def test_without_a_calibration_the_unit_is_still_the_shoulder_width_of_the_moment():
    assert pose.PoseLock().unit is None
    lm = landmarks()
    hand(lm, R_WRIST, 0.30, 0.40)
    assert pose.paddle_uv(lm, "right", W, H, unit=None) == pose.paddle_uv(lm, "right", W, H)
