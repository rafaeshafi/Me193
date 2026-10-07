"""The body as the hand's reference: where the shoulders are and how big a shoulder width is, steady and robust.

The first live games threw away 20-39% of the camera's frames in the one-player lock, mostly one frame at a time: it
compared every frame's shoulder width with ONE width captured at calibration, so a player who stands a little nearer than
at calibration, or leans in, or turns the torso in a stroke, was refused again and again.  The tracker follows the
player's distance and ignores turns, holds the anchor through a hidden shoulder, and refuses only a body that really is
someone else (it jumps, or it is a different size and stays that way).
"""

import math
from types import SimpleNamespace

import pytest

from pingpong import body

W, H = 640, 360
ASPECT = H / W
DT = 1 / 30


def lm(sh_x=(0.6, 0.4), sh_y=0.4, vis=0.95, wrist=(0.30, 0.40), wrist_vis=0.95, fingers=None):
    out = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    out[11] = SimpleNamespace(x=sh_x[0], y=sh_y, visibility=vis)
    out[12] = SimpleNamespace(x=sh_x[1], y=sh_y, visibility=vis)
    out[16] = SimpleNamespace(x=wrist[0], y=wrist[1], visibility=wrist_vis)
    for idx, (x, y, v) in (fingers or {}).items():
        out[idx] = SimpleNamespace(x=x, y=y, visibility=v)
    return out


def sh_at(width, centre=0.5):
    return (centre + width / 2, centre - width / 2)


def run(tracker, frames, t0=0.0):
    """frames: landmark lists, one per 1/30 s; -> the Body (or None) of each."""
    return [tracker.update(f, t0 + k * DT, ASPECT) for k, f in enumerate(frames)]


# --- the body --------------------------------------------------------------------------------------------------------------
def test_a_steady_player_is_followed_with_the_shoulder_midpoint_and_one_shoulder_width_as_the_unit():
    t = body.BodyTracker(ref=0.2)
    out = run(t, [lm() for _ in range(20)])
    assert all(b is not None and not b.held for b in out)
    last = out[-1]
    assert last.cx == pytest.approx(0.5, abs=1e-6) and last.cy == pytest.approx(0.4, abs=1e-6)
    assert last.unit == pytest.approx(0.2, abs=1e-3) and last.conf == pytest.approx(0.95)


def test_a_player_who_stands_nearer_than_at_calibration_is_not_refused_and_the_unit_follows_the_distance():
    # calibrated at 0.13 of the frame, playing at 0.17 (30% nearer): the old lock refused everything above 1.3 x
    t = body.BodyTracker(ref=0.13)
    out = run(t, [lm(sh_x=sh_at(0.17)) for _ in range(90)])
    assert all(b is not None for b in out)
    assert out[-1].unit == pytest.approx(0.17, abs=0.003)
    assert t.counts["rejected"] == 0


def test_a_step_towards_the_camera_is_followed_within_a_couple_of_seconds():
    t = body.BodyTracker(ref=0.2)
    frames = [lm(sh_x=sh_at(0.2)) for _ in range(60)] + [lm(sh_x=sh_at(0.26)) for _ in range(120)]
    out = run(t, frames)
    assert all(b is not None for b in out)
    assert out[59].unit == pytest.approx(0.2, abs=0.003) and out[-1].unit == pytest.approx(0.26, abs=0.004)


def test_turning_the_torso_in_every_stroke_does_not_shrink_the_unit_or_cost_a_frame():
    # a stroke turns the shoulders away: they look narrow for about 0.3 s of every second
    t = body.BodyTracker(ref=0.2)
    frames = []
    for second in range(4):
        frames += [lm(sh_x=sh_at(0.2)) for _ in range(20)]
        frames += [lm(sh_x=sh_at(w)) for w in (0.16, 0.12, 0.09, 0.08, 0.09, 0.12, 0.15, 0.18, 0.19, 0.2)]
    out = run(t, frames)
    assert all(b is not None for b in out)
    assert all(b.unit == pytest.approx(0.2, abs=0.01) for b in out[30:])             # the frontal width, not the turned one
    assert t.counts["rejected"] == 0


def test_a_hidden_shoulder_holds_the_anchor_for_a_moment_then_gives_up():
    t = body.BodyTracker(ref=0.2)
    run(t, [lm() for _ in range(15)])
    hidden = [lm(vis=0.1) for _ in range(20)]                                          # 0.67 s
    out = run(t, hidden, t0=15 * DT)
    assert all(b is not None and b.held for b in out)
    assert out[0].cx == pytest.approx(0.5, abs=1e-6) and out[0].conf < 0.95            # less sure of it, but it is there
    long_out = run(t, [lm(vis=0.1) for _ in range(40)], t0=35 * DT)
    assert long_out[-1] is None and t.reason == "no_shoulders"
    back = t.update(lm(), 100.0, ASPECT)
    assert back is not None and not back.held                                          # and it recovers


def test_a_player_stepping_sideways_is_followed_not_refused():
    t = body.BodyTracker(ref=0.2)
    frames = [lm(sh_x=sh_at(0.2, centre=0.5 - 0.002 * k)) for k in range(90)]          # 0.06 of the frame a second
    assert all(b is not None for b in run(t, frames))


def test_the_shoulder_noise_is_smoothed_out_of_the_anchor():
    import random

    rnd = random.Random(1)
    t = body.BodyTracker(ref=0.2)
    out, raw = [], []
    for k in range(180):
        jx, jy = rnd.gauss(0, 0.004), rnd.gauss(0, 0.004)
        out.append(t.update(lm(sh_x=(0.6 + jx, 0.4 + jx), sh_y=0.4 + jy), k * DT, ASPECT))
        raw.append((0.5 + jx, 0.4 + jy))

    def spread(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    assert spread([b.cx for b in out[30:]]) < 0.7 * spread([r[0] for r in raw[30:]])
    assert spread([b.cy for b in out[30:]]) < 0.7 * spread([r[1] for r in raw[30:]])


# --- a stranger --------------------------------------------------------------------------------------------------------------
def test_a_stranger_who_is_much_bigger_or_jumps_in_is_refused_and_the_player_goes_on_being_followed():
    t = body.BodyTracker(ref=0.2)
    run(t, [lm() for _ in range(30)])
    close = [lm(sh_x=sh_at(0.40)) for _ in range(10)]                                   # twice the width
    assert all(b is None for b in run(t, close, t0=30 * DT)) and t.reason == "size"
    beside = [lm(sh_x=sh_at(0.2, centre=0.8)) for _ in range(10)]                       # the same size, 0.3 of the frame away
    assert all(b is None for b in run(t, beside, t0=40 * DT)) and t.reason == "jump"
    again = t.update(lm(), 50 * DT, ASPECT)
    assert again is not None and not again.held and again.cx == pytest.approx(0.5, abs=0.01)
    assert t.counts["rejected"] == 20


def test_a_spectator_before_anybody_was_tracked_is_refused_against_the_calibrated_width():
    t = body.BodyTracker(ref=0.20)
    assert all(b is None for b in run(t, [lm(sh_x=sh_at(0.35)) for _ in range(5)]))        # 1.75 x the calibrated width


def test_a_body_that_stays_where_the_old_one_was_not_is_accepted_after_a_second_so_the_player_is_never_locked_out():
    # the player walked to another spot, or was never what the calibration measured: the lock must give way
    t = body.BodyTracker(ref=0.2)
    run(t, [lm() for _ in range(30)])
    new = [lm(sh_x=sh_at(0.2, centre=0.8)) for _ in range(60)]
    out = run(t, new, t0=30 * DT)
    assert out[0] is None and out[-1] is not None and out[-1].cx == pytest.approx(0.8, abs=0.03)
    first = next(k for k, b in enumerate(out) if b is not None)
    assert 25 <= first <= 40                                                                 # about a second (30 frames)


def test_without_any_reference_the_first_body_is_accepted():
    t = body.BodyTracker()
    out = run(t, [lm(sh_x=sh_at(0.3)) for _ in range(10)])
    assert all(b is not None for b in out) and out[-1].unit == pytest.approx(0.3, abs=0.005)


def test_reset_forgets_the_body():
    t = body.BodyTracker(ref=0.2)
    run(t, [lm() for _ in range(10)])
    t.reset()
    assert t.update(lm(sh_x=sh_at(0.28, centre=0.7)), 10.0, ASPECT) is not None


# --- the hand ----------------------------------------------------------------------------------------------------------------
INDEX, PINKY, THUMB = 20, 18, 22


def fist(wrist, v=(0.9, 0.9, 0.9), off=((-0.04, -0.06), (-0.02, -0.03), (-0.03, -0.04))):
    """A fist: index, pinky and thumb points a little up and in front of the wrist."""
    pts = {}
    for idx, vis, (dx, dy) in zip((INDEX, PINKY, THUMB), v, off):
        pts[idx] = (wrist[0] + dx, wrist[1] + dy, vis)
    return lm(wrist=wrist, fingers=pts)


def test_the_hand_is_the_fist_as_before_when_all_three_points_are_seen():
    h = body.HandTracker("right")
    x, y, conf = h.update(fist((0.30, 0.40)), 0.0)
    assert x == pytest.approx(0.30 + 0.4 * -0.04 + 0.4 * -0.02 + 0.2 * -0.03)
    assert y == pytest.approx(0.40 + 0.4 * -0.06 + 0.4 * -0.03 + 0.2 * -0.04) and conf == pytest.approx(0.95)


def test_a_finger_point_that_flickers_away_does_not_make_the_hand_jump():
    # the visible points' mean moved by up to 0.4 of the fist's width each time one dropped out
    h = body.HandTracker("right")
    before = h.update(fist((0.30, 0.40)), 0.0)
    flicker = h.update(fist((0.30, 0.40), v=(0.1, 0.9, 0.9)), DT)                       # the index point is lost
    assert flicker[0] == pytest.approx(before[0], abs=1e-9) and flicker[1] == pytest.approx(before[1], abs=1e-9)
    moved = h.update(fist((0.28, 0.38), v=(0.1, 0.9, 0.9)), 2 * DT)                    # and the hand moves while it is lost
    assert moved[0] == pytest.approx(before[0] - 0.02, abs=1e-9) and moved[1] == pytest.approx(before[1] - 0.02, abs=1e-9)


def test_fingers_lost_for_good_fall_back_to_the_wrist_and_a_hidden_wrist_is_no_hand():
    h = body.HandTracker("right", hold_s=0.5)
    h.update(fist((0.30, 0.40)), 0.0)
    lost = h.update(lm(wrist=(0.31, 0.41)), 2.0)                                       # no finger point for 2 s
    assert (lost[0], lost[1]) == pytest.approx((0.31, 0.41))
    assert h.update(lm(wrist=(0.31, 0.41), wrist_vis=0.2), 2.1) is None


def test_the_left_hand_uses_the_left_landmarks():
    left = lm()
    left[15] = SimpleNamespace(x=0.70, y=0.40, visibility=0.9)
    left[19] = SimpleNamespace(x=0.74, y=0.34, visibility=0.9)
    x, y, _ = body.HandTracker("left").update(left, 0.0)
    assert x == pytest.approx(0.70 + 0.04) and y == pytest.approx(0.40 - 0.06)


def test_the_hand_in_shoulder_widths_matches_the_old_formula_when_nothing_is_special():
    from pingpong import pose

    frame = lm(wrist=(0.30, 0.40))
    b = body.BodyTracker(ref=0.2).update(frame, 0.0, ASPECT)
    hx, hy, _ = body.HandTracker("right").update(frame, 0.0)
    u, v = body.hand_uv(b, hx, hy, ASPECT)
    ou, ov, _ = pose.paddle_uv(frame, "right", W, H)
    assert u == pytest.approx(ou) and v == pytest.approx(ov)
