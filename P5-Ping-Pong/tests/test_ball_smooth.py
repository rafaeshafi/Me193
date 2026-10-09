"""The ball as the player SEES it, frame after frame (60 a second): it never jumps.

The picture is drawn ahead of the game by the display's delay, and the game learns that the hand met the ball a camera frame and a
pose later; before that fix the ball had been drawn flying past the paddle by then and jumped back to it for the hit (120-340 px in
one frame on the player's real games).  A ball whose hit is still being decided now waits at the paddle."""

import math

import numpy as np
import pytest

from pingpong import app, contact, hud
from pingpong.clock import FakeClock
from pingpong.events import PaddlePose

S = 1_000_000_000
W, H = 1280, 720
CAMERA = hud._camera((W, H))
RADIUS = {1: 0.85, 2: 0.70, 3: 0.55}                        # shoulder widths: how far from the ball a hand still hits it


def one_ball(level, du, lag_s=0.03, seed=3, balls=1, mode="survival", stroke=False):
    """A contact-mode game: a hand reading every 1/30 s that the game hears of lag_s later, level with each incoming ball across
    the table (du shoulder widths off), and the picture's ball at 60 frames a second, until `balls` balls have been played (the
    computer's return counts as the next ball).  stroke: the hand sweeps up and across into each ball like a real stroke (so the
    return is fast, aimed and lofted).  -> (session, [(t_s, screen px or None, (x, y, z) or None)])."""
    clock = FakeClock(start_ns=1_000_000_000)
    session = app.make_session(level=level, hit_mode="contact", clock=clock, seed=seed, mode=mode)
    session.on_start()
    box = session.game.judge.box
    while session.game.phase != "RALLY":
        clock.advance_s(1 / 240)
        session.tick()
    frames, due, next_pose, next_frame = [], [], clock.now_ns(), clock.now_ns()
    u_hand, v_hand, seen, end = 0.0, 0.0, 0, None
    while end is None or clock.now_ns() < end:
        clock.advance_s(1 / 240)
        now = clock.now_ns()
        leg = session.game.incoming_leg
        if leg is not None and session.game.incoming is not None:
            u_ball = contact.ball_u(box, leg.position(leg.arrival_ns)[0]) + du
            u_hand, v_hand = u_ball, 0.0
            if stroke:                                              # sweeping in from the side and from below over the last 0.45 s
                phase = min(1.0, max(0.0, 1.0 - (leg.arrival_ns - now) / (0.45 * S)))
                u_hand, v_hand = u_ball - 0.9 * (1.0 - phase), -0.3 + 0.3 * phase
            if session.game.incoming.ball_id != seen:
                seen = session.game.incoming.ball_id
                if seen == balls:
                    end = leg.arrival_ns + int(1.2 * S)
        if now >= next_pose:
            due.append((now + round(lag_s * S), PaddlePose(t_scene_ns=now - round(lag_s * S), u=u_hand, v=v_hand, conf=0.9, hand="right")))
            next_pose += S // 30
        while due and due[0][0] <= now:
            session.on_pose(due.pop(0)[1])
        session.tick()
        if now >= next_frame:
            ball = session.hud_state().ball
            frames.append((now / S, None if ball is None else CAMERA.project(*ball)[:2], ball))
            next_frame += S // 60
        if session.game.phase in ("MATCH_OVER", "POINT_OVER") and end is None:
            end = now + int(1.5 * S)
    return session, frames


def steps(frames):
    """How far the ball moved on the screen between each two pictures in which it is on the screen."""
    on = lambda p: p is not None and 0 <= p[0] < W and 0 <= p[1] < H                        # noqa: E731
    return [math.hypot(b[1][0] - a[1][0], b[1][1] - a[1][1]) for a, b in zip(frames, frames[1:]) if on(a[1]) and on(b[1])]


def longest_rest(frames, below_m=0.004):
    """The most pictures in a row in which the ball did not move (4 mm between two pictures: a real ball does 15 mm at the very
    least, its apex only looks still on the screen)."""
    best = run = 0
    for a, b in zip(frames, frames[1:]):
        if a[2] is None or b[2] is None:
            run = 0
            continue
        run = run + 1 if math.dist(a[2], b[2]) < below_m else 0
        best = max(best, run)
    return best


@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_ball_that_is_hit_never_jumps_on_the_screen(level):
    session, frames = one_ball(level, du=0.0)
    assert session.game.tracker.streak == 1
    assert max(steps(frames)) < 70.0                           # a fast ball moves up to ~60 px between two pictures: a jump is 120-340


@pytest.mark.parametrize("level", [1, 2])
def test_a_ball_that_is_hit_sits_on_the_paddle_for_a_moment_before_it_goes(level):
    _, frames = one_ball(level, du=0.0)
    assert longest_rest(frames) >= 4


@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_ball_the_hand_clearly_misses_flies_on_past_without_stopping_or_jumping(level):
    session, frames = one_ball(level, du=3.0)
    assert session.game.tracker.streak == 0
    assert max(steps(frames)) < 70.0 and longest_rest(frames) < 3


@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_near_miss_waits_a_moment_and_then_catches_up_without_a_jump(level):
    session, frames = one_ball(level, du=1.3 * RADIUS[level])
    assert session.game.tracker.streak == 0
    _, clear = one_ball(level, du=3.0)                         # the same ball, no wait: its fastest step is what flight looks like
    assert max(steps(frames)) <= 1.6 * max(steps(clear))        # catching up is a ball a little quicker than usual, not a jump


@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_whole_rally_never_jumps_the_ball_at_a_hit_a_launch_or_the_computers_return(level):
    session, frames = one_ball(level, du=0.0, balls=4, stroke=True)
    assert session.game.tracker.streak >= 3
    assert max(steps(frames)) < 70.0


@pytest.mark.parametrize("level", [1, 2])
def test_the_ball_waits_at_the_computers_paddle_for_its_return_instead_of_flying_past_it_and_jumping_back(level):
    session, frames = one_ball(level, du=0.0, balls=3, stroke=True)
    far = [p for p in frames if p[2] is not None and p[2][2] > 2.5]
    assert far and max(p[2][2] for p in far) < 3.2                   # never drawn more than a few cm past the computer's strike (2.89 m)


def test_a_match_rally_with_a_point_won_has_no_jump_either():
    session, frames = one_ball(2, du=0.0, balls=3, mode="match", stroke=True)
    assert max(steps(frames)) < 70.0
