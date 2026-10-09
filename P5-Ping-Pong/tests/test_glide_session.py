"""The paddle and the pointer are drawn smoothly between the camera's readings (glide.py), and nothing the game decides depends on it."""

import dataclasses

import numpy as np
import pytest

from pingpong import app, latency
from pingpong.clock import FakeClock
from pingpong.events import PaddlePose
from pingpong.flow import Flow

S = 1_000_000_000
FRAME = S // 60


def session_with(glide_s, **kw):
    lat = dataclasses.replace(latency.Latency.from_config(), glide_s=glide_s)
    return app.make_session(clock=FakeClock(start_ns=1_000_000_000), latency=lat, **kw)


def drive(session, seconds=3.0, speed_sw_s=2.0, on_frame=None):
    """A hand moving across at speed_sw_s, read 30 times a second, with a picture drawn 60 times a second."""
    clock, shown = session.clock, []
    t0, next_reading = clock.now_ns(), clock.now_ns()
    for k in range(round(seconds * 60)):
        clock.advance_s(1 / 60)
        now = clock.now_ns()
        if now >= next_reading:
            u = speed_sw_s * (now - t0) / S - 1.0
            session.on_pose(PaddlePose(t_scene_ns=now, u=u, v=0.0, conf=0.9, hand="right"))
            next_reading += S // 30
        session.tick()
        shown.append(on_frame(session) if on_frame else session.hud_state().paddle)
    return shown


def wobble(points):
    v = np.diff([p[0] for p in points]) * 60.0
    return float(np.sqrt(np.mean((v - np.convolve(v, np.ones(9) / 9, mode="same"))[5:-5] ** 2)))


def test_the_glide_is_one_of_the_places_the_time_goes_and_can_be_switched_off_like_the_others():
    assert latency.Latency().glide_s == pytest.approx(0.025)
    with pytest.raises(ValueError):
        latency.Latency(glide_s=-0.01)


def test_the_paddle_moves_more_smoothly_with_the_glide_than_held_where_it_was_last_read():
    held = drive(session_with(0.0))
    smooth = drive(session_with(0.025))
    assert wobble(smooth) < 0.55 * wobble(held)
    assert smooth[-1][0] == pytest.approx(held[-1][0], abs=0.08)               # same place: a few centimetres behind a hand that moves 2 SW/s


def test_a_paddle_that_has_stopped_is_where_the_hand_is_whatever_the_glide():
    smooth = session_with(0.025)
    drive(smooth, seconds=1.0)
    for _ in range(90):                                                     # the hand stays put for a second and a half
        smooth.clock.advance_s(1 / 60)
        smooth.on_pose(PaddlePose(t_scene_ns=smooth.clock.now_ns(), u=1.0, v=0.0, conf=0.9, hand="right"))
        smooth.tick()
    held = session_with(0.0)
    drive(held, seconds=1.0)
    for _ in range(90):
        held.clock.advance_s(1 / 60)
        held.on_pose(PaddlePose(t_scene_ns=held.clock.now_ns(), u=1.0, v=0.0, conf=0.9, hand="right"))
        held.tick()
    assert smooth.hud_state().paddle == pytest.approx(held.hud_state().paddle, abs=1e-3)


def test_the_glide_changes_nothing_the_game_decides_only_what_is_drawn():
    a, b = session_with(0.0, hit_mode="contact"), session_with(0.025, hit_mode="contact")
    for s in (a, b):
        s.on_start()
        drive(s, seconds=4.0, speed_sw_s=0.0)
    assert (a.game.phase, a.game.tracker.streak, a.game.incoming_leg.t0_ns) == (b.game.phase, b.game.tracker.streak, b.game.incoming_leg.t0_ns)


def test_the_pointer_in_the_menus_is_drawn_smoothly_but_pressed_where_the_hand_really_is():
    held = session_with(0.0, flow=Flow(intro=False))
    smooth = session_with(0.025, flow=Flow(intro=False))
    cursor = lambda s: (s.hud_state().ui.cursor[0], 0.0)                    # noqa: E731
    h = drive(held, on_frame=cursor)
    sm = drive(smooth, on_frame=cursor)
    assert wobble(sm) < 0.55 * wobble(h)
    assert smooth.flow.pointer.hovered == held.flow.pointer.hovered          # what the hand is on does not depend on how it is drawn
