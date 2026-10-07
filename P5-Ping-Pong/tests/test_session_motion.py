"""The session puts everything where it will be PERCEIVED: the ball ahead of now by the display's delay, the hand ahead
by its whole pipeline's, the paddle lunging to the ball, and the thump and the sounds sent early to arrive with the
picture (see pingpong/latency.py)."""

import dataclasses

import pytest

from pingpong import app, latency, levels, physics, stage
from pingpong.events import PaddlePose

S = 1_000_000_000


class Recorder:
    def __init__(self):
        self.calls = []

    def submit(self, name, fire_at_ns=None):
        self.calls.append((name, fire_at_ns))
        return True


class FakeAudio:
    def __init__(self):
        self.played = []

    def play(self, name):
        self.played.append(name)

    def sound_for(self, event, level):
        from pingpong.audio import Audio

        return Audio.sound_for(self, event, level)


def serve(session):
    session.on_start()
    session.clock.advance_s(3.0)
    session.tick()
    return session.game.incoming_leg


def test_the_ball_is_drawn_where_it_will_be_when_the_light_reaches_the_eye():
    session = app.make_session()
    leg = serve(session)
    session.clock.advance_s(0.4)
    ahead = session.latency.view_ahead_s
    assert session.hud_state().ball == pytest.approx(leg.position(session.clock.now_ns() + round(ahead * S)))
    assert ahead > 0.04
    slow = app.make_session(latency=dataclasses.replace(latency.Latency.from_config(), display_s=0.0, loop_s=0.0))
    serve(slow)
    slow.clock.advance_s(0.4)
    assert slow.hud_state().ball == pytest.approx(slow.game.incoming_leg.position(slow.clock.now_ns()))


def test_a_frozen_game_shows_a_frozen_ball():
    session = app.make_session()
    serve(session)
    session.clock.advance_s(0.4)
    session.game.set_pause("hub", True, session.clock.now_ns())
    before = session.hud_state().ball
    session.clock.advance_s(0.5)
    assert session.hud_state().ball == pytest.approx(before)


def hand(session, u, v, t_s, conf=0.9):
    session.on_pose(PaddlePose(t_scene_ns=round(t_s * S), u=u, v=v, conf=conf, hand="right"))


def test_the_paddle_is_where_the_hand_puts_it_across_the_table_and_up_the_table():
    session = app.make_session()
    box = session.game.judge.box
    assert session.hud_state().paddle is None                                      # no hand seen yet: no paddle
    for k in range(8):
        hand(session, 0.5, 0.2, 1.0 + k / 30)
    st = session.hud_state()
    expect = stage.rest_position(box, 0.5, 0.2)
    assert st.paddle == pytest.approx(expect) and st.rest == pytest.approx(expect)


def test_a_moving_hand_is_drawn_ahead_of_its_last_reading():
    session = app.make_session()
    for k in range(8):
        hand(session, -0.6 + 1.5 * k / 30, 0.0, 1.0 + k / 30)                      # 1.5 shoulder widths a second to the right
    session.clock.advance_s(0.0)
    last_rest = stage.rest_position(session.game.judge.box, -0.6 + 1.5 * 7 / 30, 0.0)
    assert session.hud_state().rest[0] > last_rest[0]


def test_the_tilt_of_the_hub_turns_the_paddle_on_screen():
    session = app.make_session()
    hand(session, 0.0, 0.0, 1.0)
    session.paddle_angle = 30.0
    assert session.hud_state().paddle_angle == 30.0


# --- a swing: the paddle lunges ------------------------------------------------------------------------------------------
def a_swing(session, t_ns, w_pk=600.0):
    from pingpong.events import SwingEvent

    return session.on_swing(SwingEvent(kind="IMPACT", t_ns=t_ns, w_pk=w_pk, dur_ms=150.0, n_reversals=0,
                                       axis_unit=(1, 0, 0), net_rot_unit=(1, 0, 0), a_lin_unit=(0, 0, 1), clipped=False,
                                       feat=(0.0,) * 12))


def test_a_swing_at_nothing_makes_the_paddle_reach_forward_and_come_back():
    session = app.make_session()
    for k in range(6):
        hand(session, 0.0, 0.0, session.clock.now_ns() / S - 0.2 + k / 30)
    rest = session.hud_state().rest
    a_swing(session, session.clock.now_ns() - 60_000_000)
    lunged = False
    for _ in range(40):
        session.clock.advance_s(0.01)
        if session.hud_state().paddle[2] > rest[2] + 0.2:
            lunged = True
    assert lunged
    session.clock.advance_s(0.6)
    assert session.hud_state().paddle == pytest.approx(session.hud_state().rest, abs=1e-6)


def play_into_the_zone(session, hit_early_s=0.0):
    """Serve, glide the hand to the ball and swing for it; -> the events of the swing."""
    leg = serve(session)
    box = session.game.judge.box
    ball = session.game.incoming
    for k in range(10):                                                            # the hand waits where the ball will be
        t = session.clock.now_ns() / S + k / 30
        u = box.to_uv(physics.a_of_x(physics.x_of_a(ball.aim_ab[0])), 0.5)[0]
        session.on_pose(PaddlePose(t_scene_ns=round(t * S), u=u, v=0.0, conf=0.9, hand="right"))
    t_sweet = session.game.judge.sweet_ns(ball, 0.0)
    t_peak = t_sweet - round(session.game.judge.contact_lag_s * S) - round(hit_early_s * S)
    while session.clock.now_ns() < t_peak + 20_000_000:
        session.clock.advance_s(0.005)
        t = session.clock.now_ns() / S
        session.on_pose(PaddlePose(t_scene_ns=round(t * S), u=u, v=0.0, conf=0.9, hand="right"))
        session.tick()
    return a_swing(session, t_peak), leg


def test_a_hit_makes_the_paddle_meet_the_ball_where_it_is_and_the_ball_leaves_from_there():
    session = app.make_session()
    events, _ = play_into_the_zone(session)
    hit = next(e for e in events if e.kind == "hit")
    out = session.game.outgoing_leg
    assert out.t0_ns == hit.data["contact_ns"] and out.position(out.t0_ns) == pytest.approx(hit.data["contact"])
    # at the contact the paddle's face is on the ball
    session.clock.advance_s(max(0.0, (hit.data["contact_ns"] - session.clock.now_ns()) / S - session.latency.view_ahead_s))
    paddle = session.hud_state().paddle
    assert paddle == pytest.approx(stage.hit_stroke(0, 1, hit.data["contact"]).target, abs=0.05) or paddle[2] > 0.25


def test_the_incoming_ball_keeps_flying_until_the_contact_then_the_return_takes_over():
    session = app.make_session()
    events, _ = play_into_the_zone(session)
    hit = next(e for e in events if e.kind == "hit")
    out = session.game.outgoing_leg
    now = session.clock.now_ns()
    view = now + round(session.latency.view_ahead_s * S)
    if view < out.t0_ns:                                                           # (the hit was seen before the contact)
        assert session.hud_state().ball == pytest.approx(session.game.incoming_leg.position(view))
    session.clock.advance_s(0.1)
    now = session.clock.now_ns()
    assert out.t0_ns < now + round(session.latency.view_ahead_s * S) < out.end_ns
    assert session.hud_state().ball == pytest.approx(out.position(now + round(session.latency.view_ahead_s * S)))


# --- feedback is sent early to arrive with the picture ------------------------------------------------------------------------
def test_the_thump_of_a_hit_is_sent_early_by_the_motors_delay_to_arrive_with_the_contact():
    rec = Recorder()
    session = app.make_session(actuator=rec)
    events, _ = play_into_the_zone(session)
    hit = next(e for e in events if e.kind == "hit")
    hit_calls = [(n, f) for n, f in rec.calls if n.startswith("hit_")]
    assert hit_calls
    now = session.clock.now_ns()
    name, fire_at = hit_calls[-1]
    assert fire_at is not None
    assert fire_at == max(now, hit.data["contact_ns"] - round(session.latency.haptic_s * S))


def test_a_hit_sound_is_queued_for_the_moment_the_picture_shows_the_contact_less_the_audio_delay():
    session = app.make_session()
    session.audio = FakeAudio()
    events, _ = play_into_the_zone(session)
    hit = next(e for e in events if e.kind == "hit")
    due = hit.data["contact_ns"] - round(session.latency.audio_s * S)
    if session.clock.now_ns() < due:
        assert session.audio.played == [s for s in session.audio.played if s not in ("hit_good", "hit_perfect", "hit_off")]
        session.clock.advance_s((due - session.clock.now_ns()) / S + 0.002)
        session.tick()
    assert any(s.startswith("hit_") for s in session.audio.played)


def test_the_bounce_is_heard_when_it_is_seen():
    session = app.make_session()
    session.audio = FakeAudio()
    leg = serve(session)
    t_bounce = leg.bounce_ns
    session.clock.advance_s((t_bounce - session.clock.now_ns()) / S - session.latency.audio_s - 0.05)
    session.tick()
    assert "bounce" not in session.audio.played
    session.clock.advance_s(0.08)
    session.tick()
    assert session.audio.played.count("bounce") == 1
    session.clock.advance_s(0.05)
    session.tick()
    assert session.audio.played.count("bounce") == 1                               # once per bounce


def test_a_ball_the_computer_misses_flies_on_past_its_end_instead_of_vanishing():
    session = app.make_session(mode="match", target=5, latency=dataclasses.replace(latency.Latency.from_config(), display_s=0.0, loop_s=0.0))
    session.game.policy.returns = lambda *a, **k: False
    events, _ = play_into_the_zone(session)
    out = session.game.outgoing_leg
    session.clock.advance_s((out.arrival_ns - session.clock.now_ns()) / S + 0.2)
    session.tick()
    assert session.game.phase in ("POINT_OVER", "MATCH_OVER")
    x, y, z = session.hud_state().ball
    assert z > physics.CPU_Z_M and y < physics.STRIKE_Y_M                  # past the computer's paddle, on its way down
    session.clock.advance_s(2.0)
    assert session.hud_state().ball is None
