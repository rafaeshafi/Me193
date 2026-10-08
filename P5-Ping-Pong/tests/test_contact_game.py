"""A hand that moves into the ball hits it (hit_mode "contact"): no swing is needed, the ball stays on the paddle a moment so the
wrist's flick can be seen, and then leaves with the spin the flick gave it."""

import dataclasses

import pytest

from pingpong import contact, flick, physics, strokepath
from pingpong.events import PaddlePose
from test_game import BOX, S, make, start_rally, swing

M = strokepath.ShotModel()
FRAME = flick.wrist_frame((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))        # the hub's z up, x forward, -y to the right
POSE_DT = 1 / 30
LAG_S = 0.03                                                        # the game hears of a reading this long after it was taken


def contact_game(**kw):
    game, client = make(**kw)
    game.hit_mode = "contact"
    game.wrist_frame = FRAME
    return game


def ball_hand(game, du=0.0, v=0.0):
    """The hand that is level with the ball across the table when it arrives (du off), at height v."""
    leg = game.incoming_leg
    u = contact.ball_u(BOX, leg.position(leg.arrival_ns)[0]) + du
    return lambda t: (u, v)


def drive(game, hand, start_ns, seconds, poses=None, events=None, stop_on=None):
    """Feed one hand reading every 1/30 s (the game hears of it LAG_S later) and tick the clock with it; stop_on: an event
    kind that ends the drive as soon as it happens."""
    poses = [] if poses is None else poses
    events = [] if events is None else events
    t = start_ns
    for _ in range(round(seconds / POSE_DT)):
        if stop_on is not None and any(e.kind == stop_on for e in events):
            break
        t += round(POSE_DT * S)
        u, v = hand(t / S)
        pose = PaddlePose(t_scene_ns=t, u=u, v=v, conf=0.9, hand="right")
        poses.append(pose)
        now = t + round(LAG_S * S)
        events += game.on_pose(pose, list(poses), now)
        events += game.tick(now)
    return events, poses, t


def kinds(events):
    return [e.kind for e in events]


def rally(game):
    start = start_rally(game)
    return start


def test_a_hand_in_the_balls_way_hits_it_without_any_swing():
    game = contact_game()
    start = rally(game)
    events, _, _ = drive(game, ball_hand(game), start, 1.7)
    assert "hit" in kinds(events) and game.tracker.streak == 1
    hit = next(e for e in events if e.kind == "hit")
    assert hit.data["mode"] == "contact" and hit.data["label"] in ("perfect", "good") and game.outgoing_leg is not None


def test_a_hand_across_the_table_lets_it_go_by_and_the_x_ray_says_how_far():
    game = contact_game()
    start = rally(game)
    events, _, _ = drive(game, ball_hand(game, du=1.2), start, 2.5)
    assert "hit" not in kinds(events) and "miss" in kinds(events) and game.phase == "MATCH_OVER"
    rejected = next(e for e in events if e.kind == "verdict")
    verdict = rejected.data["verdict"]
    assert verdict.kind == "REJECTED"
    assert any(g.name == "J2" and not g.passed and "SW from the ball when it passed" in g.note for g in verdict.gates)


def test_the_ball_stays_on_the_paddle_for_the_hold_and_leaves_after_it():
    game = contact_game()
    start = rally(game)
    events, poses, t = drive(game, ball_hand(game), start, 0.0)
    held_seen = False
    while "hit" not in kinds(events) and t < start + 3 * S:
        events, poses, t = drive(game, ball_hand(game), t, POSE_DT, poses, events)
        held_seen = held_seen or (game.held is not None and "hit" not in kinds(events))
    assert held_seen and game.held is None
    hit = next(e for e in events if e.kind == "hit")
    assert hit.data["contact_ns"] >= hit.data["met_ns"] + round(M.hold_s * S) - 1_000_000
    assert game.outgoing_leg.t0_ns == hit.data["contact_ns"]                              # it leaves the paddle when it is let go


def test_a_hand_that_hardly_moves_blocks_the_ball_gently_and_a_fast_one_hits_it_hard():
    still_game = contact_game()
    start = rally(still_game)
    events, _, _ = drive(still_game, ball_hand(still_game), start, 1.7)
    block = next(e for e in events if e.kind == "hit")
    assert block.data["v_out"] == pytest.approx(shot_out(0.0), abs=0.01)

    fast_game = contact_game()
    start = rally(fast_game)
    centre = ball_hand(fast_game)(0.0)[0]
    arrive = fast_game.incoming_leg.arrival_ns / S
    sweep = lambda t: (centre + 4.0 * (t - arrive), 0.0)                                  # noqa: E731  4 shoulder widths a second across
    events, _, _ = drive(fast_game, sweep, start, 1.7)
    hard = next(e for e in events if e.kind == "hit")
    assert hard.data["v_out"] > block.data["v_out"] + 6.0
    assert hard.data["stroke"]["vu"] == pytest.approx(4.0, rel=0.25)


def shot_out(s):
    from pingpong import shot

    return shot.out_speed(s)


def test_a_swing_does_not_hit_in_contact_mode():
    game = contact_game()
    start = rally(game)
    t = game.incoming.t_c_ns - round(0.16 * S)
    assert game.on_swing(swing(t), [], t) == [] and game.incoming is not None


def test_the_path_of_the_hand_places_the_ball_and_a_lifting_hand_lofts_it():
    results = {}
    for name, (vu, vv) in {"still": (0.0, 0.0), "right": (2.5, 0.0), "left": (-2.5, 0.0), "lift": (0.0, 2.5)}.items():
        game = contact_game()
        start = rally(game)
        centre = ball_hand(game)(0.0)[0]
        arrive = game.incoming_leg.arrival_ns / S
        hand = lambda t, vu=vu, vv=vv, c=centre, a=arrive: (c + vu * (t - a), vv * (t - a))      # noqa: E731
        events, _, _ = drive(game, hand, start, 2.0, stop_on="hit")
        results[name] = (game.outgoing_leg, next(e for e in events if e.kind == "hit"))
    assert results["left"][0].x_end < results["still"][0].x_end < results["right"][0].x_end
    assert results["lift"][0].apex1_m > results["still"][0].apex1_m + 0.08
    assert all(r[1].data["topspin"] == 0.0 and r[1].data["sidespin"] == 0.0 for r in results.values())    # no flick, no spin


def flick_samples(rate, *, around_ns, hz=64.0, seconds=0.6):
    """Gyro samples (arrival ns, dps) with a constant rate across a stretch of time around `around_ns`."""
    t0 = around_ns - round(seconds / 2 * S)
    return [(t0 + round(i * S / hz), rate) for i in range(int(seconds * hz))]


def test_a_flick_of_the_wrist_beyond_the_usual_gives_the_ball_its_spin():
    game = contact_game()
    game.shot_model = dataclasses.replace(M, flick_warmup=1)
    spins = []
    rates = iter([(40.0, 0.0, 0.0), (40.0, 0.0, 0.0), (40.0, 350.0, 0.0), (40.0, 0.0, -350.0)])
    current = {"rate": None}
    game.gyro_window = lambda lo, hi: flick_samples(current["rate"], around_ns=(lo + hi) // 2)
    start = rally(game)
    for _ in range(4):
        current["rate"] = next(rates)
        events, _, t = drive(game, ball_hand(game), start, 1.7)
        hit = next(e for e in events if e.kind == "hit")
        spins.append((hit.data["topspin"], hit.data["sidespin"]))
        # let the computer return the ball and serve the next one
        while game.incoming is None and t < start + 30 * S:
            game.tick(t + round(0.1 * S))
            t += round(0.1 * S)
        start = t
    assert spins[0] == (0.0, 0.0) and spins[1] == (0.0, 0.0)                              # what the player usually does: no spin
    assert spins[2][0] == pytest.approx(1.0, abs=0.01) and spins[2][1] == 0.0              # the front tipped down: topspin
    assert spins[3][0] == 0.0 and spins[3][1] == pytest.approx(1.0, abs=0.01)              # turned right: sidespin right


def test_without_a_wrist_frame_or_a_gyro_the_ball_is_flat():
    game = contact_game()
    game.wrist_frame = None
    game.gyro_window = lambda lo, hi: flick_samples((0.0, 900.0, 900.0), around_ns=(lo + hi) // 2)
    start = rally(game)
    events, _, _ = drive(game, ball_hand(game), start, 1.7)
    hit = next(e for e in events if e.kind == "hit")
    assert hit.data["topspin"] == 0.0 and hit.data["sidespin"] == 0.0


def test_a_pause_during_the_hold_moves_the_launch_by_the_time_paused():
    game = contact_game()
    start = rally(game)
    events, poses, t = drive(game, ball_hand(game), start, 0.0)
    while game.held is None and t < start + 3 * S:
        events, poses, t = drive(game, ball_hand(game), t, POSE_DT, poses, events)
    assert game.held is not None
    launch_before, held_before = game.outgoing_leg.t0_ns, game.held[0]
    now = t + round(LAG_S * S)
    game.set_pause("pose", True, now)
    game.set_pause("pose", False, now + 2 * S)
    assert game.outgoing_leg.t0_ns == launch_before + 2 * S and game.held[0] == held_before + 2 * S


def test_a_miss_waits_for_the_hand_to_have_been_seen_past_the_ball():
    game = contact_game()
    start = rally(game)
    late = game.incoming_leg.arrival_ns + 3 * S
    assert game.tick(late) == [] and game.phase == "RALLY"                                # no readings at all: not a miss yet
    events, _, _ = drive(game, ball_hand(game, du=1.2), start, 2.5)
    assert "miss" in kinds(events)


def test_a_session_started_in_swing_mode_still_hits_with_a_swing_only():
    game, _ = make()
    start_rally(game)
    assert game.hit_mode == "swing"


def test_the_hold_always_covers_the_flicks_window_even_when_that_is_stretched():
    game = contact_game()
    game.shot_model = dataclasses.replace(M, flick_after_s=0.25)
    start = rally(game)
    events, poses, t = drive(game, ball_hand(game), start, 0.0)
    while game.held is None and t < start + 3 * S:
        events, poses, t = drive(game, ball_hand(game), t, POSE_DT, poses, events)
    assert game.held is not None
    assert game.outgoing_leg.t0_ns - game.held[0] >= round((0.25 + game.imu_delay_s) * S)


def test_a_hit_reports_what_the_wrist_did_so_the_spin_can_be_tuned_from_a_recording():
    game = contact_game()
    game.shot_model = dataclasses.replace(M, flick_warmup=1)
    game.gyro_window = lambda lo, hi: flick_samples((40.0, 350.0, 0.0), around_ns=(lo + hi) // 2)
    start = rally(game)
    events, _, _ = drive(game, ball_hand(game), start, 1.7, stop_on="hit")
    seen = next(e for e in events if e.kind == "hit").data["flick"]
    assert seen["rate"] == pytest.approx((40.0, 350.0, 0.0)) and seen["dev"] == (0.0, 0.0, 0.0)       # the first hit learns the usual
    blind = contact_game()
    start = rally(blind)
    events, _, _ = drive(blind, ball_hand(blind), start, 1.7, stop_on="hit")
    assert next(e for e in events if e.kind == "hit").data["flick"] == {"rate": None, "dev": None}

