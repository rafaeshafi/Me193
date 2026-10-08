"""The path of the hand and the twist of the hub decide where the ball goes after the hit and how it spins."""

import dataclasses

import pytest

from pingpong import fakerig, overrides, physics, strokepath
from pingpong.events import PaddlePose
from test_game import BOX, LAG_NS, S, make, start_rally, swing

M = strokepath.ShotModel()


def moving_poses(t_i, ab, vu, vv):
    """The hand through its place at the peak (t_i) at (vu, vv) shoulder widths a second, readings every 50 ms."""
    u0, v0 = BOX.to_uv(*ab)
    return [PaddlePose(t_scene_ns=int(t_i + k * 0.05 * S), u=u0 + vu * k * 0.05, v=v0 + vv * k * 0.05, conf=0.9, hand="right")
            for k in range(-6, 3)]


def hit(game, vu=0.0, vv=0.0, net=(0.0, 0.0, 0.0), w_pk=600.0):
    ball = game.incoming
    t = ball.t_c_ns - LAG_NS
    event = dataclasses.replace(swing(t, w_pk), net_rot_deg=net)
    events = game.on_swing(event, moving_poses(t, ball.aim_ab, vu, vv), t)
    return next(e for e in events if e.kind == "hit")


def first_hit(**kw):
    game, _ = make()
    start_rally(game)
    return game, hit(game, **kw)


def next_ball(game):
    """Let the computer return the ball and serve the next one."""
    game.tick(game.next_cpu_contact_ns + 1)
    assert game.incoming is not None


def test_a_hand_that_lifts_gives_the_return_topspin_and_a_higher_arc_than_a_hand_that_stays():
    still_game, still = first_hit()
    lift_game, lifted = first_hit(vv=2.5)
    assert still.data["topspin"] == 0.0 and lifted.data["topspin"] == pytest.approx(M.k_top, abs=0.03)
    assert lift_game.outgoing_leg.apex1_m > still_game.outgoing_leg.apex1_m + 0.8 * M.k_loft_m
    assert lift_game.outgoing_leg.flight_s < still_game.outgoing_leg.flight_s            # topspin dives: it is there sooner


def test_a_hand_that_chops_gives_backspin_and_no_extra_arc():
    still_game, _ = first_hit()
    chop_game, chopped = first_hit(vv=-2.5)
    assert chopped.data["topspin"] == pytest.approx(-M.k_top, abs=0.03)
    assert chop_game.outgoing_leg.apex1_m == pytest.approx(still_game.outgoing_leg.apex1_m)
    assert chop_game.outgoing_leg.flight_s > still_game.outgoing_leg.flight_s


def test_a_hand_that_swings_to_the_right_places_the_ball_further_right_and_curves_it_that_way():
    centre_game, centre = first_hit()
    right_game, right = first_hit(vu=2.5)
    left_game, left = first_hit(vu=-2.5)
    assert left_game.outgoing_leg.x_end < centre_game.outgoing_leg.x_end < right_game.outgoing_leg.x_end
    shift = right_game.outgoing_leg.x_end - centre_game.outgoing_leg.x_end
    assert shift == pytest.approx(M.k_aim * 1.6 * physics.HALF_WIDTH_M, abs=0.05)         # k_aim of the table's width
    assert left.data["sidespin"] < 0.0 < right.data["sidespin"] and centre.data["sidespin"] == 0.0


def test_a_twist_of_the_hub_beyond_the_usual_curves_the_ball_once_the_usual_is_known():
    game, _ = make()
    game.wrist_axis = (0.0, 1.0, 0.0)
    game.shot_model = dataclasses.replace(M, twist_warmup=2)
    start_rally(game)
    usual = hit(game, net=(75.0, -22.0, 0.0))
    next_ball(game)
    again = hit(game, net=(75.0, -22.0, 0.0))                                              # the second hit still learns what is usual
    next_ball(game)
    more_right = hit(game, net=(75.0, -22.0 + M.twist_sd_deg, 0.0))
    next_ball(game)
    more_left = hit(game, net=(75.0, -22.0 - M.twist_sd_deg, 0.0))
    assert usual.data["sidespin"] == 0.0 and again.data["sidespin"] == 0.0
    assert more_right.data["sidespin"] == pytest.approx(M.k_side_twist, abs=0.03)
    assert more_left.data["sidespin"] == pytest.approx(-M.k_side_twist, abs=0.03)


def test_without_a_wrist_axis_the_twist_is_ignored():
    game, _ = make()
    start_rally(game)
    assert game.wrist_axis is None
    assert hit(game, net=(75.0, 500.0, 0.0)).data["sidespin"] == 0.0


def test_the_hit_reports_the_path_it_was_made_from():
    _, h = first_hit(vu=1.0, vv=2.0)
    stroke = h.data["stroke"]
    assert set(stroke) == {"vu", "vv", "twist_z", "aim_shift", "loft_m", "topspin", "sidespin"}
    assert stroke["vu"] == pytest.approx(1.0, abs=0.1) and stroke["vv"] == pytest.approx(2.0, abs=0.1)
    assert stroke["topspin"] == h.data["topspin"]


def test_a_spinning_ball_is_harder_for_the_computer_to_return_in_match_play():
    game, _ = make(mode="match")
    start_rally(game)
    hit(game, vv=2.5)
    assert game._out_shot.A == pytest.approx(M.k_top * 1.0, abs=0.03) and game._out_shot.A > 0.0


def test_the_shot_model_is_tunable_with_set_shot_dot_field():
    assert overrides.check(overrides.parse(["shot.k_top=0.9", "shot.k_aim=0"])) == {"shot": {"k_top": 0.9, "k_aim": 0}}
    with pytest.raises(ValueError, match="shot setting"):
        overrides.check({"shot": {"k_topspin": 1.0}})
    rig = fakerig.FakeRig()
    overrides.apply(rig.rig, {"shot": {"k_top": 0.0, "k_loft_m": 0.5}})
    assert rig.game.shot_model.k_top == 0.0 and rig.game.shot_model.k_loft_m == 0.5
    assert rig.game.shot_model.k_aim == M.k_aim                                              # the rest is untouched
    rig.close()
