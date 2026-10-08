"""A swing that hits the ball (hit_mode "swing"): the path of the hand places and lofts the return, and the flick of the wrist, when
the game can see it, spins it.  (The hand-into-the-ball mode is tests/test_contact_game.py.)"""

import dataclasses

import pytest

from pingpong import fakerig, flick, overrides, physics, strokepath
from pingpong.events import PaddlePose
from test_game import BOX, LAG_NS, S, Script, make, start_rally, swing

M = strokepath.ShotModel()
FRAME = flick.wrist_frame((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def moving_poses(t_i, ab, vu, vv):
    """The hand through its place at the peak (t_i) at (vu, vv) shoulder widths a second, readings every 50 ms."""
    u0, v0 = BOX.to_uv(*ab)
    return [PaddlePose(t_scene_ns=int(t_i + k * 0.05 * S), u=u0 + vu * k * 0.05, v=v0 + vv * k * 0.05, conf=0.9, hand="right")
            for k in range(-6, 3)]


def hit(game, vu=0.0, vv=0.0, w_pk=600.0):
    ball = game.incoming
    t = ball.t_c_ns - LAG_NS
    events = game.on_swing(swing(t, w_pk), moving_poses(t, ball.aim_ab, vu, vv), t)
    return next(e for e in events if e.kind == "hit")


def first_hit(**kw):
    game, _ = make()
    start_rally(game)
    return game, hit(game, **kw)


def test_a_hand_that_lifts_gives_the_return_a_higher_arc_but_the_path_alone_makes_no_spin():
    still_game, still = first_hit()
    lift_game, lifted = first_hit(vv=2.5)
    assert lift_game.outgoing_leg.apex1_m > still_game.outgoing_leg.apex1_m + 0.8 * M.k_loft_m
    assert still.data["topspin"] == lifted.data["topspin"] == 0.0 and lifted.data["sidespin"] == 0.0


def test_a_hand_that_chops_does_not_flatten_the_arc_below_what_clears_the_net():
    still_game, _ = first_hit()
    chop_game, chopped = first_hit(vv=-2.5)
    assert chop_game.outgoing_leg.apex1_m == pytest.approx(still_game.outgoing_leg.apex1_m)


def test_a_hand_that_swings_to_the_right_places_the_ball_further_right():
    centre_game, _ = first_hit()
    right_game, _ = first_hit(vu=2.5)
    left_game, _ = first_hit(vu=-2.5)
    assert left_game.outgoing_leg.x_end < centre_game.outgoing_leg.x_end < right_game.outgoing_leg.x_end
    shift = right_game.outgoing_leg.x_end - centre_game.outgoing_leg.x_end
    assert shift == pytest.approx(M.k_aim * 1.6 * physics.HALF_WIDTH_M, abs=0.05)         # k_aim of the table's width


def test_the_hit_reports_the_path_it_was_made_from():
    _, h = first_hit(vu=1.0, vv=2.0)
    stroke = h.data["stroke"]
    assert set(stroke) == {"vu", "vv", "aim_shift", "loft_m", "topspin", "sidespin"} and h.data["mode"] == "swing"
    assert stroke["vu"] == pytest.approx(1.0, abs=0.1) and stroke["vv"] == pytest.approx(2.0, abs=0.1)
    assert stroke["topspin"] == h.data["topspin"]


def test_a_flick_the_game_can_see_spins_a_swung_ball_too_and_makes_it_harder_to_return_in_match_play():
    game, _ = make(mode="match", policy=Script([True, True]))     # the computer returns both balls
    game.wrist_frame = FRAME
    game.shot_model = dataclasses.replace(M, flick_warmup=1)
    game.gyro_window = lambda lo, hi: [((lo + hi) // 2 + k * 10_000_000, (40.0, 0.0, 0.0)) for k in range(-6, 7)]
    start_rally(game)
    hit(game)                                                    # the first hit learns what is usual: no spin
    game.tick(game.next_cpu_contact_ns + 1)
    game.gyro_window = lambda lo, hi: [((lo + hi) // 2 + k * 10_000_000, (40.0, 400.0, 0.0)) for k in range(-6, 7)]
    spun = hit(game)
    assert spun.data["topspin"] == pytest.approx(1.0, abs=0.01) and game._out_shot.A == pytest.approx(1.0, abs=0.01)


def test_the_shot_model_is_tunable_with_set_shot_dot_field():
    assert overrides.check(overrides.parse(["shot.k_flick_top=-1", "shot.k_aim=0"])) == {"shot": {"k_flick_top": -1, "k_aim": 0}}
    with pytest.raises(ValueError, match="shot setting"):
        overrides.check({"shot": {"k_topspin": 1.0}})
    rig = fakerig.FakeRig()
    overrides.apply(rig.rig, {"shot": {"k_flick_top": 0.0, "k_loft_m": 0.5}})
    assert rig.game.shot_model.k_flick_top == 0.0 and rig.game.shot_model.k_loft_m == 0.5
    assert rig.game.shot_model.k_aim == M.k_aim                                            # the rest is untouched
    rig.close()
