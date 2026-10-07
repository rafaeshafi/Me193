"""The first live game (2026-10-07, Rookie survival): nine balls, no hit.  What the rules made of it, then and now.

Balls 3, 4 and 5 were swung within 25 ms of their arrival, with the hand passing 0.16-0.39 shoulder widths from the ball
at the moment of impact (ball 8, 100 ms early, swept through the ball's height 35 ms after its impact).  The first fix
(the position gate's window now includes the impact) made those four hits.  Balls 1, 2 and 7 were "nowhere near" in the
old 2-D measure only because the hand never rose or fell to the ball's height; ACROSS the court it was within 0.0-0.6
shoulder widths of them.  The game was redesigned after that (a table in perspective: the paddle slides along your end
and its height is free), and with the new rules every one of the seven judged swings is a hit.
tests/data/real_game1_balls.json holds the swing, the ball and the poses around each of those balls.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from pingpong import levels
from pingpong.events import PaddlePose, SwingEvent
from pingpong.judge import BallWindow, HitJudge
from pingpong.paddle import ReachBox

DATA = json.loads((Path(__file__).parent / "data" / "real_game1_balls.json").read_text())
BALLS = {b["ball_id"]: b for b in DATA["balls"]}


def judged(ball_id, level=1, **judge_kw):
    b = BALLS[ball_id]
    swing = SwingEvent(kind="IMPACT", **{k: (tuple(v) if isinstance(v, list) else v) for k, v in b["swing"].items()})
    ball = BallWindow(ball_id=ball_id, t_c_ns=b["t_c_ns"], aim_ab=tuple(b["aim_ab"]),
                      level=levels.LEVELS[level] if isinstance(level, int) else level)
    poses = [PaddlePose(t_scene_ns=p["t"], u=p["u"], v=p["v"], conf=p["c"], hand=p["h"]) for p in b["poses"]]
    judge = HitJudge(ReachBox(**DATA["box"]), t_pk=DATA["t_pk"], **judge_kw)
    return judge.judge(swing, ball, poses, b["now_ns"])


@pytest.mark.parametrize("ball_id", sorted(BALLS))
def test_every_judged_swing_of_the_first_game_is_a_hit_by_the_new_rules(ball_id):
    verdict = judged(ball_id)
    assert verdict.kind == "HIT", [(g.name, g.note) for g in verdict.gates if not g.passed]


def test_the_distance_is_the_hands_closest_approach_across_the_court_around_the_impact():
    assert judged(4).d_min_sw < 0.05            # right on the ball: 0.00 SW across (0.16 SW in the old 2-D measure)
    assert judged(3).d_min_sw < 0.05            # the height the hand swept through does not count any more
    assert judged(1).d_min_sw > 0.4             # the one that was wide of it sideways: inside Rookie's radius, but only just


def test_the_stroke_ends_close_to_the_balls_arrival_not_the_gyros_peak():
    # the peaks came 0.0-0.3 s before the ball's nominal arrival; the stroke ends (peak + 0.1 s) are close to it
    peaks = [judged(i, contact_lag_s=0.0).e_s for i in BALLS]
    ends = [judged(i).e_s for i in BALLS]
    assert sum(abs(e) for e in ends) < sum(abs(e) for e in peaks) * 1.2
    assert all(abs(e) < levels.LEVELS[1].early_s for e in ends)


def test_a_stricter_reach_still_asks_for_the_hand_to_be_near_the_ball_across_the_court():
    # at Pro's reach (0.42) ball 1 (0.57 SW wide of the hand) is no longer a hit, the other six still are
    tight = dataclasses.replace(levels.LEVELS[1], radius_sw=levels.LEVELS[3].radius_sw)
    assert judged(1, level=tight).kind == "REJECTED"
    assert all(judged(i, level=tight).kind == "HIT" for i in BALLS if i != 1)
