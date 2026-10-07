"""The first live game (2026-10-07, Rookie survival): nine balls, no hit, and three of them were real hits.

Balls 3, 4 and 5 were swung within 25 ms of their arrival, with the hand passing 0.16-0.39 shoulder widths from the ball
at the moment of impact (ball 8, 100 ms early, swept through the ball's height 35 ms after its impact).  The position
gate threw them out: its approach window ended 50 ms BEFORE the impact (where the hand is still a metre away on a stroke
that sweeps 10 SW/s), and its "still near the ball" check looked at the newest pose, which at detection time (up to
150 ms after the peak) is the follow-through.  Balls 1, 2 and 7 were nowhere near: the hand never left the middle.
tests/data/real_game1_balls.json holds the swing, the ball and the poses around each of those balls.
"""

import json
from pathlib import Path

import pytest

from pingpong import levels
from pingpong.events import PaddlePose, SwingEvent
from pingpong.judge import BallWindow, HitJudge
from pingpong.paddle import ReachBox

DATA = json.loads((Path(__file__).parent / "data" / "real_game1_balls.json").read_text())
BALLS = {b["ball_id"]: b for b in DATA["balls"]}


def judged(ball_id, **judge_kw):
    b = BALLS[ball_id]
    swing = SwingEvent(kind="IMPACT", **{k: (tuple(v) if isinstance(v, list) else v) for k, v in b["swing"].items()})
    ball = BallWindow(ball_id=ball_id, t_c_ns=b["t_c_ns"], aim_ab=tuple(b["aim_ab"]), level=levels.LEVELS[1])
    poses = [PaddlePose(t_scene_ns=p["t"], u=p["u"], v=p["v"], conf=p["c"], hand=p["h"]) for p in b["poses"]]
    judge = HitJudge(ReachBox(**DATA["box"]), t_pk=DATA["t_pk"], **judge_kw)
    return judge.judge(swing, ball, poses, b["now_ns"])


def gate(verdict, name):
    return next(g for g in verdict.gates if g.name == name)


@pytest.mark.parametrize("ball_id", [3, 4, 5, 8])
def test_a_swing_with_the_hand_at_the_ball_at_the_impact_is_a_hit(ball_id):
    verdict = judged(ball_id)
    assert verdict.kind == "HIT", [(g.name, g.note) for g in verdict.gates if not g.passed]


@pytest.mark.parametrize("ball_id", [1, 2, 7])
def test_a_swing_with_the_hand_nowhere_near_the_ball_is_still_rejected(ball_id):
    verdict = judged(ball_id)
    assert verdict.kind == "REJECTED" and not gate(verdict, "J2").passed
    assert "SW from the ball" in gate(verdict, "J2").note


def test_the_distance_is_the_hands_closest_approach_around_the_impact():
    assert judged(4).d_min_sw < 0.25            # 0.16 SW at the impact; the old window stopped 50 ms short and saw 0.56
    assert 0.2 < judged(3).d_min_sw < 0.5       # 0.36 SW: the hand sweeps up through the ball's height at the impact
