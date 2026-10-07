"""Three live Rookie games (2026-10-07), judged by the old rules and by the new ones.

Fourteen balls were served and thirteen swings made (the old visuals: a flat ring to hit with a ball that did not
bounce).  Under the old rules five of them were hits.  The swings came early (the stroke's end a median 0.19 s before the
ball was over the paddle) and the old windows threw most of them out; the new rules date a swing by its stroke, accept
that, look only at where the hand is across the table, and so keep what was a real swing at the ball.
tests/data/real_games2_balls.json holds each ball's serve, the swings around it and the hand's poses (x_start assumed 0).
"""

import json
from pathlib import Path

import pytest

from pingpong import levels, physics
from pingpong.events import PaddlePose, SwingEvent
from pingpong.judge import BallWindow, HitJudge
from pingpong.paddle import ReachBox

S = 1_000_000_000
DATA = json.loads((Path(__file__).parent / "data" / "real_games2_balls.json").read_text())
OLD_HITS = 5                                     # what the rules of that day made of the same thirteen swings


def verdicts(**judge_kw):
    box, out = ReachBox(**DATA["box"]), []
    for b in DATA["balls"]:
        level = levels.LEVELS[b["level"]]
        leg = physics.plan_leg(b["arrival_ns"] - round(3.0 / b["v"] * S), b["v"], 0.0, tuple(b["aim_ab"]))
        ball = BallWindow(b["ball_id"], b["arrival_ns"], tuple(b["aim_ab"]), level, leg)
        poses = [PaddlePose(t_scene_ns=p["t"], u=p["u"], v=p["v"], conf=p["c"], hand=p["h"]) for p in b["poses"]]
        judge = HitJudge(box, t_pk=DATA["t_pk"], **judge_kw)
        for s in b["swings"]:
            swing = SwingEvent(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in s["swing"].items()})
            out.append(judge.judge(swing, ball, [p for p in poses if p.t_scene_ns <= s["now_ns"]], s["now_ns"]))
    return out


def test_the_new_rules_keep_most_of_the_swings_that_the_old_ones_threw_out():
    kinds = [v.kind for v in verdicts()]
    assert len(kinds) == 13
    assert kinds.count("HIT") >= 10 > OLD_HITS * 1.5
    assert kinds.count("REJECTED") == 0                       # nothing was lost to position or strength


def test_what_is_left_are_swings_far_too_early_to_have_been_meant_for_that_ball():
    for v in verdicts():
        if v.kind == "IGNORED":
            assert v.e_s < -0.5                               # at least half a second before the ball was there


def test_the_swings_that_hit_were_still_early_by_a_fifth_of_a_second_so_the_window_has_to_be_generous():
    early = sorted(v.e_s for v in verdicts() if v.kind == "HIT")
    assert early[len(early) // 2] < -0.1 and early[0] > -levels.LEVELS[1].early_s


def test_a_hit_is_not_given_for_nothing_a_hand_far_away_across_the_table_still_misses():
    box = ReachBox(**DATA["box"])
    b = DATA["balls"][0]
    level = levels.LEVELS[b["level"]]
    leg = physics.plan_leg(b["arrival_ns"] - round(3.0 / b["v"] * S), b["v"], 0.0, tuple(b["aim_ab"]))
    ball = BallWindow(b["ball_id"], b["arrival_ns"], tuple(b["aim_ab"]), level, leg)
    s = b["swings"][0]
    swing = SwingEvent(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in s["swing"].items()})
    far = [PaddlePose(t_scene_ns=p["t"], u=p["u"] + 3.0, v=p["v"], conf=p["c"], hand=p["h"]) for p in b["poses"]
           if p["t"] <= s["now_ns"]]
    verdict = HitJudge(box, t_pk=DATA["t_pk"]).judge(swing, ball, far, s["now_ns"])
    assert verdict.kind == "REJECTED" and any(g.name == "J2" and not g.passed for g in verdict.gates)
