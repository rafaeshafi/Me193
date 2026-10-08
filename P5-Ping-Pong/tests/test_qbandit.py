"""qbandit: a tabular Q-learning opponent that learns where YOU fail (opt-in, per player, kept between games)."""

import random

import numpy as np
import pytest

from pingpong import app, levels, policy, qbandit
from pingpong.rules import GameCore


def test_an_update_moves_the_value_toward_the_reward_plus_the_discounted_best_next_value():
    q = qbandit.QBandit(alpha=0.5, gamma=0.9)
    q.table[3, 4] = 0.2
    q.table[5, :] = [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]
    q.update(3, 4, reward=1.0, next_state=5)
    first = 0.2 + 0.5 * (1.0 + 0.9 * 0.6 - 0.2)
    assert q.table[3, 4] == pytest.approx(first)
    q.update(3, 4, reward=1.0, next_state=None)                       # terminal: no next value
    assert q.table[3, 4] == pytest.approx(first + 0.5 * (1.0 - first))


def test_the_nine_states_are_the_hand_position_third_times_the_column_of_the_last_zone():
    q = qbandit.QBandit()
    states = {q.state(a, col) for a in (0.1, 0.5, 0.9) for col in (0, 1, 2)}
    assert states == set(range(9))
    assert q.state(0.05, 0) == 0 and q.state(0.95, 2) == 8


def test_exploration_shrinks_with_every_game_down_to_a_floor():
    q = qbandit.QBandit(epsilon=0.15, decay=0.5, floor=0.05)
    seen = []
    for _ in range(5):
        q.end_game()
        seen.append(q.epsilon)
    assert seen[0] == pytest.approx(0.075) and seen[-1] == pytest.approx(0.05) and seen == sorted(seen, reverse=True)


def test_it_learns_to_serve_where_the_player_keeps_failing():
    rng = random.Random(1)
    q = qbandit.QBandit(rng=random.Random(2), epsilon=0.25, floor=0.1)
    weak = 7                                                          # the player cannot return zone 7
    state, picks = q.state(0.5, 1), []
    for game in range(60):
        for _ in range(12):
            zone = q.choose(state, list(range(9)), weight=1.0, base=[0.0] * 9, temp=0.25)
            fails = rng.random() < (0.9 if zone == weak else 0.1)
            q.update(state, zone, reward=1.0 if fails else 0.05, next_state=state)
            picks.append(zone)
        q.end_game()
    late = picks[-120:]
    assert late.count(weak) / len(late) > 0.5                           # versus 1/9 for random aiming


def test_the_level_weight_decides_how_much_the_learning_matters():
    assert qbandit.weight(levels.LEVELS[1]) == 0.0
    assert 0.0 < qbandit.weight(levels.LEVELS[2]) < qbandit.weight(levels.LEVELS[3])
    q = qbandit.QBandit(rng=random.Random(0), epsilon=0.0, floor=0.0)
    q.table[4, 2] = 5.0
    picks = [q.choose(4, list(range(9)), weight=0.0, base=[0.0] * 9, temp=0.3) for _ in range(40)]
    assert len(set(picks)) > 1                                          # weight 0: the table is ignored


def test_the_table_survives_saving_and_loading_by_player(tmp_path):
    q = qbandit.QBandit(epsilon=0.11)
    q.table[2, 3], q.games = 0.7, 4
    qbandit.save_for("Rafae", q, root=tmp_path)
    again = qbandit.load_for("rafae", root=tmp_path)
    assert again.table[2, 3] == pytest.approx(0.7) and again.games == 4 and again.epsilon == pytest.approx(0.11)
    assert qbandit.load_for("nobody", root=tmp_path) is None
    (tmp_path / "rafae" / "qtable.npz").write_bytes(b"junk")
    with pytest.raises(ValueError, match="qtable"):
        qbandit.load_for("rafae", root=tmp_path)


def test_a_policy_without_a_learner_serves_exactly_as_before():
    a, b = policy.CpuPolicy(random.Random(5)), policy.CpuPolicy(random.Random(5), learner=None)
    plans_a = [a.serve(levels.LEVELS[2], 0.5, n, 0.5, True) for n in range(12)]
    plans_b = [b.serve(levels.LEVELS[2], 0.5, n, 0.5, True) for n in range(12)]
    assert plans_a == plans_b


def test_a_trained_table_changes_where_a_club_or_pro_opponent_aims_but_not_a_rookie():
    q = qbandit.QBandit(rng=random.Random(0), epsilon=0.0, floor=0.0)
    q.table[:, policy.ZONES.index((0.85, 0.85))] = 4.0                 # "serve to the far right corner", in any state
    for tag, expect_more in ((1, False), (3, True)):
        p = policy.CpuPolicy(random.Random(3), learner=q)
        picks = [p.serve(levels.LEVELS[tag], 0.5, 0, 0.5, False).aim_ab for _ in range(60)]
        share = picks.count(policy.in_reach((0.85, 0.85), levels.LEVELS[tag].reach)) / len(picks)
        assert (share > 0.5) is expect_more, (tag, share)


def test_the_policy_updates_the_learner_from_what_the_game_reports():
    q = qbandit.QBandit(alpha=1.0, gamma=0.0, rng=random.Random(0), epsilon=0.0, floor=0.0)
    p = policy.CpuPolicy(random.Random(1), learner=q)
    plan = p.serve(levels.LEVELS[3], 0.5, 0, 0.5, False)
    zone = next(i for i, z in enumerate(policy.ZONES) if policy.in_reach(z, levels.LEVELS[3].reach) == plan.aim_ab)
    p.observe(0.3, terminal=False)
    assert q.table.sum() == 0.0                                         # waits for the next serve to learn from it
    p.serve(levels.LEVELS[3], 0.5, 1, 0.5, False)
    assert q.table[:, zone].max() == pytest.approx(0.3)
    p.observe(1.0, terminal=True)
    assert q.table.max() == pytest.approx(1.0)                          # a terminal reward is learned at once


def test_the_game_reports_hit_quality_faults_and_misses_to_the_policy():
    class Spy(policy.CpuPolicy):
        def __init__(self):
            super().__init__(random.Random(1))
            self.rewards = []

        def observe(self, reward, terminal=False):
            self.rewards.append((round(reward, 2), terminal))

    spy = Spy()
    session = app.make_session()
    session.game.policy = spy
    app.play_until_hits(session, 2)
    assert spy.rewards and all(0.0 <= r <= 0.5 and not t for r, t in spy.rewards)      # good returns: small rewards
    session.clock.advance_s(6.0)
    session.tick()                                                                        # a miss ends the game
    assert spy.rewards[-1] == (1.0, True)
