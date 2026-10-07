"""CpuPolicy: serve selection (softmax over wrong-footing zones) and the Match miss model."""

import random

import pytest

from pingpong import levels, policy

ROOKIE, CLUB, PRO, INSANE = (levels.LEVELS[i] for i in (1, 2, 3, 4))


def pol(seed=1):
    return policy.CpuPolicy(random.Random(seed))


def test_serve_speed_follows_the_level_and_the_survival_ramp():
    p = pol()
    plain = p.serve(CLUB, s_prev=0.5, n_hits=0, player_a=0.5, survival=False)
    assert plain.v == pytest.approx(levels.incoming_speed(CLUB, 0.5))
    ramped = p.serve(CLUB, s_prev=0.5, n_hits=15, player_a=0.5, survival=True)   # 15: not a "special" ball
    assert ramped.special is False
    assert ramped.v == pytest.approx(levels.incoming_speed(CLUB, 0.5, ramp=levels.survival_ramp(15)))


def test_rookie_uses_a_uniform_softmax_over_all_nine_zones():
    p = pol(3)
    zones = {p.serve(ROOKIE, 0.5, 0, 0.5, survival=False).aim_ab for _ in range(400)}
    assert len(zones) == 9


def test_a_sharp_opponent_wrong_foots_the_player():
    # player's hand is far left; a low temperature should aim to the RIGHT most of the time
    p = pol(5)
    aims = [p.serve(INSANE, 0.5, 0, player_a=0.1, survival=False).aim_ab[0] for _ in range(300)]
    right = sum(1 for a in aims if a > 0.5) / len(aims)
    left = sum(1 for a in aims if a < 0.5) / len(aims)
    assert right > 0.6 and left < 0.05        # by the plan's utility the far edge gets ~69%


def test_zone_utility_prefers_far_zones_and_penalises_the_edges_at_speed():
    near = policy.zone_utility((0.5, 0.5), player_a=0.5, v=5.0, v_max=14.0, player_va=0.0)
    far = policy.zone_utility((0.85, 0.5), player_a=0.1, v=5.0, v_max=14.0, player_va=0.0)
    assert far > near
    slow_edge = policy.zone_utility((0.85, 0.5), 0.1, 3.0, 14.0, 0.0)
    fast_edge = policy.zone_utility((0.85, 0.5), 0.1, 14.0, 14.0, 0.0)
    assert fast_edge < slow_edge


def test_balls_arrive_inside_the_levels_share_of_the_reach_box():
    p = pol(5)
    for level in (ROOKIE, CLUB, PRO):
        half = 0.35 * level.reach                                   # the zones sit 0.35 either side of the box's middle
        aims = [p.serve(level, 0.5, 0, 0.5, survival=False).aim_ab for _ in range(300)]
        assert all(abs(a - 0.5) <= half + 1e-9 and abs(b - 0.5) <= half + 1e-9 for a, b in aims)
        assert max(abs(a - 0.5) for a, _ in aims) == pytest.approx(half)          # the edge zones are still used


def test_a_full_reach_changes_nothing_so_the_learners_zone_labels_stay_exact():
    assert policy.in_reach((0.15, 0.85), 1.0) == (0.15, 0.85)
    assert policy.in_reach((0.15, 0.85), 0.6) == pytest.approx((0.29, 0.71))


def test_survival_specials_come_every_tenth_ball_and_wide_balls_grow_with_the_rally():
    p = pol(9)
    assert p.serve(CLUB, 0.5, n_hits=10, player_a=0.5, survival=True).special is True
    assert p.serve(CLUB, 0.5, n_hits=9, player_a=0.5, survival=True).special is False
    few = sum(abs(p.serve(CLUB, 0.5, 2, 0.5, True).aim_ab[0] - 0.5) > 0.15 for _ in range(300))
    many = sum(abs(p.serve(CLUB, 0.5, 60, 0.5, True).aim_ab[0] - 0.5) > 0.15 for _ in range(300))
    assert many > few


def test_miss_probability_grows_with_speed_spin_and_reach_deficit_and_is_capped():
    base = policy.miss_probability(CLUB, v=6.0, spin_amp=0.2, reach_deficit_m=0.0)
    assert base == pytest.approx(CLUB.p0)
    assert policy.miss_probability(CLUB, 12.0, 0.2, 0.0) > base
    assert policy.miss_probability(CLUB, 6.0, 0.9, 0.0) > base
    assert policy.miss_probability(CLUB, 6.0, 0.2, 0.5) > base
    assert policy.miss_probability(CLUB, 40.0, 1.0, 5.0) == 0.95


def test_harder_levels_miss_less_for_the_same_shot():
    shot = dict(v=10.0, spin_amp=0.5, reach_deficit_m=0.2)
    probs = [policy.miss_probability(levels.LEVELS[i], **shot) for i in (1, 2, 3, 4)]
    assert probs == sorted(probs, reverse=True)


def test_reach_deficit_is_physical_not_a_coin_flip():
    # the CPU moves at its speed limit after its reaction delay
    reach = CLUB.cpu_speed_ms * max(0.0, 0.6 - CLUB.tau_s)
    assert policy.reach_deficit_m(CLUB, x_land_m=0.0, x_cpu_m=0.0, flight_s=0.6) == 0.0
    assert policy.reach_deficit_m(CLUB, x_land_m=reach + 0.3, x_cpu_m=0.0, flight_s=0.6) == pytest.approx(0.3)
    assert policy.reach_deficit_m(CLUB, x_land_m=0.5, x_cpu_m=0.0, flight_s=CLUB.tau_s) == pytest.approx(0.5)


def test_survival_never_misses_over_ten_thousand_decisions():
    p = pol(11)
    assert all(p.returns(CLUB, v=14.0, spin_amp=1.0, reach_deficit_m=9.9, survival=True) for _ in range(10_000))


def test_match_decisions_follow_the_probability():
    p = pol(4)
    easy = sum(p.returns(ROOKIE, 4.0, 0.0, 0.0, survival=False) for _ in range(2000)) / 2000
    brutal = sum(p.returns(CLUB, 40.0, 1.0, 5.0, survival=False) for _ in range(2000)) / 2000
    assert easy > 0.85 and brutal < 0.1


def test_same_seed_gives_the_same_serve_sequence():
    a = [pol(21).serve(CLUB, 0.5, n, 0.3, True) for n in range(5)]
    b = [pol(21).serve(CLUB, 0.5, n, 0.3, True) for n in range(5)]
    assert a == b
