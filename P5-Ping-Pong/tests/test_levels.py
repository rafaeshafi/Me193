import pytest

from pingpong import levels


def test_levels_are_strictly_ordered_in_every_difficulty_dimension():
    order = [levels.LEVELS[i] for i in (1, 2, 3, 4)]
    for easier, harder in zip(order, order[1:]):
        assert harder.v_tier > easier.v_tier
        assert harder.early_s < easier.early_s and harder.late_s < easier.late_s
        assert harder.radius_sw < easier.radius_sw
        assert harder.tau_s < easier.tau_s
        assert harder.fault_th < easier.fault_th
        assert harder.cpu_speed_ms > easier.cpu_speed_ms


def test_easier_levels_throw_the_ball_into_a_smaller_part_of_the_reach_box():
    # the first live game: a Rookie was served to the far corners of a box he had stretched to reach, and never got there
    reach = [levels.LEVELS[i].reach for i in (1, 2, 3, 4)]
    assert reach == sorted(reach) and reach[-1] == 1.0
    assert levels.LEVELS[1].reach == 0.5 and levels.LEVELS[2].reach == 0.65 and levels.LEVELS[3].reach == 0.8


def test_names_and_tags_line_up_with_the_printed_cards():
    assert [levels.LEVELS[i].name for i in (1, 2, 3)] == ["Rookie", "Club", "Pro"]
    assert levels.level_for_tag(1).name == "Rookie"
    assert levels.level_for_tag(3).name == "Pro"
    assert levels.level_for_tag(0) is None        # tag 0 is START, not a level


def test_rookie_is_slow_enough_to_find_the_ring_and_bring_the_hand_to_it():
    # the first live game: 0.86 s was not enough to see where the ball was going and get there
    assert 3.0 / levels.LEVELS[1].v_tier >= 1.1
    assert levels.MODE_NAMES == {"survival": "RALLY", "match": "MATCH"}


def test_flight_times_at_three_metres_match_the_readmes_level_table():
    assert [round(3.0 / levels.LEVELS[i].v_tier, 2) for i in (1, 2, 3, 4)] == [1.5, 0.79, 0.58, 0.43]


def test_every_level_is_forgiving_enough_to_play_from_across_the_room():
    # 10/8, after playing them: each level was too hard, so each is slower to react to, wider to hit and easier to beat in a match
    assert [3.0 / levels.LEVELS[i].v_tier >= t for i, t in zip((1, 2, 3, 4), (1.4, 0.75, 0.55, 0.40))] == [True] * 4
    assert [levels.LEVELS[i].radius_sw >= r for i, r in zip((1, 2, 3, 4), (0.80, 0.65, 0.50, 0.38))] == [True] * 4
    assert [levels.LEVELS[i].p0 >= p for i, p in zip((1, 2, 3, 4), (0.15, 0.09, 0.05, 0.025))] == [True] * 4
    assert levels.LEVELS[3].fault_th >= 0.5                                  # Pro still faults a hard sloppy hit, but not a mildly sloppy one


@pytest.mark.parametrize("easier, harder", [(1, 2), (2, 3), (3, 4)])
def test_incoming_speed_bands_never_overlap_so_the_tag_always_dominates(easier, harder):
    # v_in = v_tier * (0.9 + 0.2 * s_prev): the swing can only modulate +-10%.
    hi = max(levels.incoming_speed(levels.LEVELS[easier], s) for s in (0.0, 0.5, 1.0))
    lo = min(levels.incoming_speed(levels.LEVELS[harder], s) for s in (0.0, 0.5, 1.0))
    assert hi < lo


def test_survival_ramp_scales_speed_and_is_capped():
    club = levels.LEVELS[2]
    base = levels.incoming_speed(club, 0.5)
    assert levels.incoming_speed(club, 0.5, ramp=1.5) == pytest.approx(base * 1.5)
    assert levels.survival_ramp(0) == 1.0
    assert levels.survival_ramp(10) == pytest.approx(1.3)
    assert levels.survival_ramp(1000) == 1.8


def test_only_pro_and_insane_can_fault_a_hit_on_a_beginner_level_a_hit_is_a_hit():
    # the first live games: a real hit (hard, a little off) was called "fault: out" and ended the game, which read as a bug
    assert levels.LEVELS[1].fault_th >= 1.0 and levels.LEVELS[2].fault_th >= 1.0     # s * (1 - q) can never reach 1
    assert levels.LEVELS[3].fault_th < 1.0 and levels.LEVELS[4].fault_th < 1.0


def test_the_early_windows_fit_how_people_really_swing():
    # measured on the live recordings: the contact (peak + 0.1 s) was 0.1-0.35 s before the ball's nominal arrival
    assert levels.LEVELS[1].early_s >= 0.50 and levels.LEVELS[2].early_s >= 0.30
