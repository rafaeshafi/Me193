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
    assert levels.LEVELS[1].reach == 0.6 and levels.LEVELS[2].reach == 0.8


def test_names_and_tags_line_up_with_the_printed_cards():
    assert [levels.LEVELS[i].name for i in (1, 2, 3)] == ["Rookie", "Club", "Pro"]
    assert levels.level_for_tag(1).name == "Rookie"
    assert levels.level_for_tag(3).name == "Pro"
    assert levels.level_for_tag(0) is None        # tag 0 is START, not a level


def test_flight_times_at_three_metres_match_the_plan_table():
    assert [round(3.0 / levels.LEVELS[i].v_tier, 2) for i in (1, 2, 3, 4)] == [0.86, 0.60, 0.43, 0.32]


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
