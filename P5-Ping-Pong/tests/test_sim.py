"""tools/sim.py: the whole game loop, no sensors: how the levels and your swing strength play out."""

import pytest

from tools import sim

POINTS = 50


@pytest.fixture(scope="module")
def table():
    return sim.simulate(levels=(1, 2, 3), swings=(("soft", 350.0), ("medium", 700.0), ("hard", 1200.0)),
                        points=POINTS, seed=1)


def cell(rows, level, swing):
    return next(r for r in rows if r["level"] == level and r["swing"] == swing)


def test_a_harder_swing_makes_the_computer_miss_more_at_every_level(table):
    for level in (1, 2, 3):
        p = [cell(table, level, s)["p_cpu_miss"] for s in ("soft", "medium", "hard")]
        assert p[0] <= p[1] + 0.05 and p[1] <= p[2] + 0.05 and p[2] > p[0], (level, p)


def test_a_higher_level_makes_the_computer_miss_less_at_every_swing_strength(table):
    for swing in ("soft", "medium", "hard"):
        p = [cell(table, level, swing)["p_cpu_miss"] for level in (1, 2, 3)]
        assert p[0] + 0.05 >= p[1] and p[1] + 0.05 >= p[2] and p[0] > p[2], (swing, p)


def test_a_perfect_player_never_faults_and_rallies_shrink_as_the_computer_misses_more(table):
    assert all(r["fault_rate"] == 0.0 for r in table)
    soft, hard = cell(table, 1, "soft"), cell(table, 1, "hard")
    assert soft["rally"] > 3 * hard["rally"]


def test_a_sloppy_player_faults_on_hard_swings_where_the_threshold_is_low():
    sloppy = sim.simulate(levels=(3,), swings=(("hard", 1200.0),), points=POINTS, seed=2, quality=0.3)
    perfect = sim.simulate(levels=(3,), swings=(("hard", 1200.0),), points=POINTS, seed=2, quality=1.0)
    assert sloppy[0]["fault_rate"] > 0.05 and perfect[0]["fault_rate"] == 0.0


def test_the_same_seed_gives_the_same_table():
    a = sim.simulate(levels=(2,), swings=(("medium", 700.0),), points=20, seed=5)
    b = sim.simulate(levels=(2,), swings=(("medium", 700.0),), points=20, seed=5)
    assert a == b


def test_the_table_text_names_levels_swings_and_speeds(table):
    text = sim.format_table(table, points=POINTS, quality=1.0)
    for needle in ("Rookie", "Club", "Pro", "soft", "hard", "km/h", "CPU misses"):
        assert needle in text, needle


def test_the_cli_prints_a_table_and_validates_its_arguments(capsys):
    assert sim.main(["--points", "15", "--levels", "1,3", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Rookie" in out and "Pro" in out and "Club" not in out
    assert sim.main(["--levels", "9"]) == 2


def test_the_selftest_is_green():
    assert sim.main(["--selftest"]) == 0
