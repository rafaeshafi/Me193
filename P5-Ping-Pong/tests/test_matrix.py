"""What the UNO Q's LED matrix (8 rows x 13 columns, 8 brightness levels) shows of the game: the streak you are on and your best when
no game is going, the scoreboard while one is."""

import itertools

import pytest

from pingpong import hud, matrix

ROWS, COLS = matrix.ROWS, matrix.COLS
TWELVE = [".#..###", "##....#", ".#..###", ".#..#..", "###.###"]          # "12" in the 3 x 5 font: a 1, a blank column, a 2
TWENTY_FIVE = ["###.###", "..#.#..", "###.###", "#.....#", "###.###"]


def at(frame, row, col):
    return frame[row * COLS + col]


def lit(frame, rows, cols, level=1):
    """The pixels of the block rows x cols that are lit at least this much, as strings of # and . to see and to compare."""
    return ["".join("#" if at(frame, r, c) >= level else "." for c in cols) for r in rows]


def lit_pixels(frame):
    return [(r, c) for r in range(ROWS) for c in range(COLS) if at(frame, r, c)]


def state(**kw):
    kw.setdefault("screen", "GAME")
    return hud.HudState(**kw)


def test_a_frame_is_eight_rows_of_thirteen_levels_between_zero_and_seven():
    for scene in (("idle", 12, 25), ("idle", 123, 5), ("rally", 7, 20), ("rally", 0, 0), ("match", 3, 5), ("match", 12, 10), ("standby",)):
        for t in (0.0, 1.7, 3.2, 5.9):
            frame = matrix.frame(scene, t)
            assert len(frame) == ROWS * COLS and all(isinstance(v, int) and 0 <= v <= 7 for v in frame), scene


# --- the scoreboard in a game ------------------------------------------------------------------------------------------------------------
def test_a_rally_shows_the_streak_in_big_digits_in_the_middle():
    frame = matrix.frame(("rally", 7, 0), 0.0)
    assert lit(frame, range(0, 5), range(5, 8), level=7) == ["###", "..#", "..#", "..#", "..#"]
    assert not any(at(frame, r, c) for r in range(0, 5) for c in list(range(0, 5)) + list(range(8, COLS)))
    two = matrix.frame(("rally", 12, 0), 0.0)
    assert lit(two, range(0, 5), range(3, 10), level=7) == TWELVE


def test_a_rally_shows_how_far_you_are_to_your_best_as_a_bar_along_the_bottom():
    frame = matrix.frame(("rally", 3, 13), 0.0)
    assert lit(frame, [6, 7], range(COLS), level=4) == ["###" + "." * 10] * 2
    assert all(at(frame, r, c) == 1 for r in (6, 7) for c in range(3, COLS))               # the rest of the way is a dim track
    assert not any(at(frame, 5, c) for c in range(COLS))
    assert sum(at(matrix.frame(("rally", 5, 20), 0.0), 7, c) >= 4 for c in range(COLS)) == 3        # 13 * 5 / 20 = 3.25 columns


def test_with_no_best_yet_the_bar_is_only_a_track():
    frame = matrix.frame(("rally", 4, 0), 0.0)
    assert all(at(frame, r, c) == 1 for r in (6, 7) for c in range(COLS))


def test_at_or_above_your_best_the_bar_is_full_and_flashes():
    frames = [matrix.frame(("rally", 9, 9), t / 20) for t in range(20)]
    assert all(len({at(f, r, c) for r in (6, 7) for c in range(COLS)}) == 1 for f in frames)        # the whole bar together
    assert {at(f, 7, 0) for f in frames} == {7, 2}                                                  # full bright, then dim: a flash
    beyond = matrix.frame(("rally", 30, 9), 0.0)
    assert {at(beyond, 7, c) for c in range(COLS)} == {7}


def test_a_streak_of_three_digits_still_fits():
    frame = matrix.frame(("rally", 123, 200), 0.0)
    assert lit(frame, range(0, 5), range(1, 12), level=7) == [".#..###.###", "##....#...#", ".#..###.###", ".#..#.....#", "###.###.###"]
    assert not any(at(frame, r, c) for r in range(5) for c in (0, 12))                       # nothing is cut off at either edge


def test_a_match_shows_your_points_a_dash_and_the_computers():
    frame = matrix.frame(("match", 3, 5), 0.0)
    assert lit(frame, range(1, 6), range(1, 4), level=7) == ["###", "..#", "###", "..#", "###"]
    assert lit(frame, range(1, 6), range(9, 12), level=7) == ["###", "#..", "###", "..#", "###"]
    assert lit(frame, range(1, 6), range(5, 8), level=7) == ["...", "...", "###", "...", "..."]
    assert not any(at(frame, r, c) for r in (0, 6, 7) for c in range(COLS))


@pytest.mark.parametrize("you, cpu", list(itertools.product(range(0, 20), repeat=2)))
def test_every_score_up_to_nineteen_fits_in_the_matrix_and_nothing_is_drawn_over_anything_else(you, cpu):
    items = matrix.score_items(you, cpu)                                   # [(left column, bitmap)]: the numbers and what is between them
    edges = [(left, left + len(rows[0])) for left, rows in items]
    assert edges[0][0] >= 0 and edges[-1][1] <= COLS
    assert all(a[1] <= b[0] for a, b in zip(edges, edges[1:]))              # in order, none over another
    frame = matrix.frame(("match", you, cpu), 0.0)
    assert len(lit_pixels(frame)) == sum(row.count("#") for _, rows in items for row in rows)


# --- when no game is going --------------------------------------------------------------------------------------------------------------
def test_without_a_game_it_alternates_between_the_streak_you_are_on_and_your_best():
    now, best = matrix.frame(("idle", 12, 25), 0.5), matrix.frame(("idle", 12, 25), 3.5)
    assert now != best
    assert lit(now, range(1, 6), range(0, 3)) == ["###", "#..", "#..", "#..", "###"]                          # C
    assert lit(best, range(1, 6), range(0, 3)) == ["##.", "#.#", "##.", "#.#", "##."]                        # B
    assert lit(now, range(1, 6), range(6, 13), level=7) == TWELVE
    assert lit(best, range(1, 6), range(6, 13), level=7) == TWENTY_FIVE
    assert matrix.frame(("idle", 12, 25), 0.5) == matrix.frame(("idle", 12, 25), 6.5)                        # every six seconds


def test_a_number_of_three_digits_is_shown_alone_in_the_middle():
    frame = matrix.frame(("idle", 123, 5), 0.5)
    assert lit(frame, range(1, 6), range(1, 12), level=7) == [".#..###.###", "##....#...#", ".#..###.###", ".#..#.....#", "###.###.###"]
    assert not any(at(frame, r, c) for r in range(ROWS) for c in (0, 12))


def test_the_letter_is_dimmer_than_the_number():
    frame = matrix.frame(("idle", 7, 9), 0.5)
    assert 0 < max(at(frame, r, c) for r in range(ROWS) for c in range(0, 3)) < 7
    assert max(at(frame, r, c) for r in range(ROWS) for c in range(6, COLS)) == 7


def test_the_standby_pattern_is_three_dim_dots():
    frame = matrix.frame(("standby",), 0.0)
    assert lit_pixels(frame) == [(3, 4), (3, 6), (3, 8)] and max(frame) < 7


# --- which of them the game is in ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("screen", ["INTRO", "TITLE", "MODE", "ONLINE", "WAIT", "OPPONENT", "VS", "RESULTS"])
def test_on_every_screen_round_the_game_it_is_the_streak_and_the_best(screen):
    assert matrix.scene_of(state(screen=screen, phase="MATCH_OVER", streak=7, record=12)) == ("idle", 7, 12)


@pytest.mark.parametrize("phase", ["COUNTDOWN", "RALLY", "POINT_OVER"])
def test_during_a_rally_game_the_scoreboard_is_the_streak_and_the_best(phase):
    assert matrix.scene_of(state(phase=phase, mode="survival", streak=4, record=9)) == ("rally", 4, 9)


@pytest.mark.parametrize("phase", ["COUNTDOWN", "RALLY", "POINT_OVER"])
def test_during_a_match_the_scoreboard_is_the_points(phase):
    assert matrix.scene_of(state(phase=phase, mode="match", player_points=3, cpu_points=5, streak=2, record=6)) == ("match", 3, 5)


@pytest.mark.parametrize("phase", ["LOBBY", "MATCH_OVER"])
def test_the_lobby_and_a_finished_game_without_menus_are_not_a_game(phase):
    assert matrix.scene_of(state(phase=phase, mode="match", streak=3, record=8)) == ("idle", 3, 8)


def test_a_game_with_a_friend_is_a_match_like_any_other():
    assert matrix.scene_of(state(phase="RALLY", mode="match", opponent_name="MAYA", player_points=1, cpu_points=0)) == ("match", 1, 0)


def test_a_real_game_goes_from_the_menus_to_a_scoreboard_and_back_to_the_streak_it_ended_on():
    from test_session_flow import ELSEWHERE, SPACE, hold, key, make, settle

    from pingpong import app

    s = make()
    settle(s)
    assert matrix.scene_of(s.hud_state()) == ("idle", 0, 0)
    key(s, SPACE)
    assert matrix.scene_of(s.hud_state()) == ("rally", 0, 0)                      # the countdown is already the scoreboard
    app.play_until_hits(s, 3)
    assert matrix.scene_of(s.hud_state()) == ("rally", 3, 3)
    while s.game.phase != "MATCH_OVER":                                          # nobody swings: the next ball ends it
        hold(s, ELSEWHERE, 0.1)
    hold(s, ELSEWHERE, 1.5)
    assert matrix.scene_of(s.hud_state()) == ("idle", 3, 3)                       # the last game's streak and the best, not 0


def test_a_real_match_shows_the_points_as_they_come():
    from test_session_flow import ELSEWHERE, SPACE, hold, key, settle

    from pingpong import app, flow

    s = app.make_session(mode="match", target=2, level=1, flow=flow.Flow(intro=False, level_tag=1, mode="match"))
    settle(s)
    key(s, SPACE)
    assert matrix.scene_of(s.hud_state()) == ("match", 0, 0)
    while s.game.phase != "MATCH_OVER":
        hold(s, ELSEWHERE, 0.1)
    assert matrix.scene_of(s.hud_state())[0] == "idle"
    assert (s.game.player_points, s.game.cpu_points) == (0, 2)
