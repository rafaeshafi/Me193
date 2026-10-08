"""The way into a game, like a console sports game: an intro flying down to the court, a title, a choice of game and of
opponent, a face-off, and after the match its results.  Everything can be pointed at with the hub and held on (the keys and the
AprilTag cards do the same), and nothing needs the keyboard."""

import pytest

from pingpong import flow as flowmod
from pingpong.flow import Action, Flow

S = 1_000_000_000
STEP = S // 60
SPACE, ENTER, BACK, ESC, Q = 32, 13, 8, 27, ord("q")
D, A = ord("."), ord(",")
ELSEWHERE = (0.5, 0.9)                           # (a, b) with b up: y = 0.1, over no button
START_AT = (0.95, 0.9)                           # the top right: the START button, and "play again"
LEFT_TOP = (0.05, 0.9)                           # the top left: "change opponent"
RALLY_AT, MATCH_AT = (0.28, 0.45), (0.72, 0.45)
OPP_AT = {1: (0.22, 0.4), 2: (0.5, 0.4), 3: (0.78, 0.4)}
BACK_AT = (0.08, 0.07)


class Drive:
    """Feeds a flow at 60 Hz and keeps every action it asks for, with the second it came."""

    def __init__(self, flow=None, **kw):
        self.flow, self.t, self.log = flow or Flow(**kw), 0, []

    def run(self, ab, seconds, phase="LOBBY"):
        for _ in range(round(seconds * 60)):
            self.t += STEP
            for action in self.flow.update(self.t, ab, phase):
                self.log.append((self.t / S, action))
        return self

    def key(self, key):
        actions = self.flow.on_key(key, self.t)
        for action in actions or ():
            self.log.append((self.t / S, action))
        return actions

    def tag(self, role, value=0):
        actions = self.flow.on_tag(role, value, self.t)
        for action in actions or ():
            self.log.append((self.t / S, action))
        return actions

    def started(self):
        return [a for _, a in self.log if a.kind == "start"]

    def sounds(self):
        return [a.value for _, a in self.log if a.kind == "sound"]

    def haptics(self):
        return [a.value for _, a in self.log if a.kind == "haptic"]

    def music(self):
        return [a.value for _, a in self.log if a.kind == "music"]

    def on(self, screen, ab=ELSEWHERE, seconds=0.3):
        """Settle on a screen, the hand away from every button."""
        self.run(ab, seconds)
        assert self.flow.screen == screen, (self.flow.screen, screen)
        return self


def at_title(**kw):
    return Drive(intro=False, **kw).on("TITLE")


def at_mode(**kw):
    d = at_title(**kw)
    d.run(START_AT, 1.8)
    return d.on("MODE")


def at_opponents(**kw):
    d = at_mode(**kw)
    d.run(MATCH_AT, 1.5)
    return d.on("OPPONENT")


# --- the intro ---------------------------------------------------------------------------------------------------------------------
def test_a_flow_starts_with_the_intro_or_goes_straight_to_the_title_when_told_to():
    assert Flow().screen == "INTRO" and Flow(intro=False).screen == "TITLE"


def test_the_intro_plays_for_its_length_and_then_the_title_comes_up_by_itself():
    d = Drive(intro_s=3.0)
    d.run(ELSEWHERE, 2.8)
    assert d.flow.screen == "INTRO"
    d.run(ELSEWHERE, 0.4)
    assert d.flow.screen == "TITLE"


def test_any_key_skips_the_intro_but_quitting_still_quits():
    d = Drive()
    d.run(ELSEWHERE, 0.5)
    assert d.key(ESC) is None and d.key(Q) is None and d.flow.screen == "INTRO"      # not taken: the game quits
    assert d.key(SPACE) is not None and d.flow.screen == "TITLE"


def test_the_start_card_skips_the_intro():
    d = Drive()
    d.run(ELSEWHERE, 0.5)
    d.tag("START")
    assert d.flow.screen == "TITLE"


def test_holding_the_hub_in_the_top_right_skips_the_intro():
    d = Drive()
    d.run(ELSEWHERE, 0.5)
    d.run(START_AT, 1.8)
    assert d.flow.screen in ("TITLE", "MODE") and d.flow.screen != "INTRO"


def test_the_intro_asks_for_the_title_music_and_then_the_menu_music():
    d = Drive(intro_s=2.0)
    d.run(ELSEWHERE, 2.5)
    assert d.music()[0] == "intro" and d.music()[-1] == "menu"


# --- the title ---------------------------------------------------------------------------------------------------------------------
def test_holding_the_hub_on_start_for_a_second_and_a_half_goes_to_the_choice_of_game():
    d = at_title()
    d.run(START_AT, 1.3)
    assert d.flow.screen == "TITLE"
    d.run(START_AT, 0.4)
    assert d.flow.screen == "MODE" and d.started() == []


def test_a_hand_resting_on_the_start_button_when_the_title_comes_up_does_not_press_it():
    d = Drive(intro_s=1.0)
    d.run(START_AT, 4.0)                                      # the hand never left the corner: the intro ends, nothing is pressed
    assert d.flow.screen == "TITLE"
    d.run(ELSEWHERE, 0.2)
    d.run(START_AT, 1.7)
    assert d.flow.screen == "MODE"


def test_space_starts_a_game_at_once_with_the_choices_so_far_and_enter_goes_on_to_the_menus():
    d = at_title(level_tag=2, mode="match")
    d.key(SPACE)
    assert d.started() == [Action("start", (2, "match"))] and d.flow.screen == "GAME"
    e = at_title()
    e.key(ENTER)
    assert e.flow.screen == "MODE" and e.started() == []


def test_the_start_card_starts_a_game_at_once_and_a_level_card_picks_the_opponent():
    d = at_title()
    d.tag("LEVEL", 3)
    assert d.flow.level_tag == 3
    d.tag("START")
    assert d.started() == [Action("start", (3, "survival"))] and d.flow.screen == "GAME"


def test_the_old_keys_still_work_digits_choose_the_level_and_m_the_game():
    d = at_title()
    d.key(ord("2"))
    d.key(ord("m"))
    assert d.flow.level_tag == 2 and d.flow.mode == "match"
    d.key(ord("m"))
    assert d.flow.mode == "survival"


# --- the choice of game ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("target, mode", [(RALLY_AT, "survival"), (MATCH_AT, "match")])
def test_pointing_at_a_game_for_a_moment_chooses_it_and_goes_on_to_the_opponents(target, mode):
    d = at_mode()
    d.run(target, 1.1)
    assert d.flow.screen == "MODE"
    d.run(target, 0.3)
    assert d.flow.screen == "OPPONENT" and d.flow.mode == mode


def test_back_goes_to_the_title_with_the_hub_or_the_delete_key():
    d = at_mode()
    d.run(BACK_AT, 1.3)
    assert d.flow.screen == "TITLE"
    e = at_mode()
    e.key(BACK)
    assert e.flow.screen == "TITLE"


def test_the_keys_move_along_the_choices_and_enter_takes_the_one_in_focus():
    d = at_mode()
    assert d.flow.focus == 0                                   # a rally, the first of the two games
    d.key(D)
    assert d.flow.focus == 1
    d.key(D)
    assert d.flow.focus == 1                                   # the end of the row
    d.key(ENTER)
    assert d.flow.screen == "OPPONENT" and d.flow.mode == "match"


def test_the_choice_of_game_starts_with_the_one_already_chosen_in_focus():
    assert at_mode(mode="match").flow.focus == 1


# --- the choice of opponent and the face-off ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("level", [1, 2, 3])
def test_pointing_at_an_opponent_chooses_the_level_and_the_face_off_follows(level):
    d = at_opponents()
    d.run(OPP_AT[level], 1.5)
    assert d.flow.screen == "VS" and d.flow.level_tag == level and d.started() == []


def test_the_face_off_lasts_a_moment_and_then_the_game_starts_with_those_choices():
    d = at_opponents()
    d.run(OPP_AT[2], 1.5)
    assert d.flow.screen == "VS"
    d.run(ELSEWHERE, flowmod.VS_S + 0.2)
    assert d.started() == [Action("start", (2, "match"))] and d.flow.screen == "GAME"


def test_space_cuts_the_face_off_short_and_delete_goes_back_to_the_opponents():
    d = at_opponents()
    d.run(OPP_AT[3], 1.5)
    d.key(SPACE)
    assert d.started() == [Action("start", (3, "match"))]
    e = at_opponents()
    e.run(OPP_AT[1], 1.5)
    e.key(BACK)
    assert e.flow.screen == "OPPONENT" and e.started() == []


def test_the_keys_pick_an_opponent_digits_move_the_focus_and_enter_confirms():
    d = at_opponents()
    assert d.flow.focus == d.flow.level_tag - 1
    d.key(D)
    d.key(D)
    assert d.flow.focus == 2
    d.key(ord("1"))
    assert d.flow.focus == 0 and d.flow.level_tag == 1
    d.key(D)
    d.key(ENTER)
    assert d.flow.screen == "VS" and d.flow.level_tag == 2


def test_a_level_card_shown_on_the_opponent_screen_moves_the_focus_to_that_opponent():
    d = at_opponents()
    d.tag("LEVEL", 3)
    assert d.flow.focus == 2 and d.flow.level_tag == 3 and d.flow.screen == "OPPONENT"


def test_the_start_card_on_the_opponent_screen_starts_the_game_with_the_one_in_focus():
    d = at_opponents()
    d.tag("LEVEL", 2)
    d.tag("START")
    assert d.started() == [Action("start", (2, "match"))]


# --- the game and its results ---------------------------------------------------------------------------------------------------------
def started_game(**kw):
    d = at_title(**kw)
    d.key(SPACE)
    d.run(ELSEWHERE, 0.2, phase="COUNTDOWN")
    return d


def test_while_a_game_is_on_the_flow_stays_out_of_the_way():
    d = started_game()
    assert d.flow.screen == "GAME"
    assert d.key(ord("x")) is None and d.key(SPACE) is None                  # not taken: the game's own keys
    assert d.tag("LEVEL", 2) is None


def test_a_game_that_did_not_start_sends_the_player_back_to_the_title():
    d = at_title()
    d.key(SPACE)
    d.flow.started(False)
    assert d.flow.screen == "TITLE"


def test_when_the_match_is_over_the_results_come_up_and_play_again_starts_the_same_game_with_no_face_off():
    d = started_game(level_tag=3, mode="match")
    d.run(ELSEWHERE, 0.2, phase="MATCH_OVER")
    assert d.flow.screen == "RESULTS"
    d.run(ELSEWHERE, 0.2, phase="MATCH_OVER")
    d.run(START_AT, 1.6, phase="MATCH_OVER")
    assert d.started()[-1] == Action("start", (3, "match")) and d.flow.screen == "GAME"


def test_from_the_results_the_hub_in_the_top_left_goes_to_the_choice_of_opponent_and_so_does_delete():
    d = started_game()
    d.run(ELSEWHERE, 0.3, phase="MATCH_OVER")
    d.run(LEFT_TOP, 1.5, phase="MATCH_OVER")
    assert d.flow.screen == "OPPONENT"
    e = started_game()
    e.run(ELSEWHERE, 0.3, phase="MATCH_OVER")
    e.key(BACK)
    assert e.flow.screen == "OPPONENT"


def test_space_the_start_card_and_enter_play_again_from_the_results():
    for press in (lambda d: d.key(SPACE), lambda d: d.key(ENTER), lambda d: d.tag("START")):
        d = started_game(level_tag=2)
        d.run(ELSEWHERE, 0.3, phase="MATCH_OVER")
        before = len(d.started())
        press(d)
        assert len(d.started()) == before + 1 and d.started()[-1] == Action("start", (2, "survival"))


def test_a_hand_that_ended_the_game_up_in_the_corner_does_not_press_play_again():
    d = started_game()
    d.run(START_AT, 5.0, phase="MATCH_OVER")                                 # the hand stays in the corner through the results
    assert d.flow.screen == "RESULTS" and len(d.started()) == 1


# --- the senses of it ------------------------------------------------------------------------------------------------------------------------
def test_moving_onto_a_button_ticks_and_ticks_the_motors_but_not_over_and_over():
    d = at_mode()
    ticks = lambda: (d.sounds().count("menu_tick"), d.haptics().count("menu_tick"))      # noqa: E731
    sound0, haptic0 = ticks()
    d.run(RALLY_AT, 0.2)
    assert ticks() == (sound0 + 1, haptic0 + 1)
    d.run(MATCH_AT, 0.2)
    d.run(RALLY_AT, 0.2)
    d.run(MATCH_AT, 0.2)
    assert ticks()[0] - sound0 <= 3                                            # four moves in under a second: not four ticks
    d.run(ELSEWHERE, 1.0)
    before = ticks()[0]
    d.run(MATCH_AT, 0.2)
    assert ticks()[0] == before + 1


def test_a_press_makes_a_sound_and_a_buzz():
    d = at_mode()
    d.run(RALLY_AT, 1.5)
    assert "menu_select" in d.sounds() and "menu_select" in d.haptics()


def test_back_has_its_own_sound():
    d = at_mode()
    d.run(BACK_AT, 1.3)
    assert "menu_back" in d.sounds()


def test_the_music_plays_in_the_menus_stops_for_the_face_off_and_during_the_game():
    d = at_opponents()
    assert d.music()[-1] == "menu"
    d.run(OPP_AT[1], 1.5)
    assert d.music()[-1] is None and d.flow.screen == "VS"


# --- what the screens are told ------------------------------------------------------------------------------------------------------------------
def test_the_ui_state_tells_the_screen_where_it_is_how_long_and_what_the_hand_is_doing():
    d = at_mode()
    d.run(MATCH_AT, 0.6)
    ui = d.flow.ui_state(d.t, MATCH_AT)
    assert ui.screen == "MODE" and ui.hover == "match" and 0.3 < ui.progress < 0.7
    assert ui.cursor == MATCH_AT and ui.focus == 0 and ui.mode == "survival" and ui.level_tag == 1
    assert ui.t_s > 0.5


def test_a_new_screen_starts_its_clock_from_nothing_and_forgets_the_hold():
    d = at_mode()
    d.run(RALLY_AT, 1.25)                                          # pressed at 1.2 s
    assert d.flow.screen == "OPPONENT"
    ui = d.flow.ui_state(d.t, RALLY_AT)
    assert ui.t_s < 0.1 and ui.progress == 0.0
    d.run(RALLY_AT, 2.0)                                           # the hand stays where it was: nothing is pressed from it
    assert d.flow.screen == "OPPONENT" and d.flow.ui_state(d.t, RALLY_AT).progress == 0.0
