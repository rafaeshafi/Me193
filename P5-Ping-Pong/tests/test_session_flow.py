"""The session with the way into a game (flow.py): the screens round the game follow the clock, the hand, the keys and the cards, and
carry out what the flow asks for: start a game with these choices, make this sound, buzz the hub, change the music."""

import pytest

from pingpong import app, flow, fonts, keys
from pingpong import hud as screen
from pingpong.events import PaddlePose, TagEvent
from pingpong.flow import Flow

S = 1_000_000_000
BOX = app.DEFAULT_BOX
ELSEWHERE, START_AT = (0.5, 0.9), (0.95, 0.9)
MATCH_AT, OPP2_AT = (0.5, 0.45), (0.5, 0.4)
SPACE = 32


class FakeAudio:
    muted = False

    def __init__(self):
        self.played, self.music = [], []

    def play(self, name):
        self.played.append(name)

    def play_music(self, name):
        self.music.append(name)

    def sound_for(self, event, level):
        return None


class FakeActuator:
    def __init__(self):
        self.submitted = []

    def submit(self, name, fire_at_ns=None):
        self.submitted.append(name)


def make(intro=False, **kw):
    session = app.make_session(flow=Flow(intro=intro, level_tag=kw.pop("level", 1), mode=kw.get("mode", "survival")), **kw)
    session.audio, session.actuator = FakeAudio(), FakeActuator()
    return session


def hold(session, ab, seconds, dt=1 / 30, conf=0.9):
    u, v = BOX.to_uv(*ab)
    for _ in range(round(seconds / dt)):
        session.clock.advance_s(dt)
        session.on_pose(PaddlePose(t_scene_ns=session.clock.now_ns(), u=u, v=v, conf=conf, hand="right"))
        session.tick()


def settle(session, seconds=0.4):
    hold(session, ELSEWHERE, seconds)


def hud(session):
    return session.hud_state()


def key(session, code):
    return keys.handle_key(session, code, fake=True)


# --- the screens follow the clock and the hand ---------------------------------------------------------------------------------------------
def test_without_a_flow_the_session_is_what_it_always_was():
    s = app.make_session()
    st = s.hud_state()
    assert s.flow is None and st.screen == "GAME" and st.ui is None and st.results is None


def test_with_a_flow_the_first_thing_on_screen_is_the_intro_and_it_runs_on_the_clock():
    s = make(intro=True)
    s.tick()
    st = hud(s)
    assert st.screen == "INTRO" and st.ui.t_s == pytest.approx(0.0, abs=0.05)
    hold(s, ELSEWHERE, 2.0)
    assert hud(s).ui.t_s == pytest.approx(2.0, abs=0.2) and hud(s).screen == "INTRO"
    hold(s, ELSEWHERE, flow.INTRO_S)
    assert hud(s).screen == "TITLE"


def test_the_pointer_carries_you_from_the_title_through_the_choices_to_a_started_game():
    s = make()
    settle(s)
    hold(s, START_AT, 1.8)
    assert hud(s).screen == "MODE"
    settle(s)
    hold(s, MATCH_AT, 1.5)
    assert hud(s).screen == "OPPONENT" and s.flow.mode == "match"
    settle(s)
    hold(s, OPP2_AT, 1.5)
    assert hud(s).screen == "VS" and s.game.phase == "LOBBY"
    hold(s, ELSEWHERE, flow.VS_S + 0.2)
    assert s.game.phase == "COUNTDOWN" and s.game.level.name == "Club" and s.game.mode == "match" and hud(s).screen == "GAME"


def test_the_screen_is_told_where_the_hand_points_and_how_far_the_hold_has_come():
    s = make()
    settle(s)
    hold(s, START_AT, 0.75)
    ui = hud(s).ui
    assert ui.cursor == pytest.approx(START_AT, abs=1e-6) and ui.hover == "start" and 0.3 < ui.progress < 0.7


def test_a_hand_that_is_not_seen_is_not_pointing_at_anything():
    s = make()
    settle(s)
    for _ in range(60):
        s.clock.advance_s(1 / 30)
        s.tick()
    assert hud(s).ui.cursor is None and hud(s).ui.hover is None


# --- keys and cards --------------------------------------------------------------------------------------------------------------------------
def test_space_on_the_title_starts_a_game_at_once_with_the_command_line_choices():
    s = make(level=3, mode="match")
    settle(s)
    key(s, SPACE)
    assert s.game.phase == "COUNTDOWN" and s.game.level.name == "Pro" and s.game.mode == "match"


def test_a_level_card_picks_the_opponent_and_starts_the_game_with_five_seconds_for_putting_the_card_down():
    s = make()
    settle(s)
    s.on_tag(TagEvent(role="LEVEL", value=2, t_ns=0))
    assert s.game.phase == "COUNTDOWN" and s.game.level.name == "Club"
    assert s.game.seconds_to_serve(s.clock.now_ns()) == 5.0 and hud(s).countdown == 5


def test_the_start_card_starts_the_game_with_the_choices_so_far_and_the_same_five_seconds():
    s = make(level=3, mode="match")
    settle(s)
    s.on_tag(TagEvent(role="START", value=0, t_ns=0))
    assert s.game.phase == "COUNTDOWN" and s.game.level.name == "Pro" and s.game.mode == "match"
    assert s.game.seconds_to_serve(s.clock.now_ns()) == 5.0


def test_a_start_by_key_is_the_usual_three_seconds():
    s = make()
    settle(s)
    key(s, SPACE)
    assert s.game.seconds_to_serve(s.clock.now_ns()) == 3.0


def test_the_screen_is_told_how_far_a_card_has_been_held_but_only_while_the_lobby_is_up():
    s = make()
    settle(s)
    assert hud(s).tag_hold is None                                       # nothing bound: nothing shown
    s.bind_status(card=lambda now_ns: (2, 0.4))
    assert hud(s).tag_hold == (2, 0.4)
    key(s, SPACE)
    assert s.game.phase == "COUNTDOWN" and hud(s).tag_hold is None       # the game is on: the ring is gone


def test_quitting_is_still_quitting_in_the_menus():
    s = make()
    assert key(s, ord("q"))[0] == "quit" and key(s, 27)[0] == "quit"


def test_while_a_game_is_on_the_old_keys_and_cards_are_the_games_own():
    s = make()
    settle(s)
    key(s, SPACE)
    assert s.game.phase == "COUNTDOWN"
    key(s, ord("x"))
    assert s.xray is True
    s.on_tag(TagEvent(role="LEVEL", value=3, t_ns=0))
    assert s.game.level.name == "Rookie"                         # a level card mid-game: ignored, as ever


def test_a_game_that_cannot_start_sends_you_back_to_the_title():
    s = make()
    settle(s)
    s.game.set_pause("hub", True, s.clock.now_ns())
    key(s, SPACE)
    assert s.game.phase == "LOBBY" and hud(s).screen == "TITLE"


# --- the results ---------------------------------------------------------------------------------------------------------------------------
def play_to_the_end(s):
    key(s, SPACE)
    while s.game.phase != "MATCH_OVER":                          # nobody swings: the first ball is a miss and a rally ends
        hold(s, ELSEWHERE, 0.1)
        assert s.clock.now_ns() < 90 * S


def test_when_a_game_ends_the_results_come_up_with_its_words_and_numbers():
    s = make()
    settle(s)
    play_to_the_end(s)
    hold(s, ELSEWHERE, 0.5)
    st = hud(s)
    assert st.screen == "RESULTS" and st.results.title == "GAME OVER" and st.results.won is None
    assert dict(st.results.stats)["HITS"] == "0" and [label for label, _ in st.results.stats][-1] == "TIME"


def drawn_text(state, monkeypatch):
    """Everything the screen writes for this state, as (text, size)."""
    drawn, real = [], fonts.draw

    def spy(frame, text, x, y, size, *a, **kw):
        drawn.append((text, round(size)))
        return real(frame, text, x, y, size, *a, **kw)

    monkeypatch.setattr(fonts, "draw", spy)
    screen.render(state, size=(1280, 720))
    return drawn


def rally_of_three_then_a_miss(s):
    app.play_until_hits(s, 3)                                    # three returns in a row ...
    while s.game.phase != "MATCH_OVER":                          # ... and then nobody swings: the next ball is the miss that ends it
        hold(s, ELSEWHERE, 0.1)
        assert s.clock.now_ns() < 90 * S


def test_the_results_of_a_rally_show_the_streak_it_ended_on_not_the_zero_the_last_miss_left(monkeypatch):
    s = make()
    settle(s)
    key(s, SPACE)
    rally_of_three_then_a_miss(s)
    hold(s, ELSEWHERE, 1.5)
    st = hud(s)
    assert st.screen == "RESULTS" and dict(st.results.stats)["LONGEST RALLY"] == "3"
    score = max((t for t in drawn_text(st, monkeypatch) if t[0].isdigit()), key=lambda t: t[1])         # the big number
    assert score[0] == "3" and st.streak == 3


def test_the_game_over_panel_without_menus_shows_the_streak_it_ended_on_too(monkeypatch):
    s = app.make_session()
    rally_of_three_then_a_miss(s)
    assert any(t[0] == "streak 3    best 3" for t in drawn_text(s.hud_state(), monkeypatch))


def test_a_lost_match_says_so_and_the_opponent_cheers():
    s = make(mode="match", target=1)
    settle(s)
    play_to_the_end(s)
    hold(s, ELSEWHERE, 0.5)
    st = hud(s)
    assert st.results.won is False and st.results.title == "THE CPU WINS" and st.cpu_mood == "cheer"


def test_play_again_from_the_results_starts_the_same_game_with_the_keys_the_hand_or_the_card():
    for how in ("key", "hand", "card"):
        s = make(level=2)
        settle(s)
        play_to_the_end(s)
        hold(s, ELSEWHERE, 0.6)
        assert hud(s).screen == "RESULTS"
        if how == "key":
            key(s, SPACE)
        elif how == "card":
            s.on_tag(TagEvent(role="START", value=0, t_ns=0))
        else:
            hold(s, START_AT, 1.8)
        assert s.game.phase == "COUNTDOWN" and s.game.level.name == "Club", how
        assert hud(s).results is None and hud(s).screen == "GAME"


def test_the_opponent_choice_screen_is_reachable_again_from_the_results():
    s = make()
    settle(s)
    play_to_the_end(s)
    hold(s, ELSEWHERE, 0.6)
    key(s, 8)
    assert hud(s).screen == "OPPONENT" and s.game.phase == "MATCH_OVER"
    s.flow.level_tag = 3
    key(s, SPACE)
    assert s.game.phase == "COUNTDOWN" and s.game.level.name == "Pro"


# --- the faces and the countdown -------------------------------------------------------------------------------------------------------------
def test_the_opponents_face_follows_the_score_and_goes_back_to_normal():
    s = make(mode="match", target=7)
    settle(s)
    key(s, SPACE)
    assert hud(s).cpu_mood == "happy"
    while s.game.phase != "POINT_OVER":                          # the first ball is missed: the computer scores
        hold(s, ELSEWHERE, 0.1)
        assert s.clock.now_ns() < 60 * S
    assert hud(s).cpu_mood == "cheer" and hud(s).point_for == "cpu"
    hold(s, ELSEWHERE, 3.0)
    assert hud(s).cpu_mood == "happy" or s.game.phase != "POINT_OVER"


def test_the_countdown_digit_knows_how_long_it_has_been_on_screen():
    s = make()
    settle(s)
    key(s, SPACE)
    seen = []
    for _ in range(80):
        hold(s, ELSEWHERE, 1 / 30)
        st = hud(s)
        if st.countdown is not None:
            seen.append((st.countdown, st.countdown_t))
    assert {digit for digit, _ in seen} == {3, 2, 1}
    assert all(0.0 <= t <= 1.0 for _, t in seen)
    assert min(t for digit, t in seen if digit == 2) < 0.1 and max(t for digit, t in seen if digit == 2) > 0.8


def test_the_animation_clock_runs_with_the_game_clock():
    s = make()
    settle(s, 0.2)
    first = hud(s).anim_t
    hold(s, ELSEWHERE, 1.5)
    assert hud(s).anim_t == pytest.approx(first + 1.5, abs=0.1)


# --- the senses of it --------------------------------------------------------------------------------------------------------------------------
def test_the_sounds_the_flow_asks_for_are_played_and_the_music_follows_the_screens():
    s = make(intro=True)
    hold(s, ELSEWHERE, 0.3)
    assert s.audio.music[0] == "intro"
    key(s, SPACE)                                                  # any key skips the intro
    hold(s, ELSEWHERE, 0.3)
    assert s.audio.music[-1] == "menu"
    hold(s, START_AT, 1.8)
    assert "menu_select" in s.audio.played and "menu_tick" in s.audio.played
    key(s, 13)
    key(s, 13)
    key(s, 13)                                                     # MODE -> OPPONENT -> VS
    assert hud(s).screen == "VS" and s.audio.music[-1] is None and "vs" in s.audio.played


def test_the_hub_buzzes_when_the_hand_comes_onto_a_button_and_when_it_presses_one():
    s = make()
    settle(s)
    hold(s, START_AT, 1.8)
    assert "menu_tick" in s.actuator.submitted and "menu_select" in s.actuator.submitted


def test_a_session_without_sound_or_a_hub_goes_through_the_menus_just_the_same():
    s = app.make_session(flow=Flow(intro=False))
    settle(s)
    hold(s, START_AT, 1.8)
    assert hud(s).screen == "MODE"


def test_go_flashes_on_the_screen_when_the_first_ball_is_served_and_not_on_every_ball():
    s = make()
    settle(s)
    key(s, SPACE)
    seen = []
    for _ in range(int(7 / (1 / 30))):
        hold(s, ELSEWHERE, 1 / 30)
        seen.append((s.game.phase, hud(s).message))
    went = [message for phase, message in seen if phase == "RALLY"]
    assert "GO!" in went[:30]                                      # in the first second of the rally
    assert "GO!" not in [message for phase, message in seen if phase == "COUNTDOWN"]
    assert all(message != "GO!" for message in went[45:])           # and it has gone again
