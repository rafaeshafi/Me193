"""The session starts a game when the hub is held on the START button, and says what the screen should draw."""

import pytest

from pingpong import app
from pingpong.events import PaddlePose

S = 1_000_000_000
BOX = app.DEFAULT_BOX                                              # u -1..1, v -0.5..0.5


def uv(a, b):
    return BOX.to_uv(a, b)


def hold_hand(session, ab, seconds, dt=1 / 30, conf=0.9):
    """The hand at (a, b) of the reach box for `seconds`, one reading per frame."""
    u, v = uv(*ab)
    for _ in range(round(seconds / dt)):
        session.clock.advance_s(dt)
        session.on_pose(PaddlePose(t_scene_ns=session.clock.now_ns(), u=u, v=v, conf=conf, hand="right"))
        session.tick()


def make(**kw):
    return app.make_session(hold_start=True, **kw)


def test_holding_the_hub_in_the_top_right_of_the_reach_starts_the_countdown_without_a_key():
    s = make()
    hold_hand(s, (0.5, 0.5), 0.4)
    assert s.game.phase == "LOBBY"
    hold_hand(s, (0.95, 0.9), 1.4)
    assert s.game.phase == "LOBBY"
    hold_hand(s, (0.95, 0.9), 0.3)
    assert s.game.phase == "COUNTDOWN"


def test_a_hand_anywhere_else_never_starts_it():
    s = make()
    for ab in ((0.5, 0.5), (0.1, 0.9), (0.95, 0.2), (0.5, 0.95)):
        hold_hand(s, ab, 3.0)
    assert s.game.phase == "LOBBY"


def test_a_reading_the_camera_is_not_sure_of_does_not_count():
    s = make()
    hold_hand(s, (0.5, 0.5), 0.3)
    hold_hand(s, (0.95, 0.9), 3.0, conf=0.3)
    assert s.game.phase == "LOBBY"


def test_a_hand_that_was_lost_stops_the_hold_instead_of_finishing_it_from_an_old_reading():
    s = make()
    hold_hand(s, (0.5, 0.5), 0.3)
    hold_hand(s, (0.95, 0.9), 1.0)
    for _ in range(90):                                            # 1.5 s with no new reading at all
        s.clock.advance_s(1 / 60)
        s.tick()
    assert s.game.phase == "LOBBY" and s.hud_state().start_button[0] == 0.0


def test_on_the_end_screen_the_same_hold_plays_again_but_only_after_leaving_the_corner_once():
    s = make()
    hold_hand(s, (0.5, 0.5), 0.3)
    hold_hand(s, (0.95, 0.9), 2.0)
    assert s.game.phase == "COUNTDOWN"
    while s.game.phase != "MATCH_OVER":                            # nobody swings: the first ball is a miss and survival ends
        hold_hand(s, (0.95, 0.9), 0.1)
        assert s.clock.now_ns() < 60 * S
    hold_hand(s, (0.95, 0.9), 3.0)
    assert s.game.phase == "MATCH_OVER"                            # the hand never left the corner: no restart
    hold_hand(s, (0.4, 0.5), 0.3)
    hold_hand(s, (0.95, 0.9), 1.8)
    assert s.game.phase == "COUNTDOWN"


def test_it_is_off_unless_asked_for():
    s = app.make_session()
    hold_hand(s, (0.5, 0.5), 0.3)
    hold_hand(s, (0.95, 0.9), 3.0)
    assert s.game.phase == "LOBBY" and s.hold_start is None
    st = s.hud_state()
    assert st.start_button is None and st.cursor is None


def test_the_screen_is_told_how_far_the_hold_has_come_and_where_the_hand_points():
    s = make()
    hold_hand(s, (0.5, 0.5), 0.3)
    st = s.hud_state()
    assert st.start_button == (0.0, False) and st.cursor == pytest.approx((0.5, 0.5))
    hold_hand(s, (0.95, 0.9), 0.75)
    st = s.hud_state()
    assert st.start_button[0] == pytest.approx(0.5, abs=0.06) and st.start_button[1] is True
    assert st.cursor == pytest.approx((0.95, 0.9))
    hold_hand(s, (0.95, 0.9), 1.2)
    st = s.hud_state()
    assert s.game.phase == "COUNTDOWN" and st.start_button is None and st.cursor is None      # during a game: nothing


def test_the_keyboard_and_the_card_still_start_the_game():
    s = make()
    s.on_start()
    assert s.game.phase == "COUNTDOWN"
