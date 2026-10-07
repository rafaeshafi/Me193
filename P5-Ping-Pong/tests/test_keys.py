import pytest

from pingpong import app, keys

SPACE, ESC = 32, 27


class Actuator:
    def __init__(self):
        self.calls = []
        self.core = self

    def disarm(self):
        self.calls.append("disarm")

    def arm(self):
        self.calls.append("arm")


def key(c):
    return ord(c)


def test_space_starts_a_game_from_the_lobby_and_after_game_over():
    session = app.make_session()
    action, _ = keys.handle_key(session, SPACE, fake=True)
    assert action is None and session.game.phase == "COUNTDOWN"


def test_space_in_fake_mode_swings_during_a_rally_and_does_nothing_in_live_mode():
    session = app.make_session()
    app.play_until_hits(session, 0)
    session.on_start()
    session.clock.advance_s(3.0)
    session.tick()
    assert session.game.phase == "RALLY"
    _, events = keys.handle_key(session, SPACE, fake=True)
    assert any(e.kind == "verdict" for e in events)           # the swing reached the judge
    session2 = app.make_session()
    session2.on_start()
    session2.clock.advance_s(3.0)
    session2.tick()
    _, events2 = keys.handle_key(session2, SPACE, fake=False)
    assert events2 == []                                      # live: swings come from the hub only


@pytest.mark.parametrize("c, name", [("1", "Rookie"), ("2", "Club"), ("3", "Pro")])
def test_digit_keys_pick_the_level_between_rallies(c, name):
    session = app.make_session()
    keys.handle_key(session, key(c), fake=False)
    assert session.game.level.name == name


def test_level_keys_are_ignored_mid_rally():
    session = app.make_session()
    session.on_start()
    session.clock.advance_s(3.0)
    session.tick()
    keys.handle_key(session, key("3"), fake=False)
    assert session.game.level.name == "Rookie"


def test_m_toggles_the_mode_only_when_allowed():
    session = app.make_session()
    keys.handle_key(session, key("m"), fake=False)
    assert session.game.mode == "match"
    keys.handle_key(session, key("m"), fake=False)
    assert session.game.mode == "survival"
    session.on_start()
    keys.handle_key(session, key("m"), fake=False)
    assert session.game.mode == "survival"


def test_x_toggles_the_xray_panel():
    session = app.make_session()
    keys.handle_key(session, key("x"), fake=False)
    assert session.xray is True
    keys.handle_key(session, key("x"), fake=False)
    assert session.xray is False


def test_d_disarms_and_rearms_the_motors_and_is_harmless_without_an_actuator():
    act = Actuator()
    session = app.make_session(actuator=act)
    keys.handle_key(session, key("d"), fake=False)
    keys.handle_key(session, key("d"), fake=False)
    assert act.calls == ["disarm", "arm"]
    keys.handle_key(app.make_session(), key("d"), fake=False)      # no actuator: no crash


@pytest.mark.parametrize("code", [ESC, key("q")])
def test_q_and_escape_quit(code):
    action, _ = keys.handle_key(app.make_session(), code, fake=False)
    assert action == "quit"


def test_soft_and_hard_fake_swings_differ_in_strength():
    assert keys.SWING_KEYS[key("j")] < keys.SWING_KEYS[SPACE] < keys.SWING_KEYS[key("k")]


def test_unknown_keys_do_nothing():
    action, events = keys.handle_key(app.make_session(), key("z"), fake=True)
    assert action is None and events == []
