"""A session that can play a friend: the online screens, the pairing, a match across the (in-memory) network, and putting the
session back as it was.  Two sessions, each with its flow and its scripted hand, share a clock and a network."""

import pytest

import config
from pingpong import app
from pingpong.sources_fake import FakeMqttClient
from tests.online_harness import Laptop, Pair, center

S = 1_000_000_000


def paired_pair(**kw):
    """Both laptops at the lobby; the first hosts at Club pace, the second joins; run to the face-off."""
    pair = Pair(**kw).both_to("ONLINE")
    pair.start_hosting(2)
    pair.join_first()
    assert pair.run(4, until=lambda p: p.a.screen == p.b.screen == "VS")
    return pair


def results(pair):
    assert pair.run(90, until=lambda p: p.a.screen == p.b.screen == "RESULTS")
    return pair.a.session.hud_state(), pair.b.session.hud_state()


# --- finding each other ------------------------------------------------------------------------------------------------------------------------
def test_the_online_card_opens_the_list_and_a_game_one_player_opens_shows_up_for_the_other():
    pair = Pair().both_to("ONLINE")
    assert pair.a.screen == pair.b.screen == "ONLINE"
    assert pair.a.online.state == pair.b.online.state == "browsing"
    pair.start_hosting(2)
    assert pair.a.screen == "WAIT" and pair.a.online.state == "hosting"
    rooms = pair.b.session.hud_state().ui.online.rooms
    assert [(r["host"], r["pace"]) for r in rooms] == [("MAYA", 2)]
    assert pair.a.session.hud_state().ui.online.code == rooms[0]["code"]


def test_pointing_at_the_game_pairs_them_and_both_see_the_face_off_with_each_other_at_the_hosts_pace():
    pair = paired_pair(target=3, guest_target=5)
    a, b = pair.a.session.hud_state(), pair.b.session.hud_state()
    assert a.ui.opponent == "RAFAE" and b.ui.opponent == "MAYA"
    assert a.ui.level_tag == b.ui.level_tag == 2 and a.level_name == b.level_name == "Club"
    assert pair.a.game.remote is not None and pair.b.game.remote is not None
    assert a.target == b.target == 3 and a.mode == b.mode == "match"                      # the host's match, not the guest's own settings


def test_after_the_face_off_both_games_begin():
    pair = paired_pair()
    assert pair.run(4, until=lambda p: p.a.game.phase != "LOBBY" and p.b.game.phase != "LOBBY")
    assert pair.a.screen == pair.b.screen == "GAME"


# --- a match across the network ------------------------------------------------------------------------------------------------------------------------
def test_a_match_is_played_to_its_end_with_the_same_score_on_both_and_the_results_say_who_won():
    pair = paired_pair(host_skill="perfect", guest_skill="idle", target=3)
    a, b = results(pair)
    assert (pair.a.game.player_points, pair.a.game.cpu_points) == (3, 0) == (pair.b.game.cpu_points, pair.b.game.player_points)
    assert a.results.won is True and a.results.title == "YOU WIN!" and b.results.won is False and b.results.title == "GOOD GAME!"
    assert a.opponent_name == "RAFAE" and b.opponent_name == "MAYA"


def test_a_friends_games_are_not_put_on_the_leaderboard():
    pair = paired_pair(host_skill="perfect", guest_skill="idle", target=2)
    saved = []
    pair.a.session.on_game_over = pair.b.session.on_game_over = saved.append
    results(pair)
    assert saved == []


def test_the_friend_stands_behind_the_table_with_the_paddle_where_their_hand_is():
    pair = paired_pair(host_skill="perfect", guest_skill="perfect", target=7)
    assert pair.run(30, until=lambda p: p.a.game.phase == "RALLY" and p.a.game.tracker.streak >= 2)
    state = pair.a.session.hud_state()
    assert state.opponent_name == "RAFAE" and state.cpu_x_m == pytest.approx(pair.a.game.remote.paddle_x)
    assert state.ping_ms is not None and 40 < state.ping_ms < 200                       # a 30 ms cable each way
    assert pair.b.session.hud_state().opponent_name == "MAYA"


# --- afterwards --------------------------------------------------------------------------------------------------------------------------------------------
def test_a_rematch_starts_only_when_both_ask_and_begins_at_nothing_all():
    pair = paired_pair(host_skill="perfect", guest_skill="idle", target=2)
    results(pair)
    pair.settle(0.6)
    pair.point(pair.a, "again", 2.0)
    assert pair.a.screen == "RESULTS" and pair.a.flow.rematch_pending and pair.a.game.phase == "MATCH_OVER"
    assert pair.a.session.hud_state().ui.rematch_pending
    pair.point(pair.b, "again", 2.0)
    assert pair.run(4, until=lambda p: p.a.screen == p.b.screen == "GAME")
    assert pair.a.game.phase in ("COUNTDOWN", "RALLY") and pair.b.game.phase in ("COUNTDOWN", "RALLY")
    assert (pair.a.game.player_points, pair.a.game.cpu_points) == (0, 0) and (pair.b.game.player_points, pair.b.game.cpu_points) == (0, 0)
    assert not pair.a.flow.rematch_pending and not pair.b.flow.rematch_pending
    a, b = results(pair)                                                              # and the second match is played out as well
    assert (a.player_points, b.cpu_points) == (2, 2)


def test_leaving_says_goodbye_so_the_other_sees_they_are_gone_and_can_only_leave():
    pair = paired_pair(host_skill="perfect", guest_skill="idle", target=2)
    results(pair)
    pair.settle(0.6)
    pair.point(pair.b, "change", 1.6)
    assert pair.b.screen == "ONLINE" and pair.b.game.remote is None and pair.b.online.state == "browsing"
    assert pair.run(2, until=lambda p: p.a.flow.opponent_gone)
    ui = pair.a.session.hud_state().ui
    assert ui.opponent_gone and pair.a.screen == "RESULTS"
    pair.point(pair.a, center("again", "RESULTS"), 2.0)                                # (where the button was: there is none any more)
    assert pair.a.screen == "RESULTS"
    pair.point(pair.a, "change", 1.6)
    assert pair.a.screen == "ONLINE" and pair.a.game.remote is None


def test_a_friend_who_leaves_in_the_middle_ends_the_match_for_the_one_who_stays_with_no_winner_named():
    pair = paired_pair(host_skill="perfect", guest_skill="perfect", target=9)
    assert pair.run(30, until=lambda p: p.a.game.phase == "RALLY" and p.a.game.tracker.streak >= 2)
    pair.b.session.close_online()                                                       # the window is closed
    assert pair.run(3, until=lambda p: p.a.screen == "RESULTS")
    state = pair.a.session.hud_state()
    assert state.results.title == "THEY LEFT" and state.results.won is None and state.ui.opponent_gone
    assert pair.a.game.phase == "MATCH_OVER" and not pair.a.game.paused


def test_after_a_friends_game_the_session_plays_the_computer_as_before_with_its_own_record():
    client = FakeMqttClient()
    pair = Pair(host_skill="perfect", guest_skill="idle", target=2)
    pair.a.session.game.publisher = None
    pair.a.session.game.tracker.record = 4                                              # a record from earlier games
    pair.both_to("ONLINE")
    pair.start_hosting(2)
    pair.join_first()
    results(pair)
    game = pair.a.game
    assert game.remote is not None and game.tracker.record <= 2 and game.target_points == 2
    pair.settle(0.6)
    pair.point(pair.a, "change", 1.6)
    assert pair.a.screen == "ONLINE" and game.remote is None
    assert game.tracker.record == 4 and game.tracker.streak == 0
    assert not client.published


def test_nothing_a_friend_plays_ever_reaches_the_official_score_topic_but_single_player_still_does():
    client = FakeMqttClient()
    pair = Pair(host_skill="perfect", guest_skill="perfect", target=2)
    host = Laptop(pair.clock, pair.net, "MAYA", skill="perfect", seed=1, target=2, level=2, client=client, source="live")
    pair.a = host
    assert host.game.publisher is not None
    host.game.publisher.update(5)                                                       # an earlier game against the computer
    before = list(client.published)
    assert before and all(p["topic"] == config.SCORE_TOPIC for p in before)
    pair.both_to("ONLINE")
    pair.start_hosting(2)
    pair.join_first()
    assert pair.run(4, until=lambda p: p.a.screen == "VS")
    assert host.game.publisher is None
    assert pair.run(40, until=lambda p: p.a.game.tracker.streak >= 6)                   # a long rally between two good players...
    pair.b.player.skill = "idle"                                                        # ... and then the match is finished off
    results(pair)
    assert client.published == before                                                   # not one message more, whatever the rallies
    pair.settle(0.6)
    pair.point(pair.a, "change", 1.6)
    assert host.game.publisher is not None and host.game.remote is None
    host.game.publisher.update(6)
    assert client.published[-1]["payload"] == "6.0"


# --- the keys and the cards do not undo what two people agreed -------------------------------------------------------------------------------------------------------
def test_keys_and_cards_do_not_restart_or_retune_a_friends_game():
    from pingpong import keys
    from pingpong.events import TagEvent

    pair = paired_pair(host_skill="perfect", guest_skill="idle", target=1)
    assert pair.run(30, until=lambda p: p.a.game.phase == "MATCH_OVER")
    now = pair.clock.now_ns()
    pair.a.flow.screen = "GAME"                                                         # (the instant before the results come up)
    keys.handle_key(pair.a.session, 32, fake=False)
    pair.a.session.on_tag(TagEvent(role="START", value=0, t_ns=now))
    pair.a.session.on_tag(TagEvent(role="LEVEL", value=3, t_ns=now))
    keys.handle_key(pair.a.session, ord("3"), fake=False)
    keys.handle_key(pair.a.session, ord("m"), fake=False)
    assert pair.a.game.phase == "MATCH_OVER" and pair.a.game.level.tag == 2 and pair.a.game.mode == "match"


def test_a_session_that_cannot_play_a_friend_says_so_instead_of_hanging():
    session = app.make_session(flow=app.Flow(intro=False))
    session.apply([("online", ("open", None))])
    assert session.flow.online.status == "offline" and "NOT SET UP" in session.flow.online.message
    session.apply([("online", ("host", 1))])                                           # nothing to host with: back to the list
