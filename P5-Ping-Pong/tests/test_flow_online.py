"""Playing a friend through the flow: the list of games, hosting and joining, the face-off with a friend, the rematch and leaving."""

import pytest

from pingpong import flow as flowmod
from pingpong import menu_layout
from pingpong.flow import Action
from pingpong.uistate import OnlineView
from tests.flow_harness import (BACK, BACK_AT, D, ELSEWHERE, ENTER, LEFT_TOP, ONLINE_AT, ROOM, ROOMS, SPACE, START_AT, Drive, at_mode, at_target)


# --- playing a friend ---------------------------------------------------------------------------------------------------------------------------
def at_online(**kw):
    d = at_mode(**kw)
    d.run(ONLINE_AT, 1.5)
    d.flow.set_online(OnlineView(status="browsing"))
    return d.on("ONLINE")


def lobby(d, *rooms, **view):
    d.flow.set_online(OnlineView(status="browsing", rooms=tuple(rooms), **view))
    return d


def online_actions(d):
    return [a.value for _, a in d.log if a.kind == "online"]


def test_the_choice_of_game_has_a_third_card_for_playing_a_friend():
    d = at_mode()
    d.run(ONLINE_AT, 1.1)
    assert d.flow.screen == "MODE"
    d.run(ONLINE_AT, 0.4)
    assert d.flow.screen == "ONLINE" and d.started() == [] and online_actions(d) == [("open", None)]


def test_the_keys_reach_the_online_card_too():
    d = at_mode()
    d.key(D)
    d.key(D)
    d.key(ENTER)
    assert d.flow.screen == "ONLINE" and online_actions(d) == [("open", None)]


@pytest.mark.parametrize("pace", [1, 2, 3])
def test_holding_the_hub_on_a_pace_hosts_a_game_at_that_pace_and_waits_for_a_friend(pace):
    d = at_online()
    d.run(at_target(f"host{pace}", "ONLINE"), 1.5)
    assert d.flow.screen == "WAIT" and online_actions(d)[-1] == ("host", pace)
    assert d.started() == []


def test_holding_the_hub_on_an_open_game_joins_it():
    d = lobby(at_online(), *ROOMS[:3])
    d.run(at_target("join1", "ONLINE", 3), 1.5)
    assert d.flow.screen == "WAIT" and online_actions(d)[-1] == ("join", "FGHJK")


def test_only_the_games_that_are_on_the_list_can_be_pointed_at_and_only_three_of_them():
    assert [n for n, _, _ in menu_layout.targets("ONLINE", 0) if n.startswith("join")] == []
    assert [n for n, _, _ in menu_layout.targets("ONLINE", 2) if n.startswith("join")] == ["join0", "join1"]
    assert [n for n, _, _ in menu_layout.targets("ONLINE", 9) if n.startswith("join")] == ["join0", "join1", "join2"]
    d = at_online()                                           # no games: pointing where the first would be does nothing
    d.run(at_target("join0", "ONLINE", 1), 2.0)
    assert d.flow.screen == "ONLINE" and ("join", "ABCDE") not in online_actions(d)


def test_the_keys_host_with_the_chosen_pace_or_join_the_game_in_focus():
    d = at_online()
    d.key(ord("3"))
    d.key(ENTER)
    assert d.flow.screen == "WAIT" and online_actions(d)[-1] == ("host", 3)
    e = lobby(at_online(), *ROOMS[:2])
    e.key(D)
    assert e.flow.focus == 1
    e.key(D)
    e.key(D)
    assert e.flow.focus == 2                                  # the host card, then the two games
    e.key(ENTER)
    assert e.flow.screen == "WAIT" and online_actions(e)[-1] == ("join", "FGHJK")


def test_the_online_screen_does_not_start_a_game_on_space_or_the_start_card():
    d = at_online()
    assert d.key(SPACE) == [] and d.tag("START") == [] and d.flow.screen == "ONLINE" and d.started() == []


def test_back_closes_the_lobby_and_returns_to_the_choice_of_game():
    for go in (lambda d: d.run(BACK_AT, 1.3), lambda d: d.key(BACK)):
        d = at_online()
        go(d)
        assert d.flow.screen == "MODE" and online_actions(d)[-1] == ("close", None)


def test_while_waiting_cancel_goes_back_to_the_list():
    for go in (lambda d: d.run(BACK_AT, 1.3), lambda d: d.key(BACK)):
        d = at_online()
        d.key(ENTER)
        assert d.flow.screen == "WAIT"
        d.run(ELSEWHERE, 0.3)
        go(d)
        assert d.flow.screen == "ONLINE" and online_actions(d)[-1] == ("cancel", None)


def test_a_try_that_comes_to_nothing_returns_to_the_list():
    d = at_online()
    d.key(ENTER)
    d.flow.net_failed(d.t)
    assert d.flow.screen == "ONLINE"
    e = at_online()
    e.flow.net_failed(e.t)                                    # (not waiting: nothing to go back from)
    assert e.flow.screen == "ONLINE"


def paired(d, name="MAYA", pace=2):
    d.key(ENTER)
    d.flow.paired(name, pace, d.t)
    return d


def test_a_pair_goes_to_the_face_off_with_the_friend_and_then_starts_a_match_at_the_hosts_pace():
    d = paired(at_online(), "MAYA", pace=3)
    assert d.flow.screen == "VS" and d.flow.level_tag == 3 and d.flow.mode == "match"
    assert d.flow.ui_state(d.t, None).opponent == "MAYA" and d.started() == []
    d.run(ELSEWHERE, flowmod.VS_S + 0.2)
    assert d.started() == [Action("start", (3, "match"))] and d.flow.screen == "GAME"


def test_neither_space_nor_delete_cuts_a_face_off_with_a_friend_short():
    d = paired(at_online())
    assert d.key(SPACE) == [] and d.key(BACK) == [] and d.flow.screen == "VS"
    assert d.tag("START") == []


def online_results(d):
    d = paired(d)
    d.run(ELSEWHERE, flowmod.VS_S + 0.2)
    d.run(ELSEWHERE, 0.5, phase="MATCH_OVER")
    assert d.flow.screen == "RESULTS"
    return d


def test_after_an_online_match_play_again_is_a_rematch_that_waits_for_the_friend_and_the_other_button_leaves():
    d = online_results(at_online())
    d.run(START_AT, 2.5, phase="MATCH_OVER")
    assert d.flow.screen == "RESULTS" and online_actions(d)[-1] == ("rematch", None) and d.flow.rematch_pending
    assert d.flow.ui_state(d.t, None).rematch_pending and len([a for a in d.started()]) == 1
    d.log.clear()
    actions = d.flow.start_rematch(d.t)
    assert Action("start", (2, "match")) in actions and d.flow.screen == "GAME" and not d.flow.rematch_pending
    e = online_results(at_online())
    e.run(LEFT_TOP, 1.5, phase="MATCH_OVER")
    assert e.flow.screen == "ONLINE" and online_actions(e)[-1] == ("leave", None)
    assert e.flow.ui_state(e.t, None).opponent == "" and not e.flow.online_game


def test_space_enter_and_the_start_card_ask_for_the_rematch_and_delete_leaves():
    for press in (lambda d: d.key(SPACE), lambda d: d.key(ENTER), lambda d: d.tag("START")):
        d = online_results(at_online())
        press(d)
        assert online_actions(d)[-1] == ("rematch", None)
    d = online_results(at_online())
    d.key(BACK)
    assert d.flow.screen == "ONLINE" and online_actions(d)[-1] == ("leave", None)


def test_a_rematch_is_asked_for_once_however_often_it_is_pressed():
    d = online_results(at_online())
    d.key(SPACE)
    d.key(SPACE)
    assert [v for v in online_actions(d) if v == ("rematch", None)] == [("rematch", None)]


def test_when_the_friend_has_left_there_is_no_rematch_only_leaving():
    d = online_results(at_online())
    d.flow.opponent_left(True)
    assert d.flow.ui_state(d.t, None).opponent_gone
    assert [n for n, _, _ in menu_layout.targets("RESULTS", 0, can_again=False)] == ["change"]
    d.run(START_AT, 2.5, phase="MATCH_OVER")
    assert online_actions(d)[-1] != ("rematch", None) and d.flow.screen == "RESULTS"
    assert d.key(SPACE) == [] and d.tag("START") == []
    d.run(LEFT_TOP, 1.5, phase="MATCH_OVER")
    assert d.flow.screen == "ONLINE"


def test_a_single_player_game_after_an_online_one_is_just_what_it_was():
    d = online_results(at_online())
    d.key(BACK)
    d.key(BACK)                                               # the lobby, then the choice of game (the match is in focus, as chosen)
    assert d.flow.screen == "MODE" and d.flow.focus == 1
    d.key(ENTER)                                              # a match against the computer
    assert d.flow.screen == "OPPONENT" and not d.flow.online_game and d.flow.mode == "match"


def test_the_online_screens_play_the_menu_music():
    d = at_online()
    assert d.music()[-1] == "menu"
    d.key(ENTER)
    assert d.flow.screen == "WAIT" and d.music()[-1] == "menu"


def test_the_screen_is_told_the_online_view_and_the_pace_in_focus():
    d = lobby(at_online(), ROOM, message="THAT GAME IS FULL")
    ui = d.flow.ui_state(d.t, None)
    assert ui.screen == "ONLINE" and ui.online.rooms == (ROOM,) and ui.online.message == "THAT GAME IS FULL" and ui.level_tag == 1


# --- starting straight at a friend ------------------------------------------------------------------------------------------------------------------
def test_a_flow_can_start_at_the_list_of_games_without_the_intro_or_the_menus():
    d = Drive(start=("online",))
    d.run(ELSEWHERE, 0.3)
    assert d.flow.screen == "ONLINE" and online_actions(d) == [("open", None)] and d.music()[-1] == "menu"


def test_a_flow_can_start_by_hosting_a_game_at_a_pace():
    d = Drive(start=("host", 3))
    d.run(ELSEWHERE, 0.3)
    assert d.flow.screen == "WAIT" and online_actions(d) == [("open", None), ("host", 3)] and d.flow.level_tag == 3
    d.run(ELSEWHERE, 0.3)
    assert online_actions(d) == [("open", None), ("host", 3)]                    # said once


def test_a_flow_can_start_by_joining_a_game_by_its_code_and_a_failure_lands_on_the_list():
    d = Drive(start=("join", "ABCDE"))
    d.run(ELSEWHERE, 0.3)
    assert d.flow.screen == "WAIT" and online_actions(d) == [("open", None), ("join", "ABCDE")]
    d.flow.net_failed(d.t)
    assert d.flow.screen == "ONLINE"
