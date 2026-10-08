"""Finding a friend to play: the list of open games, hosting one, joining one, and the handshake that makes two laptops a pair."""

import pytest

from pingpong import netproto as proto
from pingpong import online
from pingpong.clock import FakeClock
from pingpong.loopnet import LoopNet

S = 1_000_000_000


class Rig:
    """A network and some players on one fake clock."""

    def __init__(self, **link_kw):
        self.clock = FakeClock(start_ns=1_000_000_000)
        self.net = LoopNet(self.clock, **link_kw)
        self.events = {}

    def player(self, name, seed=1):
        player = online.Online(self.net, name=name, seed=seed)
        self.events[player.name] = []
        return player

    def run(self, seconds, *players, dt=0.02):
        for _ in range(round(seconds / dt)):
            self.clock.advance_s(dt)
            for player in players:
                self.events.setdefault(player.name, []).extend(player.step(self.clock.now_ns()))

    def of(self, player, kind):
        return [e for e in self.events[player.name] if isinstance(e, kind)]


def paired(rig, pace=2, target=7):
    host, guest = rig.player("MAYA", 1), rig.player("RAFAE", 2)
    guest.open()
    code = host.host(pace, target)
    rig.run(0.5, host, guest)
    guest.join(guest.rooms()[0]["code"])
    rig.run(1.0, host, guest)
    return host, guest, code


# --- the list ----------------------------------------------------------------------------------------------------------------------------------
def test_a_game_somebody_hosts_is_on_everyone_elses_list_and_not_on_their_own():
    rig = Rig()
    host, guest = rig.player("maya"), rig.player("RAFAE", 2)
    guest.open()
    code = host.host(2, 7)
    rig.run(0.5, host, guest)
    assert [(r["code"], r["host"], r["pace"], r["target"]) for r in guest.rooms()] == [(code, "MAYA", 2, 7)]
    assert host.rooms() == [] and host.state == "hosting" and guest.state == "browsing"
    assert proto.clean_code(code) == code


def test_a_game_is_not_put_on_the_list_until_its_room_is_listening():
    rig = Rig()
    rig.net.up = False
    host = rig.player("MAYA")
    host.host(1, 7)
    rig.run(1.0, host)
    assert rig.net.entries == {}
    rig.net.up = True
    rig.run(0.2, host)
    assert len(rig.net.entries) == 1


def test_two_hosts_never_get_the_same_code_and_never_one_that_is_on_the_list():
    rig = Rig()
    codes = set()
    for k in range(30):
        host = online.Online(rig.net, name=f"P{k}", seed=7)                    # (the same seed: the second must still find another code)
        codes.add(host.host(1, 7))
        rig.run(0.1, host)
    assert len(codes) == 30


# --- pairing -----------------------------------------------------------------------------------------------------------------------------------
def test_a_guest_who_joins_is_paired_with_the_host_and_each_is_told_who_it_is_and_how_the_match_goes():
    rig = Rig(latency_s=0.04)
    host, guest, code = paired(rig, pace=3, target=11)
    (h,), (g,) = rig.of(host, online.Paired), rig.of(guest, online.Paired)
    assert (h.role, h.opponent, h.pace, h.target, h.code) == ("host", "RAFAE", 3, 11, code)
    assert (g.role, g.opponent, g.pace, g.target, g.code) == ("guest", "MAYA", 3, 11, code)
    assert h.remote.sid == g.remote.sid and h.remote.gid == g.remote.gid and h.remote.host and not g.remote.host
    assert (h.remote.opponent_name, g.remote.opponent_name) == ("RAFAE", "MAYA")
    assert host.state == guest.state == "paired"


def test_the_game_is_off_the_list_once_it_has_a_guest():
    rig = Rig()
    paired(rig)
    assert rig.net.entries == {}


def test_a_guest_asks_again_until_it_is_answered_and_the_host_pairs_only_once():
    rig = Rig()
    host, guest = rig.player("MAYA"), rig.player("RAFAE", 2)
    code = host.host(1, 7)
    rig.run(0.3, host)
    guest.join(code)
    rig.run(2.5, guest)                                                        # the host's laptop is busy for a while
    rig.run(1.0, host, guest)
    assert len(rig.of(host, online.Paired)) == len(rig.of(guest, online.Paired)) == 1
    assert not rig.of(host, online.Failed) and not rig.of(guest, online.Failed)


def test_two_guests_at_once_one_gets_the_game_and_the_other_is_told_it_is_full():
    rig = Rig()
    host, amy, bob = rig.player("MAYA"), rig.player("AMY", 2), rig.player("BOB", 3)
    code = host.host(2, 7)
    rig.run(0.3, host)
    amy.join(code)
    bob.join(code)
    rig.run(1.5, host, amy, bob)
    got = [p for p in (amy, bob) if rig.of(p, online.Paired)]
    left_out = [p for p in (amy, bob) if rig.of(p, online.Failed)]
    assert len(got) == len(left_out) == 1 and got != left_out
    assert rig.of(left_out[0], online.Failed)[0].text == "THAT GAME IS FULL"
    assert len(rig.of(host, online.Paired)) == 1 and rig.of(host, online.Paired)[0].opponent == got[0].name
    assert left_out[0].state == "browsing" and got[0].state == "paired"


def test_a_late_guest_who_asks_after_the_game_began_is_told_it_is_full_by_the_game_itself():
    rig = Rig()
    host, guest, code = paired(rig)
    late = rig.player("LATE", 5)
    late.join(code)
    (paired_host,) = rig.of(host, online.Paired)
    for _ in range(100):                                                       # the host's game is running: its Remote hears the join
        rig.clock.advance_s(0.02)
        paired_host.remote.step(_NoGame(), rig.clock.now_ns())
        rig.events["LATE"] += late.step(rig.clock.now_ns())
    assert [f.text for f in rig.of(late, online.Failed)] == ["THAT GAME IS FULL"]


class _NoGame:
    """Just enough of a game for a Remote to listen with."""

    phase = "LOBBY"
    pause_reasons = frozenset()

    def set_pause(self, *args):
        pass

    def walkover(self, now_ns):
        return []

    def seconds_to_serve(self, now_ns):
        return None


# --- when it does not work -----------------------------------------------------------------------------------------------------------------------
def test_joining_a_game_nobody_hosts_fails_after_a_while_and_says_so():
    rig = Rig()
    guest = rig.player("RAFAE")
    guest.join("ZZZZZ")
    rig.run(online.JOIN_TIMEOUT_S - 1.0, guest)
    assert not rig.of(guest, online.Failed) and guest.state == "joining"
    rig.run(3.0, guest)
    assert [f.text for f in rig.of(guest, online.Failed)] == ["COULD NOT REACH THAT GAME"] and guest.state == "browsing"


def test_a_host_who_closes_the_game_while_a_guest_is_joining_tells_the_guest():
    rig = Rig(latency_s=0.05)
    host, guest = rig.player("MAYA"), rig.player("RAFAE", 2)
    code = host.host(1, 7)
    rig.run(0.3, host)
    guest.join(code)
    rig.run(0.01, guest)
    host.cancel()
    rig.run(0.5, host, guest)
    assert [f.text for f in rig.of(guest, online.Failed)] == ["THAT GAME WAS CLOSED"] and guest.state == "browsing"


def test_cancelling_a_hosted_game_takes_it_off_the_list_and_back_to_browsing():
    rig = Rig()
    host = rig.player("MAYA")
    host.host(1, 7)
    rig.run(0.3, host)
    assert len(rig.net.entries) == 1
    host.cancel()
    assert rig.net.entries == {} and host.state == "browsing" and host.code == ""


def test_the_server_not_being_there_is_said_after_a_few_seconds_and_forgotten_when_it_returns():
    rig = Rig()
    rig.net.up = False
    player = rig.player("MAYA")
    player.open()
    rig.run(2.0, player)
    assert player.view().status == "connecting" and player.view().message == ""
    rig.run(online.CONNECT_TIMEOUT_S, player)
    view = player.view()
    assert view.status == "offline" and "SERVER" in view.message
    rig.net.up = True
    rig.run(0.2, player)
    assert player.view().status == "browsing" and player.view().message == ""


def test_a_game_that_cannot_reach_the_server_fails_instead_of_waiting_for_ever():
    rig = Rig()
    rig.net.up = False
    host = rig.player("MAYA")
    host.host(1, 7)
    rig.run(online.CONNECT_TIMEOUT_S + 1, host)
    assert [f.text for f in rig.of(host, online.Failed)] == ["COULD NOT REACH THE GAME SERVER"] and host.state == "browsing"


# --- what the screens are told --------------------------------------------------------------------------------------------------------------------
def test_the_view_says_what_is_going_on_for_the_screens():
    rig = Rig()
    host, guest = rig.player("MAYA"), rig.player("RAFAE", 2)
    assert host.view().status == "idle" and host.view().rooms == ()
    guest.open()
    code = host.host(2, 9)
    rig.run(0.5, host, guest)
    assert (host.view().status, host.view().code, host.view().pace, host.view().target) == ("hosting", code, 2, 9)
    assert guest.view().status == "browsing" and guest.view().rooms[0]["host"] == "MAYA"
    guest.join(code)
    assert guest.view().status == "joining" and guest.view().code == code


def test_a_failure_stays_on_the_view_until_something_new_is_tried():
    rig = Rig()
    guest = rig.player("RAFAE")
    guest.join("ZZZZZ")
    rig.run(online.JOIN_TIMEOUT_S + 1, guest)
    assert guest.view().message == "COULD NOT REACH THAT GAME"
    guest.join("YYYYY")
    assert guest.view().message == ""


# --- afterwards ----------------------------------------------------------------------------------------------------------------------------------
def test_leaving_after_a_game_says_goodbye_to_the_other_and_goes_back_to_browsing():
    rig = Rig()
    host, guest, _ = paired(rig)
    (g,) = rig.of(guest, online.Paired)
    host.leave()
    rig.run(0.2, guest)
    assert [m["t"] for m in g.remote.link.poll() if m["t"] == "bye"] == ["bye"] and host.state == "browsing"


def test_closing_everything_leaves_nothing_behind():
    rig = Rig()
    host = rig.player("MAYA")
    host.host(1, 7)
    rig.run(0.3, host)
    host.close()
    assert rig.net.entries == {} and host.state == "idle" and host.rooms() == []


def test_the_name_is_cleaned_like_any_name_on_the_wire():
    assert online.Online(LoopNet(FakeClock()), name="  maya <b>!").name == "MAYA B"


def test_a_pace_or_a_target_that_is_not_one_is_refused():
    rig = Rig()
    host = rig.player("MAYA")
    for pace, target in ((0, 7), (4, 7), (1, 0), (1, 99)):
        with pytest.raises(ValueError):
            host.host(pace, target)
    assert rig.net.entries == {}
