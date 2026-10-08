"""The channel between two laptops: an in-memory pair for tests (with latency, loss and a cut cable) and the real thing over MQTT,
tried against a small local broker with real sockets: the room, the lobby of open games, the last will of a host that died."""

import time

import pytest

import config
from pingpong import netlink, netproto as proto
from pingpong.clock import FakeClock
from tests.minibroker import MiniBroker

S = 1_000_000_000
HIT = {"t": "hit", "n": 1, "v": 4.0, "start": [0.0, 0.22, 0.3], "aim": [0.5, 0.5], "top": 0.0, "side": 0.0, "loft": 0.0, "serve": True}


# --- the pair for tests ----------------------------------------------------------------------------------------------------------------------
def pair(**kw):
    clock = FakeClock(start_ns=5 * S)
    a, b = netlink.PairLink.pair(clock, **kw)
    return clock, a, b


def test_a_message_sent_arrives_at_the_other_end_after_the_latency_and_in_order():
    clock, a, b = pair(latency_s=0.05)
    a.send({"t": "pos", "x": 0.1})
    a.send({"t": "pos", "x": 0.2})
    assert b.poll() == []
    clock.advance_s(0.04)
    assert b.poll() == []
    clock.advance_s(0.02)
    got = b.poll()
    assert [m["x"] for m in got] == [0.1, 0.2] and a.poll() == [] and b.poll() == []


def test_jitter_makes_it_late_but_never_out_of_order():
    import random

    clock, a, b = pair(latency_s=0.03, jitter_s=0.08, rng=random.Random(4))
    for k in range(40):
        a.send({"t": "ping", "k": k, "ts": 0.0})
        clock.advance_s(0.01)
    clock.advance_s(1.0)
    assert [m["k"] for m in b.poll()] == list(range(40))


def test_what_comes_out_has_been_through_the_real_codec_so_a_bad_value_is_pulled_into_range():
    _, a, b = pair()
    a.send(dict(HIT, v=900.0))
    assert b.poll()[0]["v"] == proto.V_MAX


def test_junk_a_stranger_puts_on_the_wire_is_dropped():
    clock, a, b = pair()
    b.inject_raw(b"garbage")
    b.inject_raw(proto.encode({"t": "pos", "x": 0.5}))
    assert [m["t"] for m in b.poll()] == ["pos"]


def test_unreliable_messages_can_be_lost_and_reliable_ones_never_are_but_may_come_twice():
    import random

    clock, a, b = pair(loss0=0.5, dup1=0.5, rng=random.Random(2))
    for k in range(60):
        a.send({"t": "pos", "x": 0.0}, qos=0)
        a.send({"t": "ping", "k": k, "ts": 0.0}, qos=1)
    clock.advance_s(1.0)
    got = b.poll()
    assert 15 < sum(m["t"] == "pos" for m in got) < 45
    ks = [m["k"] for m in got if m["t"] == "ping"]
    assert set(ks) == set(range(60)) and len(ks) > 70                              # all arrived, some twice


def test_a_cut_cable_stops_everything_both_ways_and_says_the_link_is_dead():
    clock, a, b = pair(latency_s=0.01)
    assert a.alive and b.alive
    a.cut()
    a.send({"t": "pos", "x": 0.1})
    b.send({"t": "pos", "x": 0.2})
    clock.advance_s(1.0)
    assert a.poll() == [] and b.poll() == [] and not a.alive and not b.alive


def test_closing_a_pair_link_tells_nobody_and_stops_it_listening():
    clock, a, b = pair()
    a.close()
    b.send({"t": "pos", "x": 0.3})
    assert a.poll() == [] and not a.alive


# --- the guard on topics -----------------------------------------------------------------------------------------------------------------------
def test_nothing_here_can_reach_the_official_score_topic_or_anything_outside_its_own_corner():
    for topic in (config.SCORE_TOPIC, "ME193/Rogers/#", "ME193-pp/elsewhere", "", "#", "ME193-pp/v1/../../ME193/x"):
        with pytest.raises(ValueError):
            netlink.check_topic(topic)
    for topic in (proto.lobby_topic("ABCDE"), proto.lobby_filter(), proto.room_topic("ABCDE", "host")):
        assert netlink.check_topic(topic) == topic


# --- the real thing over MQTT ------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def broker(monkeypatch):
    made = MiniBroker().start()
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", made.port)
    yield made
    made.stop()


def wait_for(condition, seconds=8.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def collect(link, seconds=2.0, want=1):
    got = []
    wait_for(lambda: (got.extend(link.poll()) or len(got) >= want), seconds)
    return got


def test_a_host_and_a_guest_in_a_room_hear_each_other_and_not_themselves(broker):
    host, guest = netlink.RoomLink("ABCDE", "host"), netlink.RoomLink("ABCDE", "guest")
    try:
        assert wait_for(lambda: host.ready and guest.ready)
        host.send({"t": "welcome", "name": "RAFAE", "pace": 2, "target": 7, "sid": "abc123", "to": "k3x9"})
        guest.send({"t": "join", "name": "MAYA", "gid": "k3x9"})
        assert collect(host)[0] == {"t": "join", "name": "MAYA", "gid": "k3x9"}
        assert collect(guest)[0]["t"] == "welcome"
        time.sleep(0.2)
        assert host.poll() == [] and guest.poll() == []                                    # no echo of what each sent
        guest.send({"t": "pos", "x": 0.25}, qos=0)
        assert collect(host)[0] == {"t": "pos", "x": 0.25}
    finally:
        host.close(), guest.close()


def test_two_rooms_do_not_hear_each_other(broker):
    one_host, one_guest = netlink.RoomLink("AAAAA", "host"), netlink.RoomLink("AAAAA", "guest")
    other_host = netlink.RoomLink("BBBBB", "host")
    try:
        assert wait_for(lambda: one_host.ready and one_guest.ready and other_host.ready)
        one_guest.send({"t": "join", "name": "MAYA", "gid": "k3x9"})
        assert collect(one_host)[0]["t"] == "join"
        time.sleep(0.2)
        assert other_host.poll() == []
    finally:
        for link in (one_host, one_guest, other_host):
            link.close()


def test_a_stranger_shouting_junk_into_the_room_is_ignored(broker):
    host = netlink.RoomLink("ABCDE", "host")
    try:
        assert wait_for(lambda: host.ready)
        stranger = netlink.make_client()
        stranger.connect("127.0.0.1", broker.port)
        stranger.loop_start()
        stranger.publish(proto.room_topic("ABCDE", "guest"), b"\xff junk", qos=1).wait_for_publish(2)
        stranger.publish(proto.room_topic("ABCDE", "guest"), proto.encode({"t": "bye", "reason": "left"}), qos=1).wait_for_publish(2)
        assert [m["t"] for m in collect(host)] == ["bye"]
        stranger.loop_stop(), stranger.disconnect()
    finally:
        host.close()


def test_a_player_whose_laptop_dies_is_missed_at_once_through_the_last_will(broker):
    host, guest = netlink.RoomLink("ABCDE", "host"), netlink.RoomLink("ABCDE", "guest")
    try:
        assert wait_for(lambda: host.ready and guest.ready)
        guest.client.loop_stop()
        guest.client._sock.close()                                                          # no goodbye
        lost = collect(host, seconds=5)
        assert lost and lost[0]["t"] == "bye" and lost[0]["reason"] == "lost"
    finally:
        host.close()


def test_a_polite_goodbye_says_left_and_not_lost(broker):
    host, guest = netlink.RoomLink("ABCDE", "host"), netlink.RoomLink("ABCDE", "guest")
    try:
        assert wait_for(lambda: host.ready and guest.ready)
        guest.close()
        bye = collect(host)
        assert bye and bye[0] == {"t": "bye", "reason": "left"}
        time.sleep(0.3)
        assert host.poll() == []                                                             # and the will did not fire as well
    finally:
        host.close()


def test_a_goodbye_and_a_last_will_say_who_is_leaving_when_the_link_was_told_who_it_is(broker):
    host = netlink.RoomLink("ABCDE", "host", ident={"sid": "abc123"})
    guest = netlink.RoomLink("ABCDE", "guest", ident={"gid": "k3x9"})
    other = netlink.RoomLink("ABCDE", "guest", ident={"gid": "zz11"})
    try:
        assert wait_for(lambda: host.ready and guest.ready and other.ready)
        other.close()
        assert collect(host)[0] == {"t": "bye", "reason": "left", "gid": "zz11"}                   # polite: names the guest
        guest.client.loop_stop()
        guest.client._sock.close()                                                                   # no goodbye: the will
        lost = collect(host, seconds=5)
        assert lost and lost[0] == {"t": "bye", "reason": "lost", "gid": "k3x9"}
    finally:
        host.close()


def test_a_hosts_goodbye_and_will_carry_the_session_id(broker):
    host = netlink.RoomLink("ABCDE", "host", ident={"sid": "abc123"})
    guest = netlink.RoomLink("ABCDE", "guest", ident={"gid": "k3x9"})
    try:
        assert wait_for(lambda: host.ready and guest.ready)
        host.close()
        assert collect(guest)[0] == {"t": "bye", "reason": "left", "sid": "abc123"}
    finally:
        guest.close()


def test_a_link_to_a_broker_that_is_not_there_is_just_not_ready_and_not_alive(monkeypatch):
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", 1)
    link = netlink.RoomLink("ABCDE", "host")
    time.sleep(0.3)
    assert not link.ready and not link.alive
    link.send({"t": "pos", "x": 0.0}, qos=0)                                                 # sending into the void is not an error
    link.close()


def test_a_bad_room_code_is_refused_before_it_gets_into_a_topic():
    with pytest.raises(ValueError):
        netlink.RoomLink("AB/DE", "host")
    with pytest.raises(ValueError):
        netlink.RoomLink("ABCDE", "spectator")


# --- the lobby of open games ------------------------------------------------------------------------------------------------------------------------------
def test_a_game_a_host_opens_is_listed_for_a_watcher_even_one_who_looks_later(broker):
    lobby_host = netlink.Lobby()
    lobby_host.host("ABCDE", "RAFAE", 2, 7)
    early = netlink.Lobby()
    early.watch()
    try:
        assert wait_for(lambda: [r["code"] for r in early.rooms()] == ["ABCDE"])
        late = netlink.Lobby()
        late.watch()
        assert wait_for(lambda: late.rooms() and late.rooms()[0]["host"] == "RAFAE")
        assert late.rooms()[0] == dict(early.rooms()[0]) and late.rooms()[0]["pace"] == 2 and late.rooms()[0]["target"] == 7
        late.close()
    finally:
        early.close(), lobby_host.close()


def test_a_game_that_is_closed_disappears_from_the_list(broker):
    host = netlink.Lobby()
    watcher = netlink.Lobby()
    host.host("ABCDE", "RAFAE", 1, 7)
    watcher.watch()
    try:
        assert wait_for(lambda: len(watcher.rooms()) == 1)
        host.withdraw()
        assert wait_for(lambda: watcher.rooms() == [])
    finally:
        watcher.close(), host.close()


def test_a_host_whose_laptop_dies_does_not_leave_a_game_standing_in_the_list(broker):
    host = netlink.Lobby()
    watcher = netlink.Lobby()
    host.host("ABCDE", "RAFAE", 1, 7)
    watcher.watch()
    try:
        assert wait_for(lambda: len(watcher.rooms()) == 1)
        host._host_client.loop_stop()
        host._host_client._sock.close()
        assert wait_for(lambda: watcher.rooms() == [], seconds=5)
    finally:
        watcher.close()


def test_the_list_hides_your_own_game_old_games_and_nonsense_and_shows_the_newest_first(broker):
    now = [1_000_000.0]
    watcher = netlink.Lobby(now=lambda: now[0])
    for code, host, age in (("AAAAA", "OLD", 5 * 3600), ("BBBBB", "FRESH", 60), ("CCCCC", "NEWEST", 5), ("DDDDD", "ME", 10)):
        broker._route(proto.lobby_topic(code), proto.lobby_entry(code, host, 2, 7, now=now[0] - age), True)
    broker._route(proto.lobby_topic("EEEEE"), b"not json", True)
    broker._route(proto.lobby_topic("FFFFF"), proto.lobby_entry("GGGGG", "LIAR", 2, 7, now=now[0]), True)      # a code that does not match its topic
    watcher.watch(own_code="DDDDD")
    try:
        assert wait_for(lambda: len(watcher.rooms()) >= 2)
        time.sleep(0.2)
        assert [r["host"] for r in watcher.rooms()] == ["NEWEST", "FRESH"]
    finally:
        watcher.close()


def test_a_lobby_with_no_broker_is_empty_and_quiet(monkeypatch):
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", 1)
    lobby = netlink.Lobby()
    lobby.watch()
    lobby.host("ABCDE", "RAFAE", 1, 7)
    time.sleep(0.3)
    assert lobby.rooms() == [] and not lobby.connected
    lobby.close()


def test_a_mended_cable_carries_messages_again():
    clock, a, b = pair(latency_s=0.01)
    a.cut()
    a.send({"t": "pos", "x": 0.1})
    a.mend()
    a.send({"t": "pos", "x": 0.2})
    clock.advance_s(0.1)
    assert [m["x"] for m in b.poll()] == [0.2] and a.alive and b.alive


def test_withdrawing_a_game_lets_go_of_its_connection_and_hosting_again_does_not_leave_the_old_one_open(broker):
    lobby = netlink.Lobby()
    watcher = netlink.Lobby()
    watcher.watch()
    try:
        lobby.host("ABCDE", "RAFAE", 1, 7)
        first = lobby._host_client
        assert wait_for(lambda: first.is_connected() and [r["code"] for r in watcher.rooms()] == ["ABCDE"])
        lobby.withdraw()
        assert wait_for(lambda: watcher.rooms() == [] and not first.is_connected())
        lobby.host("FGHJK", "RAFAE", 2, 7)
        assert wait_for(lambda: [r["code"] for r in watcher.rooms()] == ["FGHJK"])
        second = lobby._host_client
        lobby.host("MNPQR", "RAFAE", 3, 7)                                                  # a second game while one is open: the first goes
        assert wait_for(lambda: [r["code"] for r in watcher.rooms()] == ["MNPQR"] and not second.is_connected())
    finally:
        watcher.close(), lobby.close()
