"""Finding a friend and pairing over real MQTT (the in-process broker of tests/minibroker.py, real paho clients, real threads)."""

import time

import pytest

import config
from pingpong import netlink, online
from tests.minibroker import MiniBroker

NS = 1_000_000_000


@pytest.fixture
def broker(monkeypatch):
    made = MiniBroker().start()
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", made.port)
    yield made
    made.stop()


def run_until(condition, players, seconds=12.0):
    """Step the players on the wall clock until the condition holds; -> {name: events}."""
    events = {p.name: [] for p in players}
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition(events):
        for p in players:
            events[p.name] += p.step(time.monotonic_ns())
        time.sleep(0.01)
    return events


def test_two_laptops_find_each_other_through_the_lobby_and_pair_over_mqtt(broker):
    net = netlink.Network()
    host, guest = online.Online(net, name="MAYA", seed=1), online.Online(net, name="RAFAE", seed=2)
    try:
        guest.open()
        code = host.host(3, 9)
        seen = []
        run_until(lambda e: seen.extend(guest.rooms()) or seen, [host, guest])
        assert seen and seen[0]["code"] == code and seen[0]["host"] == "MAYA" and (seen[0]["pace"], seen[0]["target"]) == (3, 9)
        guest.join(code)
        events = run_until(lambda e: e["MAYA"] and e["RAFAE"], [host, guest])
        (h,), (g,) = events["MAYA"], events["RAFAE"]
        assert isinstance(h, online.Paired) and isinstance(g, online.Paired)
        assert (h.role, h.opponent, h.pace, h.target) == ("host", "RAFAE", 3, 9)
        assert (g.role, g.opponent, g.pace, g.target) == ("guest", "MAYA", 3, 9)
        assert h.remote.sid == g.remote.sid
        deadline = time.monotonic() + 5
        while guest.rooms() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert guest.rooms() == []                                               # the game left the list when it was full
    finally:
        host.close(), guest.close()


def test_a_guest_who_asks_for_a_game_that_is_not_there_is_told_it_could_not_be_reached(broker, monkeypatch):
    monkeypatch.setattr(online, "JOIN_TIMEOUT_S", 1.0)
    guest = online.Online(netlink.Network(), name="RAFAE", seed=2)
    try:
        guest.join("ZZZZZ")
        events = run_until(lambda e: e["RAFAE"], [guest], seconds=8)
        assert [f.text for f in events["RAFAE"]] == ["COULD NOT REACH THAT GAME"]
    finally:
        guest.close()


def test_with_no_server_at_all_the_view_says_so_and_nothing_raises(monkeypatch):
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", 1)
    monkeypatch.setattr(online, "CONNECT_TIMEOUT_S", 0.5)
    player = online.Online(netlink.Network(), name="RAFAE")
    try:
        player.open()
        run_until(lambda e: player.view().status == "offline", [player], seconds=5)
        assert player.view().status == "offline" and "SERVER" in player.view().message
    finally:
        player.close()
