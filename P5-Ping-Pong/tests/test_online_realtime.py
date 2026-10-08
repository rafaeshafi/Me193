"""Two games played against each other in real time over real MQTT (an in-process broker, real paho clients and threads, the real clock):
the one place where what the network does, the clock and the game meet.  Kept short: the countdown and the face-off are shortened."""

import time

import pytest

import config
from pingpong import netlink, versus
from pingpong.clock import Clock
from tests.minibroker import MiniBroker
from tests.online_harness import Laptop

S = 1_000_000_000


@pytest.fixture
def broker(monkeypatch):
    made = MiniBroker().start()
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", made.port)
    yield made
    made.stop()


def run_until(laptops, condition, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for laptop in laptops:
            laptop.step(laptop.session.clock.now_ns())
        if condition():
            return True
        time.sleep(0.004)
    return False


def test_a_host_and_a_guest_find_each_other_over_mqtt_and_play_a_match_to_the_same_score_in_real_time(broker):
    net, clock = netlink.Network(), Clock()
    host = Laptop(clock, net, "MAYA", skill="perfect", seed=1, target=2, level=2, start=("host", 2), vs_s=0.3, countdown_s=0.4)
    guest = None
    try:
        assert run_until([host], lambda: host.online.code, 5)
        guest = Laptop(clock, net, "RAFAE", skill="idle", seed=2, target=5, level=1, start=("join", host.online.code), vs_s=0.3, countdown_s=0.4)
        assert run_until([host, guest], lambda: host.screen == guest.screen == "VS", 10)
        assert run_until([host, guest], lambda: host.screen == guest.screen == "RESULTS", 40)
        assert (host.game.player_points, host.game.cpu_points) == (2, 0) == (guest.game.cpu_points, guest.game.player_points)
        assert guest.game.target_points == 2 and guest.game.level.tag == 2                    # the host's match
        assert host.session.hud_state().results.title == "YOU WIN!" and guest.session.hud_state().results.title == "GOOD GAME!"
        assert run_until([host, guest], lambda: host.game.remote.ping_ms is not None and guest.game.remote.ping_ms is not None, 5)
    finally:
        host.session.close_online()
        if guest is not None:
            guest.session.close_online()


def test_a_laptop_that_vanishes_mid_match_is_noticed_through_the_brokers_last_will(broker, monkeypatch):
    monkeypatch.setattr(versus, "LOST_GRACE_S", 0.5)                                         # (twelve seconds of waiting in real life)
    net, clock = netlink.Network(), Clock()
    host = Laptop(clock, net, "MAYA", skill="perfect", seed=1, target=9, level=2, start=("host", 2), vs_s=0.3, countdown_s=0.4)
    guest = None
    try:
        assert run_until([host], lambda: host.online.code, 5)
        guest = Laptop(clock, net, "RAFAE", skill="perfect", seed=2, target=9, level=1, start=("join", host.online.code), vs_s=0.3, countdown_s=0.4)
        assert run_until([host, guest], lambda: host.game.tracker.streak >= 2, 30)
        link = guest.game.remote.link
        link.client.loop_stop()
        link.client._sock.close()                                                              # the laptop dies without a word
        assert run_until([host], lambda: host.screen == "RESULTS", 15)
        assert host.session.hud_state().results.title == "THEY LEFT" and host.game.remote.reason == "lost"
    finally:
        host.session.close_online()


def test_a_connection_that_drops_and_comes_back_in_the_middle_of_a_match_costs_a_pause_not_the_game(broker):
    net, clock = netlink.Network(), Clock()
    host = Laptop(clock, net, "MAYA", skill="perfect", seed=1, target=3, level=2, start=("host", 2), vs_s=0.3, countdown_s=0.4)
    guest = None
    try:
        assert run_until([host], lambda: host.online.code, 5)
        guest = Laptop(clock, net, "RAFAE", skill="perfect", seed=2, target=3, level=1, start=("join", host.online.code), vs_s=0.3, countdown_s=0.4)
        assert run_until([host, guest], lambda: host.game.tracker.streak >= 2, 30)
        guest.game.remote.link.client._sock.close()                                           # the wifi drops: paho notices and reconnects
        assert run_until([host, guest], lambda: guest.game.remote.link.ready and host.game.tracker.streak >= 5, 30)
        assert not host.game.remote.gone and not guest.game.remote.gone
        guest.player.skill = "idle"                                                           # and now the guest lets the balls go
        assert run_until([host, guest], lambda: host.screen == guest.screen == "RESULTS", 60)
        assert (host.game.player_points, host.game.cpu_points) == (3, 0) == (guest.game.cpu_points, guest.game.player_points)
    finally:
        host.session.close_online()
        if guest is not None:
            guest.session.close_online()
