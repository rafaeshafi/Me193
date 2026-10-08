"""./pp lobby: the games friends have opened, and whether the game server can be reached at all."""

import config
from pingpong import netproto as proto
from pingpong.clock import FakeClock
from pingpong.loopnet import LoopNet
from tools import lobby


def looking(net, **kw):
    out, clock = [], net.clock

    def sleep(seconds):
        clock.advance_s(seconds)

    code = lobby.run(net, secs=kw.pop("secs", 2.0), out=out.append, sleep=sleep, now=lambda: clock.now_ns() / 1e9, **kw)
    return code, "\n".join(out)


def test_it_lists_the_open_games_with_their_host_speed_points_and_code():
    net = LoopNet(FakeClock(start_ns=1_000_000_000))
    host = net.lobby()
    host.host("KQMDA", "maya", 2, 7)
    other = net.lobby()
    other.host("FGHJK", "leo", 1, 11)
    code, text = looking(net)
    assert code == 0
    assert "KQMDA" in text and "MAYA" in text and "CLUB" in text and "7" in text
    assert "FGHJK" in text and "LEO" in text and "ROOKIE" in text and "11" in text


def test_it_says_when_nobody_has_opened_a_game_and_that_the_server_was_reached():
    code, text = looking(LoopNet(FakeClock(start_ns=1_000_000_000)))
    assert code == 0 and "no open games" in text.lower() and "connected" in text.lower()


def test_it_says_when_the_game_server_cannot_be_reached_and_where_it_tried(monkeypatch):
    net = LoopNet(FakeClock(start_ns=1_000_000_000))
    net.up = False
    monkeypatch.setattr(config, "NET_BROKER_HOST", "192.168.1.20")
    monkeypatch.setattr(config, "NET_BROKER_PORT", 1884)
    code, text = looking(net)
    assert code == 1 and "could not reach" in text.lower() and "192.168.1.20:1884" in text


def test_it_checks_itself():
    assert lobby.main(["--selftest"]) == 0
    assert proto.NAMESPACE                                      # (the lobby it reads is the game's own corner of the broker)
