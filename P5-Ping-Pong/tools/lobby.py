"""The games friends have opened: what `./pp play --online` would list, and whether the game server can be reached at all.

Usage:
    ./pp lobby                                   # a few seconds of looking, then the list
    ./pp lobby --secs 10 --net-broker 192.168.1.20
    ./pp lobby --selftest

When a friend cannot find your game, run this on both laptops: if neither sees the game server the network is blocking it
(port 1883: tether to a phone, or run a broker of your own for everybody on one network), and if both do, the game is not open.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pingpong import levels  # noqa: E402


def run(net, *, secs, out, sleep=time.sleep, now=time.monotonic):
    """Look for `secs` seconds and print what is there; -> 0 if the game server was reached, else 1."""
    lobby = net.lobby()
    lobby.watch()
    start = now()
    try:
        while now() - start < secs:
            sleep(0.1)
            if lobby.connected and now() - start >= min(secs, 1.5):
                break
        sleep(0.5 if lobby.connected else 0.0)
        rooms = lobby.rooms()
        if not lobby.connected:
            host, port = config.net_broker()
            out(f"could not reach the game server at {host}:{port}: the network may block port 1883 (try a phone's hotspot or --net-broker)")
            return 1
        out("connected to the game server")
        if not rooms:
            out("no open games")
        for room in rooms:
            out(f"{room['code']}   {room['host']:<16} {levels.LEVELS[room['pace']].name.upper():<7} first to {room['target']}")
        return 0
    finally:
        lobby.close()


def selftest():
    from pingpong.clock import FakeClock
    from pingpong.loopnet import LoopNet

    clock = FakeClock(start_ns=1_000_000_000)
    net, seen = LoopNet(clock), []
    net.lobby().host("KQMDA", "maya", 2, 7)
    assert run(net, secs=2.0, out=seen.append, sleep=clock.advance_s, now=lambda: clock.now_ns() / 1e9) == 0 and any("KQMDA" in line for line in seen)
    net.up = False
    assert run(net, secs=2.0, out=seen.append, sleep=clock.advance_s, now=lambda: clock.now_ns() / 1e9) == 1
    print("lobby selftest OK: lists an open game, and says when the game server cannot be reached")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=4.0, help="how long to look (default 4 s)")
    ap.add_argument("--net-broker", metavar="HOST[:PORT]", default=None, help="the broker online games are played over (default: the class broker)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.net_broker:
        config.NET_BROKER_HOST, config.NET_BROKER_PORT = config.parse_broker(args.net_broker)
    from pingpong import netlink

    return run(netlink.Network(), secs=args.secs, out=print)


if __name__ == "__main__":
    raise SystemExit(main())
