"""Shake a ghost connection loose: reconnect to the hub, stop its motors, beep, and let go cleanly.

A run that was killed leaves the Double Motor "connected" to a dead process for about 24 seconds, and a
connected hub does not show up in the next scan.  This keeps trying (default: 6 tries, 5 s apart) and
ends with a clean disconnect so the next run finds it at once.

Usage (Terminal.app only -- Bluetooth):
    ./pp reset_hub
    ./pp reset_hub --attempts 10
    ./pp reset_hub --selftest
"""

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def run(env, *, card, attempts=6, wait_s=5.0, out=print):
    for n in range(1, attempts + 1):
        hub = env.make_hub(config.NOTIFY_MS, card)
        try:
            hub.connect()
        except ConnectionError:
            out(f"attempt {n}/{attempts}: hub not found (a hub still 'connected' to a dead process hides for "
                "about 24 s)")
            if n < attempts:
                env.sleep(wait_s)
            continue
        try:
            hub.dev.beep(frequency=880, blocking=False)
        except Exception:
            pass
        battery = hub.battery_pct()
        hub.close()                                   # motors stopped, then a clean disconnect
        out("hub found, stopped and released" + (f" (battery {battery}%)" if battery is not None else ""))
        return 0
    out("the hub never appeared. Check: wake it with its button (the light should pulse), it is charged, "
        "the card colour/serial are right (./pp scan_hubs shows what is in range), no other program or phone "
        "holds the connection, and Bluetooth is on.")
    return 1


def _selftest():
    from pingpong.sources_fake import FakeEnv

    env = FakeEnv()
    assert run(env, card={}, attempts=2, wait_s=0.0, out=lambda *_: None) == 0
    assert run(FakeEnv(hub_found=False), card={}, attempts=2, wait_s=0.0, out=lambda *_: None) == 1
    print("reset_hub selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-color", default=config.CARD_COLOR)
    ap.add_argument("--card-serial", default=config.CARD_SERIAL)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--wait-s", type=float, default=5.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    from pingpong import hostcheck, live
    from pingpong.realenv import RealEnv

    try:
        card = live.require_card(args)
    except live.LiveSetupError as exc:
        print(f"cannot reset the hub: {exc}", file=sys.stderr)
        return 2
    hostcheck.require_host("Bluetooth")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    return run(RealEnv(), card=card, attempts=args.attempts, wait_s=args.wait_s)


if __name__ == "__main__":
    raise SystemExit(main())
