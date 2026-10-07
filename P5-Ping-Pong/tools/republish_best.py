"""Re-send your best value to the official topic, once (if the broker lost the retained message).

Safe by default: without --yes it only prints what it WOULD send.  The value comes from --value or from
your best live Survival streak in the leaderboard database.  It sends exactly one retained QoS 1 float.

Usage:
    ./pp republish_best                       # dry run: shows the best value and what would be sent
    ./pp republish_best --yes                 # sends it
    ./pp republish_best --value 21 --yes
    ./pp republish_best --selftest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pingpong import store  # noqa: E402
from pingpong.mqtt_pub import ScorePublisher  # noqa: E402


def best_from_store(db_path, player):
    """The player's best live streak on file, or None."""
    if not Path(db_path).exists():
        return None
    stats = store.Store(db_path).player_stats(player)
    return stats["best_streak"] if stats["games"] and stats["best_streak"] > 0 else None


def run(client, *, value, topic=config.SCORE_TOPIC, yes=False, out=print):
    """-> 0 done (or dry run), 2 bad value."""
    try:
        payload = ScorePublisher.format(value)
    except (ValueError, TypeError) as exc:
        out(f"refusing to send {value!r}: {exc}")
        return 2
    if not yes:
        out(f"dry run: would publish {payload} (retained, QoS 1) to {topic}; add --yes to send it")
        return 0
    info = client.publish(topic, payload, qos=1, retain=True)
    wait = getattr(info, "wait_for_publish", None)
    if wait is not None:
        wait(timeout=5.0)
    out(f"sent {payload} (retained, QoS 1) to {topic}")
    return 0


def _selftest():
    from pingpong.sources_fake import FakeMqttClient

    client = FakeMqttClient()
    assert run(client, value=12, yes=False, out=lambda *_: None) == 0 and client.published == []
    assert run(client, value=12, yes=True, out=lambda *_: None) == 0
    assert client.published == [{"topic": config.SCORE_TOPIC, "payload": "12.0", "qos": 1, "retain": True}]
    assert run(client, value=-3, yes=True, out=lambda *_: None) == 2 and len(client.published) == 1
    print("republish_best selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--value", default=None)
    ap.add_argument("--player", default="rafae")
    ap.add_argument("--db", default=None)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    value = args.value
    if value is None:
        value = best_from_store(args.db or store.default_path(), args.player)
        if value is None:
            print(f"no best value: no live games for {args.player!r} in the leaderboard database; pass --value N")
            return 1
        print(f"best live streak for {args.player}: {value}")
    if not args.yes:
        return run(None, value=value, yes=False)
    from pingpong import mqtt_link

    client = mqtt_link.make_paho_client()
    client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE_S)
    client.loop_start()
    try:
        return run(client, value=value, yes=True)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
