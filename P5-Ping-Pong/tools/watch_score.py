"""Log what the broker holds on the score topic, the way the instructor's subscriber sees it.

test.mosquitto.org says not to rely on it, so this keeps a durable record: every message with its
time, topic, payload, retain flag and QoS, appended to recordings/watch_score.log, plus a summary of
the official topic (updates, last and highest value) and of any other topics seen.

Usage:
    ./pp watch_score                       # until Ctrl-C; topic ME193/Rogers/# (classmates too)
    ./pp watch_score --secs 120            # two minutes, then the summary
    ./pp watch_score --topic ME193/Rogers/RafaeShafi
    ./pp watch_score --selftest
"""

import argparse
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def _stamp():
    return datetime.now().isoformat(timespec="milliseconds")


def run(client, *, topic, secs, log_path, out, sleep=time.sleep, now=time.monotonic, stamp=_stamp):
    """Watch for `secs` seconds (0 = until interrupted); -> the stats dict."""
    stats = {"count": 0, "topics": Counter(), "official_last": None, "official_max": None, "official_count": 0,
             "official_bad": 0}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")

    def on_connect(c, userdata, flags, reason_code, properties=None):
        c.subscribe(topic, qos=1)

    def on_message(c, userdata, msg):
        payload = msg.payload.decode(errors="replace")
        line = f"{stamp()} {msg.topic} [{payload}] retain={int(bool(msg.retain))} qos={msg.qos}"
        stats["count"] += 1
        stats["topics"][msg.topic] += 1
        if msg.topic == config.SCORE_TOPIC:
            try:
                value = float(payload)
            except ValueError:
                stats["official_bad"] += 1
                line += "  NOT A FLOAT"
            else:
                stats["official_count"] += 1
                stats["official_last"] = value
                stats["official_max"] = value if stats["official_max"] is None else max(stats["official_max"], value)
        log.write(line + "\n")
        log.flush()
        out(line)

    client.on_connect, client.on_message = on_connect, on_message
    client.connect_async(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE_S)
    client.loop_start()
    try:
        end = now() + secs if secs else None
        while end is None or now() < end:
            sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        client.loop_stop()
        log.close()
    if stats["official_count"]:
        out(f"official topic {config.SCORE_TOPIC}: {stats['official_count']} updates, last {stats['official_last']}, "
            f"max {stats['official_max']}")
    else:
        out(f"nothing seen on {config.SCORE_TOPIC} (is anything publishing, and is the retained value gone?)")
    if stats["official_bad"]:
        out(f"WARNING: {stats['official_bad']} payload(s) on the official topic were not a float")
    others = {t: n for t, n in stats["topics"].items() if t != config.SCORE_TOPIC}
    if others:
        out("other topics: " + ", ".join(f"{t} x{n}" for t, n in sorted(others.items())))
    return stats


def _selftest():
    import tempfile

    from pingpong.sources_fake import FakeMqttClient

    client, clock = FakeMqttClient(), {"t": 0.0}

    def sleep(seconds):
        clock["t"] += seconds
        if 0.19 < clock["t"] < 0.41:
            client.deliver(config.SCORE_TOPIC, "6.0", retain=True)
        if 0.39 < clock["t"] < 0.61:
            client.deliver(config.SCORE_TOPIC, "9.0")

    with tempfile.TemporaryDirectory() as tmp:
        stats = run(client, topic="ME193/Rogers/#", secs=1.0, log_path=Path(tmp) / "w.log", out=lambda *_: None,
                    sleep=sleep, now=lambda: clock["t"])
    assert stats["official_max"] == 9.0 and stats["official_count"] >= 2, stats
    print("watch_score selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="ME193/Rogers/#")
    ap.add_argument("--secs", type=float, default=0.0)
    ap.add_argument("--log", default=str(config.HERE / "recordings" / "watch_score.log"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    from pingpong import mqtt_link

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run(mqtt_link.make_paho_client(), topic=args.topic, secs=args.secs, log_path=args.log, out=print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
