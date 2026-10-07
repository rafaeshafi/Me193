"""First contact with the hardware: imports, camera, hub (BLE), MQTT, hub IMU rate.

Usage (run it from Terminal.app -- macOS only lets your own terminal use the
camera and Bluetooth):
    cd ~/ME193/P5-Ping-Pong
    ./pp scan_hubs                                   # find your hub's colour + serial
    ./pp env_check --card-color green --card-serial 0997
    ./pp env_check --skip camera,mqtt                # hub only
    ./pp env_check --selftest                        # no hardware needed

The hub-rate step measures the plan's biggest unknown: how fast IMU notifications
really arrive at the play position (>=40 Hz GO, 25-40 WARN, <25 NO-GO -> pose
fallback).  Hold the hub in your fist about 1.8 m from the laptop and swing gently.
Measured numbers are written to config_local.json; the full report to
recordings/env_check.json.
"""

import argparse
import json
import signal
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pingpong import benchstats  # noqa: E402
from pingpong.hub import card_kwargs  # noqa: E402

IMPORTS = ("mediapipe", "cv2", "legoeducation", "bleak", "paho.mqtt.client", "sounddevice", "sklearn")


@dataclass
class CheckResult:
    name: str
    status: str   # PASS | WARN | FAIL | SKIP
    detail: str


def run_checks(env, *, card_color, card_serial, secs, skip, camera_index, config_path,
               report_path, prompt, out, notify_ms=None):
    notify_ms = notify_ms or config.NOTIFY_MS
    results = []

    def add(name, status, detail):
        results.append(CheckResult(name, status, detail))
        out(f"[{status:<4}] {name}: {detail}")

    link = None
    try:
        add(*_check_imports())
        _check_camera(env, camera_index, skip, add)
        link = _check_hub_connect(env, card_color, card_serial, notify_ms, skip, add, config_path)
        _check_mqtt(env, skip, add)
        _check_hub_rate(env, link, secs, notify_ms, skip, add, prompt, config_path)
    finally:
        if link is not None:
            link.close()
    report = {"results": [asdict(r) for r in results]}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return results, 1 if any(r.status == "FAIL" for r in results) else 0


def _check_imports():
    import importlib

    missing = []
    for mod in IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            missing.append(f"{mod} ({type(exc).__name__})")
    try:
        import cv2

        assert cv2.aruco.DICT_APRILTAG_36h11 is not None
    except Exception:
        missing.append("cv2.aruco AprilTag 36h11")
    if missing:
        return "imports", "FAIL", "missing: " + ", ".join(missing) + " -- activate the venv via ./pp"
    return "imports", "PASS", "mediapipe, cv2 (+aruco 36h11), legoeducation, bleak, paho, sounddevice, sklearn"


def _check_camera(env, index, skip, add):
    if "camera" in skip:
        return add("camera", "SKIP", "skipped")
    cap = env.open_camera(index)
    try:
        if not cap.isOpened():
            return add("camera", "FAIL", f"could not open camera index {index}: allow Camera for this "
                       "terminal (System Settings > Privacy & Security > Camera), quit other apps using it, "
                       "and turn Continuity Camera off on your iPhone if index 0 is the phone")
        last = None
        for _ in range(45):
            ok, frame = cap.read()
            if ok:
                last = frame
        if last is None:
            return add("camera", "FAIL", "camera opened but returned no frames")
        if int(last.max()) < 10:
            return add("camera", "FAIL", "frames are black: camera permission is missing for this "
                       "terminal -- grant Camera in System Settings > Privacy & Security and restart it")
        h, w = last.shape[:2]
        return add("camera", "PASS", f"{w}x{h}, mean brightness {float(last.mean()):.0f}/255 "
                   "(front-light yourself; avoid a window behind you)")
    finally:
        cap.release()


def _check_hub_connect(env, color, serial, notify_ms, skip, add, config_path):
    if "hub" in skip:
        add("hub_connect", "SKIP", "skipped")
        return None
    if not color or not serial:
        add("hub_connect", "FAIL", "no hub card configured: wake the Double Motor, run ./pp scan_hubs, "
            "then pass --card-color/--card-serial (they are saved to config_local.json after a "
            "successful connect)")
        return None
    try:
        card = card_kwargs(color, serial)
    except ValueError as exc:
        add("hub_connect", "FAIL", f"{exc}; see ./pp scan_hubs")
        return None
    link = env.make_hub(notify_ms, card)
    try:
        link.connect()
    except ConnectionError as exc:
        add("hub_connect", "FAIL", str(exc))
        return None
    try:
        link.dev.beep(frequency=880, blocking=False)   # one audible confirmation
    except Exception:
        pass
    config.write_local({"CARD_COLOR": str(color).lower(), "CARD_SERIAL": card["card_serial"],
                        "NOTIFY_MS": notify_ms}, config_path)
    battery = link.battery_pct()
    add("hub_connect", "PASS", f"connected to {str(color).lower()} {card['card_serial']} at {notify_ms} ms "
        f"notifications" + (f", battery {battery}%" if battery is not None else "") + "; you should hear a beep")
    return link


def _check_mqtt(env, skip, add):
    if "mqtt" in skip:
        return add("mqtt", "SKIP", "skipped")
    rtt = env.mqtt_roundtrip(config.SELFTEST_TOPIC, 5.0)
    if rtt is None:
        return add("mqtt", "FAIL", f"no echo from {config.BROKER_HOST}:{config.BROKER_PORT} within 5 s "
                   "(outbound 1883 blocked? try the iPhone hotspot)")
    status = "PASS" if rtt < 2000 else "WARN"
    return add("mqtt", status, f"round trip {rtt:.0f} ms via {config.BROKER_HOST}:{config.BROKER_PORT} "
               f"(scratch topic {config.SELFTEST_TOPIC}; the official topic was not touched)")


def _check_hub_rate(env, link, secs, notify_ms, skip, add, prompt, config_path):
    if "hub" in skip or link is None:
        return add("hub_rate", "SKIP", "no connected hub")
    if prompt:
        prompt(f"Hold the hub in your fist at the play position (~1.8 m from the laptop) and swing "
               f"gently for {secs:.0f} s after you press Enter.")
    while not link.imu.empty():
        link.imu.get_nowait()
    t0 = env.clock.now_ns()
    stamps = []
    while (env.clock.now_ns() - t0) / 1e9 < secs:
        env.sleep(0.05)
        while not link.imu.empty():
            stamps.append(link.imu.get_nowait().t_ns)
    stats = benchstats.rate_stats(stamps)
    verdict = benchstats.rate_verdict(stats["hz"])
    stale_ms = max(config.STALE_MS, round(stats["p999_gap_ms"] * 1.2))
    config.write_local({"HUB_RATE_HZ": round(stats["hz"], 1),
                        "HUB_WORST_GAP_MS": round(stats["worst_gap_ms"], 1),
                        "HUB_P999_GAP_MS": round(stats["p999_gap_ms"], 1),
                        "STALE_MS": float(stale_ms)}, config_path)
    detail = (f"{stats['hz']:.1f} Hz over {secs:.0f} s ({stats['n']} samples), worst gap "
              f"{stats['worst_gap_ms']:.0f} ms, p99.9 {stats['p999_gap_ms']:.0f} ms, gaps >100 ms: "
              f"{stats['gaps_over_100ms']} -> {verdict}")
    if verdict == "NO-GO":
        detail += (". Below 25 Hz `play` and `calibrate_swing` take the swing from the camera's hand speed "
                   "(--swing-source auto; run './pp calibrate_swing --swing-source pose' once, the hub stays for "
                   "haptics); re-run closer to the laptop and with other BLE devices off first")
    elif verdict == "WARN":
        detail += ". Use NOTIFY_MS 20-30 and widen the swing windows x1.3 (plan section 5)"
    add("hub_rate", {"GO": "PASS", "WARN": "WARN", "NO-GO": "FAIL"}[verdict], detail)


def _selftest():
    from pingpong.sources_fake import FakeEnv

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for hz, expect in ((66.0, 0), (20.0, 1)):
            _, code = run_checks(FakeEnv(hz=hz), card_color="green", card_serial="0997", secs=4.0, skip=(),
                                 camera_index=0, config_path=tmp / "c.json", report_path=tmp / "r.json",
                                 prompt=None, out=lambda *_: None)
            assert code == expect, (hz, code)
    print("env_check selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-color", default=config.CARD_COLOR)
    ap.add_argument("--card-serial", default=config.CARD_SERIAL)
    ap.add_argument("--secs", type=float, default=30.0, help="hub IMU stream length")
    ap.add_argument("--skip", default="", help="comma list of: camera, mqtt, hub")
    ap.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    skip = tuple(s.strip() for s in args.skip.split(",") if s.strip())
    from pingpong import hostcheck

    if "hub" not in skip or "camera" not in skip:
        hostcheck.require_host("Bluetooth" if "camera" in skip else "Camera and Bluetooth")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # still run teardown on kill/timeout
    from pingpong.realenv import RealEnv

    prompt = None if args.no_prompt else (lambda msg: input(msg + "\n> "))
    results, code = run_checks(RealEnv(), card_color=args.card_color, card_serial=args.card_serial,
                               secs=args.secs, skip=skip, camera_index=args.camera_index,
                               config_path=config.LOCAL_PATH,
                               report_path=config.HERE / "recordings" / "env_check.json",
                               prompt=prompt, out=print)
    print("\nenv_check:", "ALL GOOD" if code == 0 else "FIX THE FAIL LINES ABOVE, then re-run")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
