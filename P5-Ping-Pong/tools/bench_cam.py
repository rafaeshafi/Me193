"""Camera bench: pose speed, AprilTag read rate at the play position, camera-vs-IMU lag, hub rate under load.

Usage (Terminal.app only -- camera and Bluetooth):
    ./pp bench_cam                          # fps, tags, wave (lag + hub rate under load), mqtt, hud
    ./pp bench_cam --only tags              # just the printed cards
    ./pp bench_cam --only wave --secs-wave 15
    ./pp bench_cam --official-check         # also one retained 0.0 to the OFFICIAL topic, then cleared (asks first)
    ./pp bench_cam --selftest

Each step says what to do and waits for Enter; you then have --ready-s seconds (default 5) to walk to the
play position (about 1.8 m from the laptop).  Trustworthy numbers go to config_local.json (CAMERA_LAG_S,
CAMERA_INDEX), untrustworthy ones are reported and NOT written; the full report goes to
recordings/bench_cam.json.
"""

import argparse
import json
import signal
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

import config  # noqa: E402
from pingpong import benchstats, hud  # noqa: E402
from pingpong.hub import card_kwargs  # noqa: E402
from pingpong.vision import TAG_WIDTH, VisionWorker  # noqa: E402

STEPS = ("fps", "tags", "wave", "mqtt", "hud")
MIN_FPS, TAG_PASS, TAG_WARN, LAG_PASS_Q, LAG_WARN_Q, MAX_LAG_S = 20.0, 0.90, 0.60, 0.50, 0.35, 0.35
TAG_ADVICE = ("bigger cards, hold them nearer the camera or higher, matte print (no glare), or play at 1.5 m "
              "and re-measure")


@dataclass
class CheckResult:
    name: str
    status: str   # PASS | WARN | FAIL | SKIP
    detail: str


def _pump(env, vision, seconds, each=None):
    """Run the camera (stepping it ourselves when the environment has no camera thread) for `seconds`."""
    end = env.clock.now_ns() + round(seconds * 1e9)
    while env.clock.now_ns() < end:
        if not env.threaded:
            vision.step()
        if each is not None:
            each()
        env.sleep(0.01)


def _fps(env, vision, secs, add, prompt, ready_s):
    prompt("Stand at the play position (~1.8 m), head to hips in frame, front-lit, then stay put. Press Enter.")
    env.sleep(ready_s)
    frames0, no_pose0 = vision.n_frames, vision.n_no_pose
    _pump(env, vision, secs)
    stats, frames = vision.stats(), vision.n_frames - frames0
    missing = (vision.n_no_pose - no_pose0) / max(1, frames)
    detail = (f"{stats['fps']:.1f} fps, inference p50 {stats['infer_p50_ms']:.0f} ms / p95 "
              f"{stats['infer_p95_ms']:.0f} ms, no person found in {missing:.0%} of {frames} frames")
    if stats["fps"] < MIN_FPS:
        add("camera_fps", "FAIL", detail + ": below 20 fps. Close other apps using the camera or CPU, and keep "
            "the laptop on power")
    elif missing > 0.2:
        add("camera_fps", "FAIL", detail + ": no person found much of the time. Front-light yourself (no window "
            "behind you) and step back until head to hips fits in frame")
    else:
        add("camera_fps", "PASS", detail)


def _tags(env, vision, ids, secs, add, prompt, ready_s):
    detector = env.make_tag_detector()
    for tag_id in ids:
        prompt(f"Hold card {tag_id} up in your off hand, face to the camera, steady, for {secs:.0f} s. Press Enter.")
        env.sleep(ready_s)
        state = {"frames": 0, "hits": 0, "edges": [], "frame": None, "t": None}

        def sample():
            now, frame = env.clock.now_ns(), vision.latest_frame()
            if frame is None or frame is state["frame"] or (state["t"] is not None and now - state["t"] < 100_000_000):
                return
            state["frame"], state["t"] = frame, now
            h, w = frame.shape[:2]
            view = frame if w <= TAG_WIDTH else cv2.resize(frame, (TAG_WIDTH, int(h * TAG_WIDTH / w)))
            state["frames"] += 1
            for tag in detector.detect(view):
                if tag.id == tag_id:
                    state["hits"] += 1
                    pts = tag.corners
                    state["edges"].append(sum(((pts[(i + 1) % 4][0] - pts[i][0]) ** 2 + (pts[(i + 1) % 4][1] - pts[i][1]) ** 2)
                                              ** 0.5 for i in range(4)) / 4)

        _pump(env, vision, secs, each=sample)
        rate = state["hits"] / max(1, state["frames"])
        edge = sum(state["edges"]) / len(state["edges"]) if state["edges"] else 0.0
        detail = f"read in {rate:.0%} of {state['frames']} looks" + (f", about {edge:.0f} px wide" if edge else "")
        status = "PASS" if rate >= TAG_PASS else "WARN" if rate >= TAG_WARN else "FAIL"
        add(f"tag_{tag_id}", status, detail + ("" if status == "PASS" else f". Try: {TAG_ADVICE}"))


def _wave(env, vision, secs, add, prompt, ready_s, mqtt_load, card, config_path):
    hub = env.make_hub(config.NOTIFY_MS, card)
    try:
        hub.connect()
    except ConnectionError as exc:
        add("camera_lag", "FAIL", str(exc))
        return
    client = None
    try:
        prompt(f"Hold the hub in your fist at the play position and wave it side to side, IRREGULARLY (fast, slow, "
               f"big, small), for {secs:.0f} s. Press Enter.")
        env.sleep(ready_s)
        while not hub.imu.empty():
            hub.imu.get_nowait()
        if mqtt_load:                                        # a publisher at 5 Hz: the game's broker traffic
            client = env.make_mqtt_client()
            client.connect_async(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE_S)
            client.loop_start()
        data = {"imu_t": [], "imu_g": [], "pose_t": [], "pose_uv": [], "last": None, "load_t": None}

        def collect():
            while not hub.imu.empty():
                sample = hub.imu.get_nowait()
                data["imu_t"].append(sample.t_ns)
                data["imu_g"].append(sample.g)
            for pose in vision.snapshot():
                if data["last"] is None or pose.t_scene_ns > data["last"]:
                    data["pose_t"].append(pose.t_scene_ns)
                    data["pose_uv"].append((pose.u, pose.v))
                    data["last"] = pose.t_scene_ns
            now = env.clock.now_ns()
            if client is not None and (data["load_t"] is None or now - data["load_t"] >= 200_000_000):
                client.publish(config.SELFTEST_TOPIC, "load", qos=0)
                data["load_t"] = now

        _pump(env, vision, secs, each=collect)
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()
        hub.close()
    rate = benchstats.rate_stats(data["imu_t"])
    verdict = benchstats.rate_verdict(rate["hz"])
    add("hub_rate_under_load", {"GO": "PASS", "WARN": "WARN", "NO-GO": "FAIL"}[verdict],
        f"{rate['hz']:.1f} Hz with the camera, pose and MQTT running, worst gap {rate['worst_gap_ms']:.0f} ms, "
        f"p99.9 {rate['p999_gap_ms']:.0f} ms ({verdict})")
    try:
        lag, quality = benchstats.wave_lag_s(data["pose_t"], data["pose_uv"], data["imu_t"], data["imu_g"])
    except ValueError as exc:
        add("camera_lag", "FAIL", str(exc))
        return
    detail = f"camera trails the IMU by {lag * 1000:.0f} ms (correlation {quality:.2f})"
    if quality < LAG_WARN_Q or not 0.0 < lag <= MAX_LAG_S:
        add("camera_lag", "FAIL", detail + ": not trustworthy, NOT written. Wave bigger and less regularly, hold "
            "the hub clearly in view, and repeat (./pp bench_cam --only wave)")
        return
    config.write_local({"CAMERA_LAG_S": round(lag, 3)}, config_path)
    add("camera_lag", "PASS" if quality >= LAG_PASS_Q else "WARN", detail + f": written as CAMERA_LAG_S={lag:.3f}")


def _mqtt(env, official, add, ask):
    rtt = env.mqtt_roundtrip(config.SELFTEST_TOPIC, 10.0)
    if rtt is None:
        add("mqtt", "FAIL", f"no echo from {config.BROKER_HOST}:{config.BROKER_PORT} within 10 s (outbound 1883 "
            "blocked? try the iPhone hotspot)")
    else:
        add("mqtt", "PASS" if rtt < 2000 else "WARN", f"round trip {rtt:.0f} ms via {config.BROKER_HOST} "
            f"(scratch topic {config.SELFTEST_TOPIC})")
    if not official:
        return
    if not ask(f"Publish ONE retained 0.0 to the OFFICIAL topic {config.SCORE_TOPIC} and clear it again? "
               "It briefly changes what the instructor's subscriber sees. [y/N]"):
        add("mqtt_official", "SKIP", "declined; the first real publish will be the first on the official topic")
        return
    rtt = env.official_roundtrip(10.0)
    add("mqtt_official", "FAIL" if rtt is None else "PASS",
        "the broker did not accept/echo a retained QoS 1 publish on the official topic" if rtt is None
        else f"retained QoS 1 publish accepted and echoed in {rtt:.0f} ms, then cleared")


def _hud(show, add, ask):
    state = hud.HudState(phase="RALLY", streak=12, record=17, ball=(0.2, 0.5, 0.1), message="NEW RECORD",
                         last_kmh=27.0, last_label="perfect")
    show(hud.render(state, size=(1280, 720)))
    if ask("A sample HUD is on screen. From where you play, can you read the big score and see the ball? [y/N]"):
        add("hud_legibility", "PASS", "readable from the play position")
    else:
        add("hud_legibility", "FAIL", "not readable: use an external display or TV, move the laptop closer, or "
            "play at 1.5 m")


def run(env, *, card, ids=(0, 1, 2, 3), secs=None, only=STEPS, camera_index=0, official=False, mqtt_load=True,
        config_path, report_path, prompt, ask, out, ready_s=5.0, show=lambda frame: None):
    secs = {"fps": 10.0, "tag": 5.0, "wave": 12.0, **(secs or {})}
    results = []

    def add(name, status, detail):
        results.append(CheckResult(name, status, detail))
        out(f"[{status:<4}] {name}: {detail}")

    capture = vision = None
    try:
        if {"fps", "tags", "wave"} & set(only):
            capture = env.open_camera(camera_index)
            if not capture.isOpened():
                add("camera", "FAIL", f"could not open camera index {camera_index}: allow Camera for this terminal "
                    "(System Settings > Privacy & Security > Camera), quit apps using it, and turn Continuity Camera "
                    "off on your iPhone (permission or index problem)")
                capture = None
            else:
                from pingpong.live import request_720p

                request_720p(capture)
                if camera_index != config.CAMERA_INDEX:
                    config.write_local({"CAMERA_INDEX": camera_index}, config_path)
                vision = VisionWorker(capture, env.make_landmarker(), clock=env.clock, lag_s=0.0,
                                      to_image=getattr(env, "to_image", None))
                if env.threaded:
                    vision.start()
        if vision is not None:
            if "fps" in only:
                _fps(env, vision, secs["fps"], add, prompt, ready_s)
            if "tags" in only:
                _tags(env, vision, ids, secs["tag"], add, prompt, ready_s)
            if "wave" in only:
                _wave(env, vision, secs["wave"], add, prompt, ready_s, mqtt_load, card, config_path)
        if "mqtt" in only:
            _mqtt(env, official, add, ask)
        if "hud" in only:
            _hud(show, add, ask)
    finally:
        if vision is not None:
            vision.stop()
        elif capture is not None:
            capture.release()
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps({"results": [asdict(r) for r in results]}, indent=2) + "\n")
    return results, 1 if any(r.status == "FAIL" for r in results) else 0


def _selftest():
    import math

    from pingpong import fakerig
    from pingpong.sources_fake import FakeEnv, FakeLandmarker, FakeTagDetector

    S = 1_000_000_000
    env = FakeEnv(hz=66.0)
    t0, held = env.clock.now_ns(), {"id": None}
    env.scenario = lambda now: (0, 0, 1000, round(80.0 * fakerig.wave_speed((now - t0) / S) * fakerig.GPD), 0, 0)
    env.make_landmarker = lambda: FakeLandmarker(
        lambda t: (fakerig.wave_u((t - t0) / S), 0.0), env.clock, lag_s=0.08)
    env.make_tag_detector = lambda: FakeTagDetector(lambda t: {held["id"]} if held["id"] is not None else set(), env.clock)

    def prompt(text):
        held["id"] = int(text.split("card ")[1].split()[0]) if "Hold card" in text else None

    with tempfile.TemporaryDirectory() as tmp:
        results, code = run(env, card={}, secs={"fps": 2.0, "tag": 1.5, "wave": 12.0}, config_path=Path(tmp) / "c.json",
                            report_path=Path(tmp) / "r.json", prompt=prompt, ask=lambda text: True, out=lambda *_: None,
                            ready_s=0.0)
        lag = json.loads((Path(tmp) / "c.json").read_text())["CAMERA_LAG_S"]
    assert code == 0 and abs(lag - 0.08) < 0.02, (code, lag, results)
    print(f"bench_cam selftest OK: camera lag recovered as {lag * 1000:.0f} ms (true 80 ms)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-color", default=config.CARD_COLOR)
    ap.add_argument("--card-serial", default=config.CARD_SERIAL)
    ap.add_argument("--only", default=",".join(STEPS), help=f"comma list of: {', '.join(STEPS)}")
    ap.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--cards", default="0,1,2,3", help="which printed cards to test")
    ap.add_argument("--secs-fps", type=float, default=10.0)
    ap.add_argument("--secs-tag", type=float, default=5.0)
    ap.add_argument("--secs-wave", type=float, default=12.0)
    ap.add_argument("--ready-s", type=float, default=5.0, help="seconds to get into position after Enter")
    ap.add_argument("--official-check", action="store_true", help="also one retained 0.0 on the official topic (asks)")
    ap.add_argument("--no-mqtt-load", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    only = tuple(s.strip() for s in args.only.split(",") if s.strip())
    card = None
    if "wave" in only:
        from pingpong import live

        try:
            card = live.require_card(args)
        except live.LiveSetupError as exc:
            print(f"cannot run the wave step: {exc}", file=sys.stderr)
            return 2
    from pingpong import hostcheck
    from pingpong.realenv import RealEnv

    hostcheck.require_host("Camera and Bluetooth")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    def show(frame):                     # imshow creates the window on first use: one made earlier would sit undrawn
        cv2.imshow("P5 bench_cam", frame)   # through every Enter prompt and keep the Dock icon bouncing
        cv2.waitKey(1)

    try:
        _, code = run(RealEnv(), card=card, ids=tuple(int(x) for x in args.cards.split(",")),
                      secs={"fps": args.secs_fps, "tag": args.secs_tag, "wave": args.secs_wave}, only=only,
                      camera_index=args.camera_index, official=args.official_check, mqtt_load=not args.no_mqtt_load,
                      config_path=config.LOCAL_PATH, report_path=config.HERE / "recordings" / "bench_cam.json",
                      prompt=lambda text: input(text + "\n> "), ask=lambda text: input(text + "\n> ").strip().lower() == "y",
                      out=print, ready_s=args.ready_s, show=show)
    finally:
        cv2.destroyAllWindows()
    print("\nbench_cam:", "ALL GOOD" if code == 0 else "FIX THE FAIL LINES ABOVE, then re-run those steps (--only ...)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
