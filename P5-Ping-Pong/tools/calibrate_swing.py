"""Guided calibration: shoulders, four reach corners, soft and full swings.

Usage (from Terminal.app: camera and Bluetooth are not available to the Claude app):
    cd ~/ME193/P5-Ping-Pong
    ./pp calibrate_swing --player rafae            # the card comes from config_local.json
    ./pp calibrate_swing --player rafae --hand left
    ./pp calibrate_swing --selftest                # no hardware needed

Stand about 1.8 m from the laptop with the hub in your fist, as you will play.  The window
tells you what to do and the hub beeps at each capture (you are too far away to press keys):
  1. stand still, arms down (shoulder width: the one-player lock)
  2. hold the hub still at four corners of where you can comfortably reach
  3. five SOFT swings, then five FULL swings
The result (forward axis, soft/full strengths, reach box) is saved to
data/players/<name>/calibration.json and `./pp play --player <name>` uses it.
Q cancels without saving.
"""

import argparse
import signal
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pingpong import canvas, live, profile  # noqa: E402
from pingpong.calibflow import CORNER_NAMES, CalibrationFlow  # noqa: E402

W, H = 1280, 720
TITLE = "P5 calibration  (Q to cancel)"
AMBER, GREEN, WHITE, GREY, RED = (40, 170, 255), (80, 220, 80), (255, 255, 255), (170, 170, 170), (70, 70, 240)
MAP_U, MAP_V = (-2.5, 2.5), (-2.0, 1.5)             # the part of the hand's (u, v) plane the map shows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-color", default=None)
    ap.add_argument("--card-serial", default=None)
    ap.add_argument("--player", default="rafae")
    ap.add_argument("--hand", choices=("right", "left"), default="right")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def _map_point(rect, u, v):
    x0, y0, w, h = rect
    fx = (u - MAP_U[0]) / (MAP_U[1] - MAP_U[0])
    fy = 1.0 - (v - MAP_V[0]) / (MAP_V[1] - MAP_V[0])
    return int(x0 + max(0.0, min(1.0, fx)) * w), int(y0 + max(0.0, min(1.0, fy)) * h)


def _draw_map(frame, flow, hand_uv, rect):
    import cv2

    x0, y0, w, h = rect
    panel = frame[y0:y0 + h, x0:x0 + w]
    panel[:] = (panel * 0.4).astype("uint8")
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), GREY, 1)
    cv2.circle(frame, _map_point(rect, 0.0, 0.0), 5, GREY, -1)                 # the shoulders' midpoint
    for i, (u, v) in enumerate(flow.corners):
        cv2.circle(frame, _map_point(rect, u, v), 9, GREEN, -1, cv2.LINE_AA)
        canvas.draw_text(frame, str(i + 1), (_map_point(rect, u, v)[0] + 12, _map_point(rect, u, v)[1] + 6), 0.5, GREEN, 1)
    if flow.step == "corners" and len(flow.corners) < 4:
        canvas.draw_text(frame, f"next: {CORNER_NAMES[len(flow.corners)]}", (x0 + 8, y0 + h - 10), 0.55, AMBER, 1)
    if hand_uv is not None:
        px, py = _map_point(rect, *hand_uv)
        cv2.circle(frame, (px, py), 12, WHITE, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3, WHITE, -1, cv2.LINE_AA)
    canvas.draw_text(frame, "YOUR HAND", (x0 + 8, y0 + 20), 0.5, GREY, 1)


def render_frame(flow, background, hand_uv, last_note, size=(W, H)):
    """One window frame: the mirrored camera, the step, the prompt, progress and the hand map."""
    w, h = size
    frame = canvas.fit_background(background, w, h) if background is not None else canvas.new_frame(w, h)
    done, total = flow.progress()
    canvas.draw_text(frame, f"{flow.step.upper()}  {done}/{total}" if flow.step != "done" else "DONE", (24, 54),
                     1.3, AMBER, 3)
    for i, line in enumerate(textwrap.wrap(flow.prompt(), 34)):
        canvas.draw_text(frame, line, (24, 120 + 56 * i), 1.35, WHITE, 3)
    bar_y, bar_w = h - 90, w - 48
    import cv2

    cv2.rectangle(frame, (24, bar_y), (24 + bar_w, bar_y + 18), GREY, 1)
    cv2.rectangle(frame, (24, bar_y), (24 + int(bar_w * done / max(1, total)), bar_y + 18), GREEN, -1)
    if last_note:
        canvas.draw_text(frame, last_note, (24, h - 30), 0.8, GREEN if "too" not in last_note else RED, 2)
    _draw_map(frame, flow, hand_uv, (w - 360, 90, 320, 240))
    return frame


def _beep(hub, note):
    """Audible feedback from the hub itself: you cannot read the screen mid-swing."""
    hz, count = 880, 1
    if "swing" in note and "too" not in note and "waving" not in note:
        hz = 1320
    if any(word in note for word in ("too ", "waving", "harder", "firmer", "on too long")):
        hz = 220
    if "calibration done" in note:
        hz, count = 2000, 3
    try:
        hub.dev.beep(frequency=hz, count=count, blocking=False)
    except Exception:
        pass


def run(env, args, *, profile_root=None, show, wait_key, notify, size=(W, H), frame_hz=30.0):
    """-> 0 saved, 1 cancelled, 2 could not start."""
    from pingpong.vision import VisionWorker

    try:
        card = live.require_card(args)
    except live.LiveSetupError as exc:
        print(f"cannot calibrate: {exc}", file=sys.stderr)
        return 2
    hub = env.make_hub(config.NOTIFY_MS, card)
    try:
        hub.connect()
    except ConnectionError as exc:
        print(f"cannot calibrate: {exc}", file=sys.stderr)
        return 2
    capture = vision = None
    try:
        capture = env.open_camera(config.CAMERA_INDEX)
        if not capture.isOpened():
            print("cannot calibrate: could not open the camera (allow Camera for this terminal, quit other "
                  "apps using it, turn Continuity Camera off)", file=sys.stderr)
            return 2
        live.request_720p(capture)
        vision = VisionWorker(capture, env.make_landmarker(), clock=env.clock, hand=args.hand,
                              to_image=getattr(env, "to_image", None))
        flow = CalibrationFlow(gyro_per_dps=config.GYRO_PER_DPS, hand=args.hand, accel_per_g=config.ACCEL_PER_G,
                               fs_raw=config.HUB_FS_RAW)
        if env.threaded:
            vision.start()
        return _loop(env, hub, vision, flow, args, profile_root, show, wait_key, notify, size, round(1e9 / frame_hz))
    finally:
        if vision is not None:
            vision.stop()                                 # also releases the camera
        elif capture is not None:
            capture.release()
        hub.close()


def _loop(env, hub, vision, flow, args, profile_root, show, wait_key, notify, size, frame_ns):
    import cv2

    last_pose_ns, last_note, hand, last_shown_ns = None, "", None, None
    while not flow.finished():
        if not env.threaded:
            vision.step()                                 # a synchronous environment has no camera thread
        while not hub.imu.empty():
            flow.feed_imu(hub.imu.get_nowait())
        for pose in vision.snapshot():
            if last_pose_ns is None or pose.t_scene_ns > last_pose_ns:
                flow.feed_pose(pose.t_scene_ns, pose.u, pose.v, pose.conf, vision.last_shoulder_w)
                last_pose_ns, hand = pose.t_scene_ns, (pose.u, pose.v)
        for note in flow.take_notes():
            last_note = note
            notify(note)
            _beep(hub, note)
        now = env.clock.now_ns()
        if last_shown_ns is None or now - last_shown_ns >= frame_ns:         # the camera is 30 fps: draw at 30 fps
            frame = vision.latest_frame()
            show(render_frame(flow, None if frame is None else cv2.flip(frame, 1), hand, last_note, size=size))
            last_shown_ns = now
            if wait_key(1) & 0xFF in (ord("q"), 27):
                return 1
        env.sleep(0.01)
    path = profile.save(args.player, flow.calibration(), root=profile_root)
    notify(f"saved {path}")
    show(render_frame(flow, None, hand, f"saved for {args.player}", size=size))
    wait_key(1500)
    return 0


def _selftest():
    import tempfile

    from pingpong import fakerig
    from pingpong.sources_fake import FakeEnv, FakeLandmarker

    S = 1_000_000_000
    script = fakerig.CalibrationScript()
    env = FakeEnv(hz=66.0)
    t0 = env.clock.now_ns()
    env.scenario = lambda now: (0, 0, 1000, *script.imu_raw((now - t0) / S))
    env.make_landmarker = lambda: FakeLandmarker(lambda t_ns: script.hand_uv((t_ns - t0) / S), env.clock)
    with tempfile.TemporaryDirectory() as tmp:
        args = parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "selftest"])
        code = run(env, args, profile_root=Path(tmp), show=lambda frame: None, wait_key=lambda ms: 255,
                   notify=lambda note: None, size=(320, 180), frame_hz=2.0)
        cal = profile.load("selftest", root=Path(tmp))
    assert code == 0 and cal is not None and cal.calibrated, (code, cal)
    dot = sum(a * b for a, b in zip(cal.swing.u_fwd, script.u_true))
    assert dot > 0.99, f"forward axis off by more than 8 degrees (cos {dot:.3f})"
    print(f"calibrate_swing selftest OK: axis cos {dot:.4f}, omega {cal.swing.omega_lo:.0f}/{cal.swing.omega_hi:.0f} dps")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return _selftest()
    try:
        live.require_card(args)                           # fail fast, before touching any hardware
    except live.LiveSetupError as exc:
        print(f"cannot calibrate: {exc}", file=sys.stderr)
        return 2
    import cv2

    from pingpong import hostcheck
    from pingpong.realenv import RealEnv

    hostcheck.require_host("Camera and Bluetooth")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    cv2.namedWindow(TITLE)
    try:
        return run(RealEnv(), args, show=lambda frame: cv2.imshow(TITLE, frame), wait_key=cv2.waitKey, notify=print)
    except KeyboardInterrupt:
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
