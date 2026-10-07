"""Teach the game your spin: 12 flat, 12 top and 12 back swings -> a per-player classifier.

Usage (Terminal.app: Bluetooth; no camera is needed):
    ./pp calibrate_swing --player rafae     # first: the swing axis and strengths the features are computed with
    ./pp train_spin --player rafae          # ~3 minutes; the hub beeps at every swing it recorded
    ./pp train_spin --per-class 15          # more swings give a steadier estimate
    ./pp train_spin --selftest

You are asked for the three kinds of swing in turn: FLAT (push straight through), TOPSPIN (brush up over
the ball, the wrist rolling forward) and BACKSPIN (chop down under it).  The model is a StandardScaler plus a
LogisticRegression on the 12 swing features.  Its accuracy is cross-validated (each swing is predicted by a
model that never saw it) and the game only uses it if that reaches 75%; otherwise it stays flat, and the
report says so.  Q cancels without saving.
"""

import argparse
import signal
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pingpong import canvas, live, profile, spin  # noqa: E402
from pingpong.spinflow import SpinCollector  # noqa: E402

W, H = 1280, 720
TITLE = "P5 spin training  (Q to cancel)"
AMBER, GREEN, WHITE, GREY = (40, 170, 255), (80, 220, 80), (255, 255, 255), (170, 170, 170)
COLORS = {"flat": WHITE, "top": AMBER, "back": (230, 200, 60)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-color", default=None)
    ap.add_argument("--card-serial", default=None)
    ap.add_argument("--player", default="rafae")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.per_class < spin.MIN_PER_CLASS:
        ap.error(f"--per-class must be at least {spin.MIN_PER_CLASS}")
    return args


def render_frame(collector, last_note, size=(W, H)):
    """Big prompt in the colour of the class being asked for, a progress bar and the latest note."""
    import cv2

    w, h = size
    frame = canvas.new_frame(w, h)
    done, total = collector.progress()
    label = collector.order[min(done, total - 1)]
    canvas.draw_text(frame, f"SPIN TRAINING  {done}/{total}", (24, 54), 1.2, GREY, 3)
    for i, line in enumerate(textwrap.wrap(collector.prompt(), 30)):
        canvas.draw_text(frame, line, (24, 150 + 70 * i), 1.6, COLORS[label] if not collector.finished() else GREEN, 4)
    cv2.rectangle(frame, (24, h - 90), (w - 24, h - 72), GREY, 1)
    cv2.rectangle(frame, (24, h - 90), (24 + int((w - 48) * done / max(1, total)), h - 72), GREEN, -1)
    if last_note:
        canvas.draw_text(frame, last_note, (24, h - 30), 0.8, GREEN, 2)
    return frame


def format_report(report):
    counts = report["n_per_class"]
    lines = ["swings recorded: " + ", ".join(f"{c} {counts[c]}" for c in spin.CLASSES),
             f"cross-validated accuracy: {report['cv_accuracy']:.0%} ({report['cv_folds']}-fold; each swing was "
             "predicted by a model that never saw it)", "confusion (row = what you did, column = what it guessed):"]
    for actual in spin.CLASSES:
        lines.append(f"  {actual:<5} " + "  ".join(f"{guess} {report['confusion'][actual][guess]:>2}"
                                                    for guess in spin.CLASSES))
    lines.append("the game will use spin from now on" if report["ship"] else
                 f"below {spin.SHIP_ACCURACY:.0%}: the game stays flat. Make the three gestures clearly different "
                 "and repeat (./pp train_spin)")
    return "\n".join(lines)


def run(env, args, *, profile_root=None, show, wait_key, notify, size=(W, H), frame_hz=30.0):
    """-> 0 trained (and saved), 1 cancelled, 2 could not start."""
    try:
        live.require_card(args)
    except live.LiveSetupError as exc:
        print(f"cannot train: {exc}", file=sys.stderr)
        return 2
    calibration = profile.load(args.player, root=profile_root)
    if calibration is None or not calibration.calibrated:
        print(f"cannot train: {args.player!r} has no calibration yet: run ./pp calibrate_swing --player "
              f"{args.player} first (the spin features are computed with it)", file=sys.stderr)
        return 2
    hub = env.make_hub(config.NOTIFY_MS, live.require_card(args))
    try:
        hub.connect()
    except ConnectionError as exc:
        print(f"cannot train: {exc}", file=sys.stderr)
        return 2
    collector = SpinCollector(calibration.swing_params(config.GYRO_PER_DPS, config.ACCEL_PER_G, config.HUB_FS_RAW),
                              args.per_class)
    last_note, last_shown, frame_ns = "", None, round(1e9 / frame_hz)
    try:
        while not collector.finished():
            while not hub.imu.empty():
                collector.feed_imu(hub.imu.get_nowait())
            for note in collector.take_notes():
                last_note = note
                notify(note)
                try:
                    hub.dev.beep(frequency=1320, blocking=False)
                except Exception:
                    pass
            now = env.clock.now_ns()
            if last_shown is None or now - last_shown >= frame_ns:
                show(render_frame(collector, last_note, size=size))
                last_shown = now
                if wait_key(1) & 0xFF in (ord("q"), 27):
                    return 1
            env.sleep(0.01)
    finally:
        hub.close()
    X, y = collector.result()
    model, report = spin.train(X, y)
    spin.save(args.player, model, report, root=profile_root)
    for line in format_report(report).splitlines():
        notify(line)
    wait_key(1500)
    return 0


def _selftest():
    import tempfile

    from pingpong import fakerig
    from pingpong.sources_fake import FakeEnv

    S = 1_000_000_000
    with tempfile.TemporaryDirectory() as tmp:
        profile.save("selftest", profile.Calibration.default(), root=Path(tmp))
        env = FakeEnv(hz=66.0)
        t0 = env.clock.now_ns()
        script = fakerig.SpinScript([c for _ in range(8) for c in spin.CLASSES], seed=3)
        env.scenario = lambda now: script.imu_raw((now - t0) / S)
        args = parse_args(["--card-color", "red", "--card-serial", "1131", "--player", "selftest", "--per-class", "8"])
        code = run(env, args, profile_root=Path(tmp), show=lambda f: None, wait_key=lambda ms: 255,
                   notify=lambda note: None, size=(320, 180), frame_hz=2.0)
        model = spin.load_for("selftest", root=Path(tmp))
    assert code == 0 and model is not None, (code, model)
    print("train_spin selftest OK: 24 simulated swings -> a model that ships")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return _selftest()
    try:
        live.require_card(args)
    except live.LiveSetupError as exc:
        print(f"cannot train: {exc}", file=sys.stderr)
        return 2
    import cv2

    from pingpong import hostcheck
    from pingpong.realenv import RealEnv

    hostcheck.require_host("Bluetooth")
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
