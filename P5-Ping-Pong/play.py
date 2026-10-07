"""P5 Ping-Pong: virtual table tennis with a LEGO Double Motor paddle.

Usage:
    ./pp play --fake                  # no hardware: mouse = paddle, SPACE starts and swings
    ./pp play --fake --level 2 --mode match --target 7
    ./pp play --selftest              # scripted 10-hit rally, checks the published scores
    ./pp play                         # LIVE: hub + camera + tags (see README / plan section 5)

Keys:  SPACE start (and swing in --fake)  1-3 level  M mode  X x-ray  D motors  Q/ESC quit
       --fake only:  J soft swing   K hard swing
Run camera/Bluetooth modes from Terminal.app, not from the Claude app.
"""

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

W, H = 1280, 720
TITLE = "P5 Ping-Pong  (Q to quit)"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="no hardware: mouse paddle, keyboard swings")
    ap.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--mode", choices=("survival", "match"), default="survival")
    ap.add_argument("--target", type=int, default=7, help="match target points (11 = win by 2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-publish", action="store_true", help="never touch the MQTT broker")
    ap.add_argument("--no-motor", action="store_true", help="mute the hub motors (beep + light stay)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    return ap.parse_args(argv)


def selftest():
    import config
    from pingpong import app, hud

    result = app.run_scripted(10, level=1)
    payloads = [p["payload"] for p in result["client"].published if p["topic"] == config.SCORE_TOPIC]
    assert payloads == [f"{n}.0" for n in range(1, 11)], payloads
    assert result["streak"] == 10 and result["record"] == 10
    frame = hud.render(result["session"].hud_state(), size=(W, H))
    assert frame.shape == (H, W, 3)
    print(f"play selftest OK: 10 hits in {result['sim_seconds']:.1f} simulated s, published {payloads[0]}...{payloads[-1]}")
    return 0


def run_fake(args):
    import cv2

    from pingpong import app, hud, keys
    from pingpong.clock import Clock
    from pingpong.events import PaddlePose

    clock = Clock()
    session = app.make_session(level=args.level, mode=args.mode, target=args.target, clock=clock,
                               client=None, source="fake", seed=args.seed)
    mouse = {"xy": (W // 2, int(H * 0.7))}
    cv2.namedWindow(TITLE)
    cv2.setMouseCallback(TITLE, lambda event, x, y, flags, param: mouse.update(xy=(x, y)))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            a = min(1.0, max(0.0, mouse["xy"][0] / W))
            b = min(1.0, max(0.0, 1.0 - mouse["xy"][1] / H))
            u, v = session.game.judge.box.to_uv(a, b)
            session.on_pose(PaddlePose(t_scene_ns=clock.now_ns(), u=u, v=v, conf=0.95, hand="right"))
            session.tick()
            cv2.imshow(TITLE, hud.render(session.hud_state(), size=(W, H)))
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and keys.handle_key(session, key, fake=True)[0] == "quit":
                break
    finally:
        cv2.destroyAllWindows()
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if args.fake and not args.no_window:
        return run_fake(args)
    print("Live mode (Double Motor + camera + AprilTags) is wired up in the next build step; "
          "run './pp play --fake' to play without hardware.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
