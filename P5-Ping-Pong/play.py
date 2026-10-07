"""P5 Ping-Pong: virtual table tennis with a LEGO Double Motor paddle.

Usage:
    ./pp play --fake                  # no hardware: mouse = paddle, SPACE starts and swings
    ./pp play --fake --level 2 --mode match --target 7
    ./pp play --selftest              # scripted rally + the whole live pipeline on fake hardware
    ./pp play                         # LIVE: Double Motor + camera + AprilTags (card from config_local.json)
    ./pp play --player rafae --level 2 --mode match
    ./pp play --player guest          # no saved calibration, never publishes to the score topic
    ./pp play --no-publish --no-motor # rehearse without the broker / without motor pulses

Keys:  SPACE start (and swing in --fake)  1-3 level  M mode  X x-ray  D motors  R reconnect hub  Q/ESC quit
       --fake only:  J soft swing   K hard swing
Run camera/Bluetooth modes from Terminal.app, not from the Claude app.
"""

import argparse
import signal
import sys
import time
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
    ap.add_argument("--card-color", default=None, help="Connection Card colour (default: config_local.json)")
    ap.add_argument("--card-serial", default=None, help="Connection Card serial, a 4-digit string")
    ap.add_argument("--player", default="rafae", help="player profile; 'guest' = no saved calibration, never publishes")
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

    from pingpong import fakerig

    rig = fakerig.FakeRig(level=1, vibration=True)
    rig.run(until=lambda: rig.game.tracker.streak >= 10, max_s=120)
    sent = [p["payload"] for p in rig.client.published if p["topic"] == config.SCORE_TOPIC]
    assert sent == [f"{n}.0" for n in range(1, 11)], sent
    assert rig.rig.n_impacts == rig.player.n_swings == 10, (rig.rig.n_impacts, rig.player.n_swings)
    rig.close()
    print(f"live pipeline selftest OK: fake hub + camera + tags + haptics -> 10 hits in {rig.now_s():.1f} "
          f"simulated s, one detected swing per swing made despite the motor pulses, published {sent[0]}...{sent[-1]}")
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


def run_loop(rig, *, show, wait_key, fps=60.0):
    """The window loop: pump the rig, draw the HUD, handle keys.  Returns when the player quits."""
    from pingpong import hud, keys

    frame_ms = 1000.0 / fps
    while True:
        t0 = time.perf_counter()
        rig.pump()
        show(hud.render(rig.session.hud_state(), size=(W, H), background=rig.display_frame()))
        key = wait_key(max(1, int(frame_ms - (time.perf_counter() - t0) * 1000.0))) & 0xFF
        if key == 255:
            continue
        if key == ord("r"):
            rig.reconnect_now()                       # give a lost hub a fresh budget of attempts
        elif keys.handle_key(rig.session, key, fake=False)[0] == "quit":
            return


def run_live(args):
    import cv2

    from pingpong import hostcheck, live
    from pingpong.realenv import RealEnv

    try:
        live.require_card(args)                       # cheap and hardware-free: fail fast with the fix
        hostcheck.require_host("Camera and Bluetooth")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))    # a kill still runs the teardown
        rig = live.build_live(args, RealEnv())
    except live.LiveSetupError as exc:
        print(f"cannot start live mode: {exc}", file=sys.stderr)
        return 2
    cv2.namedWindow(TITLE)
    try:
        rig.start()
        print(f"live: player {rig.player!r}, hub ready, camera on. SPACE or the START card begins; Q quits.")
        run_loop(rig, show=lambda frame: cv2.imshow(TITLE, frame), wait_key=cv2.waitKey)
    except KeyboardInterrupt:
        pass
    finally:
        rig.close()
        cv2.destroyAllWindows()
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if args.fake and not args.no_window:
        return run_fake(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
