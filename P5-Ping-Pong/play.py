"""P5 Ping-Pong: virtual table tennis with a LEGO Double Motor paddle.

Usage:
    ./pp play --fake                  # no hardware: mouse = paddle, SPACE starts and swings
    ./pp play --fake --level 2 --mode match --target 7
    ./pp play --selftest              # scripted rally + the whole live pipeline on fake hardware
    ./pp play                         # LIVE: Double Motor + camera + AprilTags (card from config_local.json)
    ./pp play --player rafae --level 2 --mode match
    ./pp play --player guest          # no saved calibration, never publishes to the score topic
    ./pp play --no-publish --no-motor # rehearse without the broker / without motor pulses
    ./pp play --no-intro --no-music   # straight to the title, quietly (the intro flies in from the sky and lasts ten seconds)
    ./pp play --classic               # the plain lobby with its hold-the-hub-on-START button, no menus, no intro
    ./pp play --hit-mode swing        # the old game: only a swing the hub's IMU sees meets the ball (default: contact)
    ./pp play --swing-source pose     # the camera's hand speed detects swings (auto when the hub measured < 25 Hz)
    ./pp play --no-hub                # camera only: no hub, no haptics (bring-up, or a flat battery)
    ./pp play --board                 # the leaderboard (best streaks, match wins) and nothing else

No keyboard needed: point with the hub (it is the cursor over the whole screen) and hold on a button to press it.
Keys:  SPACE start now (and swing in --fake)  ENTER next  , . or arrows move  DELETE back  1-3 opponent  M game  X x-ray  D motors  S sound
       R reconnect hub  Q/ESC quit       --fake only:  J soft swing   K hard swing
Run camera/Bluetooth modes from Terminal.app, not from the Claude app.
"""

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

W, H = 1280, 720
TITLE = "P5 Ping-Pong  (Q to quit)"


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="no hardware: mouse paddle, keyboard swings")
    ap.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--mode", choices=("survival", "rally", "match"), default="survival",
                    help="rally (= survival): the computer never misses, how long can you keep it going; match: to 7")
    ap.add_argument("--target", type=int, default=7, help="match target points (11 = win by 2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-publish", action="store_true", help="never touch the MQTT broker")
    ap.add_argument("--resume", action="store_true",
                    help="start from the best the broker already holds: nothing is published until you beat it")
    ap.add_argument("--no-motor", action="store_true", help="mute the hub motors (beep + light stay)")
    ap.add_argument("--no-record", action="store_true", help="do not write recordings/<session>/ (IMU, pose, events)")
    ap.add_argument("--no-audio", action="store_true", help="no game sounds (the S key mutes while playing)")
    ap.add_argument("--no-music", action="store_true", help="the sounds but no music in the intro and the menus")
    ap.add_argument("--no-intro", action="store_true", help="start at the title: skip the flight in from the sky")
    ap.add_argument("--classic", action="store_true",
                    help="the plain lobby (hold the hub on START, or SPACE) instead of the intro, title and menus")
    ap.add_argument("--no-spin", action="store_true", help="ignore the trained spin model: every ball is flat")
    ap.add_argument("--hit-mode", choices=("contact", "swing"), default="contact",
                    help="contact: the hand moving into the ball hits it and a flick of the wrist spins it; "
                         "swing: only a swing the hub's IMU detects meets the ball")
    ap.add_argument("--learn", action="store_true", help="the computer learns where you fail (Q-learning, saved per player)")
    ap.add_argument("--no-store", action="store_true", help="do not save finished games to the leaderboard database")
    ap.add_argument("--set", action="append", default=[], metavar="section.name=value",
                    help="tune without editing code, e.g. level.radius_sw=0.8, level.late_s=0.25, swing.t_pk=150, "
                         "judge.d95_s=0.2 (repeatable; ./pp report shows them; ./pp replay starts from them)")
    ap.add_argument("--board", action="store_true", help="print the leaderboard and exit")
    ap.add_argument("--db", default=None, help="leaderboard database (default: data/pingpong.db)")
    ap.add_argument("--card-color", default=None, help="Connection Card colour (default: config_local.json)")
    ap.add_argument("--card-serial", default=None, help="Connection Card serial, a 4-digit string")
    ap.add_argument("--player", default="rafae", help="player profile; 'guest' = no saved calibration, never publishes")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    from pingpong import livebuild

    livebuild.add_swing_source_args(ap)
    return ap


def parse_args(argv=None):
    args = make_parser().parse_args(argv)
    if args.mode == "rally":                       # what the screen calls the survival mode
        args.mode = "survival"
    return args


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

    cam = fakerig.FakeRig(level=1, swing_source="pose", hz=8.0)         # a hub far too slow to see a swing
    cam.run(until=lambda: cam.game.tracker.streak >= 10, max_s=120)
    sent = [p["payload"] for p in cam.client.published if p["topic"] == config.SCORE_TOPIC]
    assert sent == [f"{n}.0" for n in range(1, 11)], sent
    cam.close()
    print(f"camera pipeline selftest OK: the hand's own speed is the swing sensor (hub at 8 Hz ignored) -> 10 hits in "
          f"{cam.now_s():.1f} simulated s, published {sent[0]}...{sent[-1]}")
    return 0


def fake_loop(session, *, show, wait_key, mouse_xy):
    """The --fake window loop: the mouse is the paddle, keys swing.  Returns when the player quits."""
    from pingpong import hud, keys
    from pingpong.events import PaddlePose

    while True:
        x, y = mouse_xy()
        a, b = min(1.0, max(0.0, x / W)), min(1.0, max(0.0, 1.0 - y / H))
        u, v = session.game.judge.box.to_uv(a, b)
        session.on_pose(PaddlePose(t_scene_ns=session.clock.now_ns(), u=u, v=v, conf=0.95, hand="right"))
        session.tick()
        show(hud.render(session.hud_state(), size=(W, H)))
        key = wait_key(1) & 0xFF
        if key != 255 and keys.handle_key(session, key, fake=True)[0] == "quit":
            return


def make_flow(args):
    """The way into a game (intro, title, the choice of game and of opponent, the results), or None for --classic."""
    from pingpong.flow import Flow

    return None if args.classic else Flow(intro=not args.no_intro, level_tag=args.level, mode=args.mode)


def make_fake_session(args, clock=None):
    """The --fake game: the mouse is the hand, so pointing it at the screen and holding works as with the hub."""
    from pingpong import app
    from pingpong.clock import Clock

    return app.make_session(level=args.level, mode=args.mode, target=args.target, clock=clock or Clock(), client=None,
                            source="fake", seed=args.seed, hold_start=True, flow=make_flow(args))


def run_fake(args):
    import cv2

    session = make_fake_session(args)
    if not args.no_audio:
        from pingpong.audio import Audio

        session.audio = Audio(music=not args.no_music)
        session.audio.start()
    mouse = {"xy": (W // 2, int(H * 0.7))}
    cv2.namedWindow(TITLE)
    cv2.setMouseCallback(TITLE, lambda event, x, y, flags, param: mouse.update(xy=(x, y)))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        fake_loop(session, show=lambda frame: cv2.imshow(TITLE, frame), wait_key=cv2.waitKey,
                  mouse_xy=lambda: mouse["xy"])
    finally:
        if session.audio is not None:
            session.audio.stop()
        cv2.destroyAllWindows()
    return 0


MAX_BAD_FRAMES = 30            # this many frames in a row that raise end the session (with the error)


def run_loop(rig, *, show, wait_key, fps=60.0, log=print):
    """The window loop: pump the rig, draw the HUD, handle keys.  Returns when the player quits.

    One frame that raises (a bug that only shows on the real sensors) is reported once and skipped: the game, the
    score on the broker and the hub's connection are worth more than that frame.  A loop that fails every frame
    is not skipped forever: after MAX_BAD_FRAMES in a row the error ends the session (the rig is closed by the
    caller).  Control-C is a KeyboardInterrupt and is never caught here.
    """
    from pingpong import hud, keys

    frame_ms = 1000.0 / fps
    bad, seen = 0, {}
    while True:
        t0 = time.perf_counter()
        try:
            rig.pump()
            show(hud.render(rig.hud_state(), size=(W, H), background=rig.display_frame()))
            bad = 0
        except Exception as exc:
            bad += 1
            where = traceback.extract_tb(exc.__traceback__)[-1]               # the same bug at the same line is one report
            key = (type(exc).__name__, where.filename, where.lineno)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 1 or seen[key] % 100 == 0:
                log(f"frame skipped ({type(exc).__name__}: {exc} at {Path(where.filename).name}:{where.lineno}, "
                    f"{seen[key]} so far); the game carries on")
            if bad >= MAX_BAD_FRAMES:
                raise
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
        if not args.no_hub:
            live.require_card(args)                   # cheap and hardware-free: fail fast with the fix
        hostcheck.require_host("Camera" if args.no_hub else "Camera and Bluetooth")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))    # a kill still runs the teardown
        rig = live.build_live(args, RealEnv())
    except live.LiveSetupError as exc:
        print(f"cannot start live mode: {exc}", file=sys.stderr)
        return 2
    cv2.namedWindow(TITLE)
    try:
        rig.start()
        sensor = "camera" if rig.swing_source == "pose" else "hub gyro"
        hub = "no hub" if rig.hub_status() == "off" else "hub ready"
        print(f"live: player {rig.player!r}, swings from the {sensor}, {hub}, camera on. "
              "Point the hub at the screen and hold on a button, or SPACE / the START card to begin; Q quits.")
        run_loop(rig, show=lambda frame: cv2.imshow(TITLE, frame), wait_key=cv2.waitKey)
    except KeyboardInterrupt:
        pass
    finally:
        rig.close()
        cv2.destroyAllWindows()
    return 0


def print_board(args):
    from pingpong import store

    path = Path(args.db) if args.db else store.default_path()
    print(store.format_board(store.Store(path)) if path.exists() else "no games yet: play one with ./pp play")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if args.board:
        return print_board(args)
    if args.fake and not args.no_window:
        return run_fake(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
