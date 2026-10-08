"""build_live: the command line and an environment (real or fake) -> a LiveRig.

The setup half of the live pipeline (live.py is the running half and re-exports these names): which sensor
detects swings, whose calibration, then the hub, camera, broker, sounds and leaderboard.  Every failure the
player can fix says how, and a failure after the hub connected lets the hub go again.
"""

import sqlite3
from pathlib import Path

import cv2

import config
from pingpong import benchstats, online, posegyro, posemodel, profile, qbandit, spin
from pingpong import overrides as overrides_mod
from pingpong import recorder as recorder_mod
from pingpong import store as store_mod
from pingpong.hub import NoHub, card_kwargs


class LiveSetupError(Exception):
    """Something the player can fix (card, camera, hub asleep); the message says how."""


def require_card(args):
    """The Connection Card filter from --card-color/--card-serial or config_local.json."""
    color = getattr(args, "card_color", None) or config.CARD_COLOR
    serial = getattr(args, "card_serial", None) or config.CARD_SERIAL
    if not color or not serial:
        raise LiveSetupError("no hub card configured: wake the Double Motor, run './pp scan_hubs' to read its card "
                             "colour and serial, then pass --card-color/--card-serial (env_check saves them)")
    try:
        return card_kwargs(color, serial)
    except ValueError as exc:
        raise LiveSetupError(f"{exc} (see './pp scan_hubs')") from exc


def request_720p(capture):
    setter = getattr(capture, "set", None)
    if setter is not None:
        setter(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        setter(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def flow_for(args):
    """The intro, title and menus in front of the game, with the level and game from the command line (None: --classic).  --online,
    --host and --join skip them: the list of friends' games, a game opened at once, a game joined at once."""
    from pingpong.flow import Flow

    if getattr(args, "classic", False):
        return None
    start = (("host", args.level) if getattr(args, "host", False) else ("join", args.join) if getattr(args, "join", None)
             else ("online",) if getattr(args, "online", False) else None)
    return Flow(intro=not getattr(args, "no_intro", False), level_tag=args.level, mode=args.mode, start=start)


def add_swing_source_args(parser):
    """--swing-source and --no-hub, shared by play and the calibration tool so they always agree."""
    parser.add_argument("--swing-source", choices=("auto", "imu", "pose"), default="auto",
                        help="what detects swings: the hub's gyro (imu), the camera's hand speed (pose), or auto = "
                             "the camera if the hub bench measured under 25 Hz")
    parser.add_argument("--no-hub", action="store_true",
                        help="camera only: no hub, no haptics (implies --swing-source pose)")


def resolve_swing_source(args, log=print):
    """"imu" or "pose" from --swing-source / --no-hub and, for auto, the hub rate the bench measured."""
    choice = getattr(args, "swing_source", "auto")
    if getattr(args, "no_hub", False):
        if choice == "imu":
            raise LiveSetupError("--no-hub means no hub, so no hub gyro: drop --swing-source imu "
                                 "(the camera detects the swings)")
        return "pose"
    if choice != "auto":
        return choice
    rate = config.HUB_RATE_HZ
    if rate is not None and benchstats.rate_verdict(rate) == "NO-GO":
        log(f"swing source: the camera (the hub measured {rate:.0f} Hz, too slow to see a swing); "
            "--swing-source imu overrides")
        return "pose"
    return "imu"


def _save_game(db, player, summary, log):
    """A finished game goes on file; a database problem must never reach the game."""
    try:
        db.record_game(player, {**summary, "source": "live"})
    except Exception as exc:
        log(f"could not save the game to the leaderboard: {exc}")


def _landmarker(env, pose_model):
    """The pose model that reads the camera: the one the player's training chose (the light one without any)."""
    name = "lite" if pose_model is None else pose_model.landmarker
    return env.make_landmarker() if name == "lite" else env.make_landmarker(model=name)


def build_live(args, env, *, player_root=None, record_root=None, store_path=None, log=print):
    """Everything a live session needs, from the command line and an environment (real or fake).

    A failure after the hub connected lets the hub go again: a leaked connection would keep the
    hub invisible to the next run for about 24 seconds.
    """
    from pingpong.live import assemble                           # live.py imports this module for its public names

    swing_source = resolve_swing_source(args, log)
    camera = swing_source == "pose"
    try:
        settings = overrides_mod.check(overrides_mod.parse(getattr(args, "set", None) or []))
    except ValueError as exc:
        raise LiveSetupError(f"--set: {exc}") from exc
    if settings:
        log(f"settings changed from the defaults: {overrides_mod.format_settings(settings)}")
    if not camera and not config.is_measured("GYRO_PER_DPS"):
        log("WARNING: the hub's gyro units were never measured (GYRO_PER_DPS is a guess): run './pp bench_hub --guided' "
            "first (its three 360-degree turns), or swing strengths and thresholds can be off by 10x")
    card = None if args.no_hub else require_card(args)
    guest = profile.slug(args.player) == "guest"
    owner = profile.slug(args.player) == config.OWNER
    calibration = (None if guest else profile.load(args.player, root=player_root, source=swing_source)) \
        or profile.Calibration.default(swing_source)
    no_publish = bool(args.no_publish or guest or not owner)    # a guest, or anyone but the owner, must never touch the owner's score
    if not (guest or owner or args.no_publish):
        log(f"scores are not published to the score topic: it belongs to the player {config.OWNER}, and this player is {args.player}")
    record_dir = None if args.no_record else Path(record_root or recorder_mod.default_root()) / \
        recorder_mod.session_name(args.player)
    model = None
    if camera:
        if not args.no_spin:
            log("spin is off: it needs the hub's accelerometer, and the camera is the swing sensor (balls are flat)")
    else:
        try:
            model = None if args.no_spin else spin.load_for(args.player, root=player_root)
        except ValueError as exc:                                 # a damaged model file: say so, play without spin
            log(f"spin disabled: {exc}")
    pose_model = None if guest else posemodel.load_for(args.player, root=player_root)
    if pose_model is not None:
        log("pose model: " + ("trained filter" if pose_model.filter != posemodel.FilterParams() else "default filter")
            + (", trained hand predictor" if pose_model.predictor is not None else "")
            + (", the full pose model" if pose_model.landmarker != "lite" else ""))
    learner = None
    if args.learn:
        try:
            learner = qbandit.load_for(args.player, root=player_root) or qbandit.QBandit()
        except ValueError as exc:
            log(f"learning starts afresh: {exc}")
            learner = qbandit.QBandit()
    if args.no_hub:
        hub = NoHub()
    else:
        hub = env.make_hub(config.NOTIFY_MS, card)
        try:
            hub.connect()
        except ConnectionError as exc:
            raise LiveSetupError(f"{exc}; with --no-hub the game runs on the camera alone") from exc
    capture = None
    try:
        capture = env.open_camera(config.CAMERA_INDEX)
        if not capture.isOpened():
            raise LiveSetupError(f"could not open the camera (index {config.CAMERA_INDEX}): allow Camera for this "
                                 "terminal in System Settings > Privacy & Security, quit apps using it, and turn "
                                 "Continuity Camera off on your iPhone")
        request_720p(capture)
        rig = assemble(
            hub=hub, capture=capture, landmarker=_landmarker(env, pose_model), calibration=calibration, clock=env.clock,
            tag_detector=env.make_tag_detector(), mqtt_client=None if no_publish else env.make_mqtt_client(),
            level=args.level, mode=args.mode, target=args.target, seed=args.seed, source="live",
            no_publish=no_publish, no_motor=args.no_motor, threaded=env.threaded,
            to_image=getattr(env, "to_image", None), record_dir=record_dir, player=args.player,
            spin_probs_fn=None if model is None else model.probs, learner=learner,
            pose_gyro=posegyro.PoseGyro() if camera else None, resume=bool(getattr(args, "resume", False)), overrides=settings,
            pose_model=pose_model, hold_start=True, hit_mode=getattr(args, "hit_mode", "contact"), flow=flow_for(args),
            online=online.Online(env.make_network(), name=args.player), log=log)
    except BaseException:
        if capture is not None:
            capture.release()
        hub.close()
        raise
    rig.calibration, rig.player, rig.session.player = calibration, args.player, args.player
    battery = hub.battery_pct()
    if battery is not None and battery < 20:
        log(f"WARNING: the hub's battery is at {battery}%: charge it before a long session or the graded take")
    if learner is not None:
        rig.closers.append(("learner", lambda: qbandit.save_for(args.player, learner, root=player_root)))
    if not args.no_audio:
        rig.audio = rig.session.audio = env.make_audio()
        rig.audio.music_on = not getattr(args, "no_music", False)
        rig.audio.start()
    if not args.no_store:
        try:
            db = store_mod.Store(store_path)
        except (OSError, sqlite3.Error) as exc:
            log(f"leaderboard disabled: {exc}")
        else:
            rig.store = db
            rig.session.on_game_over = lambda summary: _save_game(db, args.player, summary, log)
            rig.session.leaderboard_fn = lambda: tuple(db.leaderboard(rig.session.game.mode, limit=5))
    if not (guest or owner or args.no_publish):
        rig.session.set_notice(f"SCORE NOT SENT: only the player {config.OWNER} publishes it")
    if not calibration.calibrated:
        flags = (" --swing-source pose" + (" --no-hub" if args.no_hub else "")) if camera else ""
        rig.session.set_notice(f"UNCALIBRATED: run ./pp calibrate_swing --player {args.player}{flags}")
    elif calibration.tilt is None and not camera:
        rig.session.set_notice(f"PADDLE STAYS UPRIGHT: run ./pp calibrate_swing --player {args.player} --tilt-only "
                               "to teach it how you turn the hub", soft=True)
    return rig
