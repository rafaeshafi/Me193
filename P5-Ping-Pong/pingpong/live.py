"""LiveRig: the real sensors, the actuator and the broker wired into one Session.

pump() is the body of the main loop (~60 Hz).  Its order matters:

    new poses -> sensor health (pause / reconnect) -> tags -> swings -> game tick -> haptics

Poses go in first because the judge looks back over the last 0.3 s of hand positions when a
swing arrives, and the swing is detected a few frames after it happened.  Sensor loss PAUSES
the game (a pause is never a hit or a fault: the ball clock stops and every deadline moves by
the time spent paused), but only while a game is running -- in the lobby a silent sensor
must not stop you from showing the START card.

Threaded mode (real hardware): the camera, the IMU parser and the actuator each own a thread
and pump() only polls their results.  Sync mode (tests, the fake rig): pump() drains the IMU
queue itself and runs the actuator's due work, and the caller steps the camera.
"""

import dataclasses
import threading
import time
from collections import deque

import cv2

import config
from pathlib import Path

from pingpong import app, mqtt_link, profile
from pingpong import recorder as recorder_mod
from pingpong.haptics import Actuator, ActuatorCore
from pingpong.hub import card_kwargs
from pingpong.imu_worker import ImuWorker
from pingpong.pose import PoseLock
from pingpong.shake import ShakeMonitor
from pingpong.swing import SwingDetector
from pingpong.vision import VisionWorker

ACTIVE_PHASES = ("COUNTDOWN", "RALLY", "POINT_OVER")
POSE_STALE_S = 0.6            # no hand reading for this long -> paused ("pose"); shorter than a
                              # Rookie flight (0.86 s), so a dropout cannot cost the player a ball
RECONNECT_AFTER_S = 2.0       # hub silent this long (after it went stale) -> try to reconnect
RECONNECT_COOLDOWN_S = 10.0   # ... then at most one attempt per this
MAX_RECONNECTS = 3            # per outage; then the HUD says LOST until the player presses R


class LiveSetupError(Exception):
    """Something the player can fix (card, camera, hub asleep); the message says how."""


class LiveRig:
    def __init__(self, session, *, hub, imu, vision, actuator=None, mqtt_client=None, clock,
                 threaded=False, stale_ms=None, pose_stale_s=POSE_STALE_S,
                 reconnect_after_s=RECONNECT_AFTER_S, reconnect_cooldown_s=RECONNECT_COOLDOWN_S,
                 max_reconnects=MAX_RECONNECTS, recorder=None, log=print):
        self.session, self.hub, self.imu, self.vision = session, hub, imu, vision
        self.recorder = recorder
        self._seen = {"phase": None, "pauses": None, "hub": None}
        self.actuator, self.mqtt_client, self.clock = actuator, mqtt_client, clock
        self.threaded, self.log = threaded, log
        self.stale_ms = config.STALE_MS if stale_ms is None else stale_ms
        self.pose_stale_s = pose_stale_s
        self.reconnect_after_ns = round(reconnect_after_s * 1e9)
        self.reconnect_cooldown_ns = round(reconnect_cooldown_s * 1e9)
        self.max_reconnects = max_reconnects
        self.n_impacts = 0
        self._last_pose_ns = None
        self._stale_since_ns = self._last_attempt_ns = None
        self._attempts, self._force_reconnect, self._reconnecting = 0, False, False
        self._loop_ms = deque(maxlen=20_000)
        self._closed = False

    # --- lifecycle ------------------------------------------------------------------------------
    def start(self):
        if self.threaded:
            for worker in (self.imu, self.actuator, self.vision):
                start = getattr(worker, "start", None)
                if start is not None:
                    start()
        return self

    def close(self):
        """Teardown order: actuator -> camera -> IMU -> broker -> hub (motors stopped, then disconnect).

        Each step has its own try/except: a hung or failing step must not leave the hub
        "connected" for the next run (a killed process leaves it so for ~24 s).
        """
        if self._closed:
            return
        self._closed = True
        steps = [("actuator", self._stop_actuator), ("camera", self.vision.stop), ("imu", self.imu.stop)]
        if self.recorder is not None:                # after the IMU thread stopped: the last samples are in
            steps.append(("recorder", self.recorder.close))
        if self.mqtt_client is not None and self.session.game.publisher is not None:
            steps.append(("mqtt", lambda: mqtt_link.shutdown(self.mqtt_client, self.session.game.publisher)))
        steps.append(("hub", self.hub.close))
        for name, step in steps:
            try:
                step()
            except Exception as exc:
                self.log(f"teardown: {name}: {exc}")

    def _stop_actuator(self):
        if self.actuator is None:
            return
        stop = getattr(self.actuator, "stop", None) or self.actuator.close
        stop()

    # --- the main loop body --------------------------------------------------------------------------
    def pump(self):
        t0 = time.perf_counter()
        now = self.clock.now_ns()
        if not self.threaded:
            self.imu.step()
        self._feed_poses()
        hub_stale = self.hub.is_stale(self.stale_ms)
        self._update_pauses(now, hub_stale)
        self._maybe_reconnect(now, hub_stale)
        for tag in self.vision.poll_tags():
            self._record("tag", tag.t_ns, {"role": tag.role, "value": tag.value})
            self.session.on_tag(tag)
        for until in self.imu.poll_locks():
            self._record("shake_lock", now, {"until_ns": until})
            self.session.game.judge.lock_paddle(until)               # gate J6: the hub is being shaken
        for swing in self.imu.poll():
            if swing.kind == "IMPACT":
                self.n_impacts += 1
                self._record("swing", swing.t_ns, dataclasses.asdict(swing))
                self._game_events(self.session.on_swing(swing))
        self._game_events(self.session.tick(data_ns=self.hub.last_rx_ns))
        if not self.threaded and hasattr(self.actuator, "process"):
            self.actuator.process(now)
        self._record_changes(now)
        if self.recorder is not None:
            self.recorder.tick(now)
        self._loop_ms.append((time.perf_counter() - t0) * 1000.0)

    # --- recording hooks (no-ops without a recorder) ------------------------------------------------------
    def _record(self, kind, t_ns, data):
        if self.recorder is not None:
            self.recorder.event(kind, t_ns, data)

    def _game_events(self, events):
        if self.recorder is not None and events:
            self.recorder.game_events(events)

    def _record_changes(self, now):
        """Phase, pause and hub-status transitions: what a replay needs to rebuild the same game."""
        if self.recorder is None:
            return
        game = self.session.game
        phase = (game.phase, game.mode, game.level.tag)
        if phase != self._seen["phase"]:
            self._seen["phase"] = phase
            self._record("phase", now, {"phase": game.phase, "level": game.level.tag, "mode": game.mode,
                                        "started_at_ns": game.started_at_ns, "player_points": game.player_points,
                                        "cpu_points": game.cpu_points})
        pauses = sorted(game.pause_reasons)
        if pauses != self._seen["pauses"]:
            self._seen["pauses"] = pauses
            self._record("pause", now, {"reasons": pauses})
        status = self.hub_status()
        if status != self._seen["hub"]:
            self._seen["hub"] = status
            self._record("hub", now, {"status": status})

    def _feed_poses(self):
        for pose in self.vision.snapshot():
            if self._last_pose_ns is None or pose.t_scene_ns > self._last_pose_ns:
                self.session.on_pose(pose)
                if self.recorder is not None:
                    self.recorder.pose(pose)
                self._last_pose_ns = pose.t_scene_ns

    def _update_pauses(self, now, hub_stale):
        """Pause while a sensor is lost; the pause is back-dated to the last moment it was heard.

        Noticing the loss takes the stale threshold, and the ball must not fly unobserved during
        it: on resume every deadline moves by the time since the LAST DATA, so the ball is
        exactly where the player last saw it.  (Deactivation passes the real time: that is the
        end of the pause.)
        """
        game = self.session.game
        active = game.phase in ACTIVE_PHASES
        age = self.vision.pose_age_s(now)
        pose_lost = active and (age is None or age > self.pose_stale_s)
        hub_since = now if self.hub.last_rx_ns is None else min(now, self.hub.last_rx_ns)
        pose_since = now if age is None else now - round(age * 1e9)
        hub_down = active and hub_stale
        game.set_pause("hub", hub_down, hub_since if hub_down else now)
        game.set_pause("pose", pose_lost, pose_since if pose_lost else now)

    # --- hub health -------------------------------------------------------------------------------------------
    def hub_status(self):
        """HUD text: ok | stale | reconnecting | lost."""
        if self._reconnecting:
            return "reconnecting"
        if self._stale_since_ns is None:
            return "ok"
        return "lost" if self._attempts >= self.max_reconnects else "stale"

    def reconnect_now(self):
        """The R key: give a lost hub a fresh budget of attempts and try at the next frame."""
        self._attempts, self._last_attempt_ns, self._force_reconnect = 0, None, True

    def _maybe_reconnect(self, now, hub_stale):
        if not hub_stale:
            self._stale_since_ns, self._attempts, self._last_attempt_ns = None, 0, None
            self._force_reconnect = False
            return
        if self._stale_since_ns is None:
            self._stale_since_ns = now
        if not self._force_reconnect:
            if self._attempts >= self.max_reconnects or now - self._stale_since_ns < self.reconnect_after_ns:
                return
            if self._last_attempt_ns is not None and now - self._last_attempt_ns < self.reconnect_cooldown_ns:
                return
        self._attempts += 1
        self._last_attempt_ns, self._force_reconnect = now, False
        if self.threaded:                          # a BLE scan blocks for seconds: never on the game thread
            self._reconnecting = True
            threading.Thread(target=self._reconnect, name="reconnect", daemon=True).start()
        else:
            self._reconnect()

    def _reconnect(self):
        try:
            self.hub.reconnect()
        except Exception as exc:
            self.log(f"hub reconnect failed: {exc}")
        finally:
            self._reconnecting = False

    # --- display + stats ------------------------------------------------------------------------------------------
    def display_frame(self):
        """The latest camera frame, mirrored so the player sees themselves as in a mirror."""
        frame = self.vision.latest_frame()
        return None if frame is None else cv2.flip(frame, 1)

    def hud_state(self):
        """The session's HUD state plus the IMU trace and the thresholds a swing is judged against."""
        game = self.session.game
        trace = tuple(rate for _, rate in self.imu.trace(1.5))
        return dataclasses.replace(self.session.hud_state(), swing_trace=trace, swing_scale=game.omega_hi,
                                   swing_threshold=game.judge.t_pk)

    def loop_stats(self):
        times = sorted(self._loop_ms)
        if not times:
            return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {"n": len(times), "p50_ms": times[len(times) // 2],
                "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))], "max_ms": times[-1]}


def assemble(*, hub, capture, landmarker, calibration, clock, tag_detector=None, mqtt_client=None, level=1,
             mode="survival", target=7, seed=1, source="live", scope=None, no_publish=False, no_motor=False,
             threaded=False, lag_s=None, gyro_per_dps=None, accel_per_g=None, fs_raw=None, stale_ms=None,
             to_image=None, record_dir=None, player="rafae", log=print):
    """Wire every piece into one LiveRig.  The real play.py and the fake rig both come through here,
    so the wiring that matters on hardware (haptic blank windows, phase-gated tag search, the pose
    lock, status lights) is exactly the wiring the tests run."""
    gpd = config.GYRO_PER_DPS if gyro_per_dps is None else gyro_per_dps
    apg = config.ACCEL_PER_G if accel_per_g is None else accel_per_g
    fs = config.HUB_FS_RAW if fs_raw is None else fs_raw
    params = calibration.swing_params(gpd, apg, fs)
    shake = ShakeMonitor(gyro_per_dps=gpd, rms_min_dps=0.35 * params.t_pk)     # motion smaller than this is tremor
    recorder = _start_recording(record_dir, log, source=source, player=player, seed=seed, level=level, mode=mode,
                                target=target, scope=scope or config.RECORD_SCOPE, t0_ns=clock.now_ns(),
                                calibration=calibration, gyro_per_dps=gpd, accel_per_g=apg, fs_raw=fs,
                                lag_s=config.CAMERA_LAG_S if lag_s is None else lag_s,
                                stale_ms=config.STALE_MS if stale_ms is None else stale_ms, no_motor=no_motor,
                                clock=clock)
    imu = ImuWorker(hub.imu, SwingDetector(params), shake=shake, recorder=recorder)
    core = ActuatorCore(hub.dev, clock=clock, no_motor=no_motor, on_blank=imu.blank)   # the pulse blanks the IMU
    actuator = Actuator(core, log=log) if threaded else core
    publishing = mqtt_client is not None and not no_publish
    session = app.make_session(
        level=level, mode=mode, target=target, clock=clock, actuator=actuator,
        client=mqtt_client if publishing else None, source=source, scope=scope or config.RECORD_SCOPE,
        no_publish=no_publish, seed=seed, box=calibration.box, omega_lo=calibration.swing.omega_lo,
        omega_hi=calibration.swing.omega_hi, t_pk=params.t_pk)
    lock = PoseLock()
    if calibration.shoulder_w:
        lock.calibrate(calibration.shoulder_w)
    vision = VisionWorker(capture, landmarker, clock=clock, hand=calibration.hand, lag_s=lag_s,
                          tag_detector=tag_detector, phase_fn=lambda: session.game.phase, lock=lock,
                          to_image=to_image)
    rig = LiveRig(session, hub=hub, imu=imu, vision=vision, actuator=actuator,
                  mqtt_client=mqtt_client if publishing else None, clock=clock, threaded=threaded,
                  stale_ms=stale_ms, recorder=recorder, log=log)
    session.bind_status(hub=rig.hub_status, mqtt=mqtt_link.status_fn(mqtt_client) if publishing else None)
    if publishing:
        mqtt_link.attach(mqtt_client, session.game.publisher)
    return rig


def _start_recording(record_dir, log, *, clock, **meta):
    """A Recorder for this session, or None: a recording that cannot start must never stop the game."""
    if record_dir is None:
        return None
    try:
        return recorder_mod.Recorder(record_dir, recorder_mod.session_meta(**meta), clock=clock, log=log)
    except OSError as exc:
        log(f"recording disabled: {exc}")
        return None


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


def _request_720p(capture):
    setter = getattr(capture, "set", None)
    if setter is not None:
        setter(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        setter(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def build_live(args, env, *, player_root=None, record_root=None, log=print):
    """Everything a live session needs, from the command line and an environment (real or fake).

    A failure after the hub connected lets the hub go again: a leaked connection would keep the
    hub invisible to the next run for about 24 seconds.
    """
    card = require_card(args)
    guest = profile.slug(args.player) == "guest"
    calibration = (None if guest else profile.load(args.player, root=player_root)) or profile.Calibration.default()
    no_publish = bool(args.no_publish or guest)                  # a guest must never touch the owner's score
    record_dir = None if args.no_record else Path(record_root or recorder_mod.default_root()) / \
        recorder_mod.session_name(args.player)
    hub = env.make_hub(config.NOTIFY_MS, card)
    try:
        hub.connect()
    except ConnectionError as exc:
        raise LiveSetupError(str(exc)) from exc
    capture = None
    try:
        capture = env.open_camera(config.CAMERA_INDEX)
        if not capture.isOpened():
            raise LiveSetupError(f"could not open the camera (index {config.CAMERA_INDEX}): allow Camera for this "
                                 "terminal in System Settings > Privacy & Security, quit apps using it, and turn "
                                 "Continuity Camera off on your iPhone")
        _request_720p(capture)
        rig = assemble(
            hub=hub, capture=capture, landmarker=env.make_landmarker(), calibration=calibration, clock=env.clock,
            tag_detector=env.make_tag_detector(), mqtt_client=None if no_publish else env.make_mqtt_client(),
            level=args.level, mode=args.mode, target=args.target, seed=args.seed, source="live",
            no_publish=no_publish, no_motor=args.no_motor, threaded=env.threaded,
            to_image=getattr(env, "to_image", None), record_dir=record_dir, player=args.player, log=log)
    except BaseException:
        if capture is not None:
            capture.release()
        hub.close()
        raise
    rig.calibration, rig.player = calibration, args.player
    if not calibration.calibrated:
        rig.session.set_notice(f"UNCALIBRATED: run ./pp calibrate_swing --player {args.player}")
    return rig
