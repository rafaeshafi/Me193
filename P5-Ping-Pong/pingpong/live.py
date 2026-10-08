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
import queue
import threading
import time
from collections import deque

import cv2

import config
from pingpong import app, flick, mqtt_link, posegyro
from pingpong import latency as latency_mod
from pingpong import posemodel
from pingpong import overrides as overrides_mod
from pingpong import recorder as recorder_mod
from pingpong.calibration import POSE_SHAKE_SETTINGS
from pingpong.haptics import Actuator, ActuatorCore
from pingpong.imu_worker import ImuWorker
from pingpong.livebuild import LiveSetupError, build_live, request_720p, require_card  # noqa: F401  (the setup half)
from pingpong.body import BodyTracker
from pingpong.pose import hand_xy
from pingpong.shake import ShakeMonitor
from pingpong.swing import SwingDetector
from pingpong.tilt import TiltEstimator
from pingpong.vision import VisionWorker

ACTIVE_PHASES = ("COUNTDOWN", "RALLY", "POINT_OVER")
POSE_STALE_S = 0.6            # no hand reading for this long -> paused ("pose"); shorter than a
                              # Rookie flight (1.2 s), so a dropout cannot cost the player a ball
RECONNECT_AFTER_S = 2.0       # hub silent this long (after it went stale) -> try to reconnect
RECONNECT_COOLDOWN_S = 10.0   # ... then at most one attempt per this
MAX_RECONNECTS = 3            # per outage; then the HUD says LOST until the player presses R


class LiveRig:
    def __init__(self, session, *, hub, imu, vision, actuator=None, mqtt_client=None, clock,
                 threaded=False, stale_ms=None, pose_stale_s=POSE_STALE_S,
                 reconnect_after_s=RECONNECT_AFTER_S, reconnect_cooldown_s=RECONNECT_COOLDOWN_S,
                 max_reconnects=MAX_RECONNECTS, recorder=None, pose_gyro=None, swing_source="imu", log=print):
        self.session, self.hub, self.imu, self.vision = session, hub, imu, vision
        self.pose_gyro, self.swing_source = pose_gyro, swing_source     # pose_gyro: the camera makes the swing samples
        self.recorder, self.store, self.audio = recorder, None, None    # store + audio are attached by build_live
        self.closers = []                                               # extra (name, callable) teardown steps
        self._seen = {"phase": None, "pauses": None, "hub": None}
        self.actuator, self.mqtt_client, self.clock = actuator, mqtt_client, clock
        self.threaded, self.log = threaded, log
        self.stale_ms = config.STALE_MS if stale_ms is None else stale_ms
        self.pose_stale_s = pose_stale_s
        self.reconnect_after_ns = round(reconnect_after_s * 1e9)
        self.reconnect_cooldown_ns = round(reconnect_cooldown_s * 1e9)
        self.max_reconnects = max_reconnects
        self.n_impacts = 0
        self._last_pose_ns = self._horizon_ns = None
        self._retained_told = False
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
        if self.audio is not None:
            steps.insert(1, ("audio", self.audio.stop))
        if self.recorder is not None:                # after the IMU thread stopped: the last samples are in
            steps.append(("summary", self._record_summary))
            steps.append(("recorder", self.recorder.close))
        if self.store is not None:
            steps.append(("store", self.store.close))
        steps += self.closers
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
        if self.pose_gyro is not None:
            self._drain_hub()
            if not self.threaded:
                self.imu.step()                                           # the camera's samples were queued just now
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
                data = dataclasses.asdict(swing)
                if self.session.game.spin_probs_fn is not None:           # what the game used: a replay reuses it
                    data["spin_probs"] = self.session.game.spin_probs_fn(swing.feat)
                self._record("swing", swing.t_ns, data)
                self._game_events(self.session.on_swing(swing))
        self._game_events(self.session.tick(data_ns=self._data_horizon_ns()))
        if not self.threaded and hasattr(self.actuator, "process"):
            self.actuator.process(now)
        self._record_changes(now)
        self._announce_retained()
        if self.recorder is not None:
            self.recorder.tick(now)
        self._loop_ms.append((time.perf_counter() - t0) * 1000.0)

    def _announce_retained(self):
        """Once, when the broker says what it holds: tell the player before a plain run writes over it."""
        publisher = self.session.game.publisher
        if self._retained_told or publisher is None or not publisher.retained:
            return
        self._retained_told, held = True, publisher.retained
        if publisher.resume:
            notice = f"RESUMING: the broker holds {held}, your best starts there"
            self.log(f"broker holds {held}.0 on the score topic: --resume keeps it as the best to beat")
        else:
            notice = f"BROKER HOLDS {held} - a new run publishes 1 over it (--resume keeps it, --no-publish leaves it)"
            self.log(f"broker holds {held}.0 on the score topic: this run starts from 0 and its first hit publishes "
                     "1.0 over it; stop and use --resume to keep it, or --no-publish to leave it alone")
        if not self.session.has_notice():
            self.session.set_notice(notice)

    def _record_summary(self):
        stats = getattr(self.vision, "stats", None)
        self._record("summary", self.clock.now_ns(), {"loop": self.loop_stats(), "impacts": self.n_impacts,
                                                      "vision": stats() if stats else None})

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
                self._game_events(self.session.on_pose(pose))             # in contact mode a hand into the ball is a hit
                if self.recorder is not None:
                    self.recorder.pose(pose)
                self._last_pose_ns = pose.t_scene_ns
                if self.pose_gyro is not None:                            # the camera is the swing sensor
                    sample = self.pose_gyro.feed(pose)
                    if sample is not None:
                        self.imu.samples.put_nowait(sample)
                        self._horizon_ns = sample.t_ns

    def _drain_hub(self):
        """Camera mode: the hub's own gyro is not used, and a queue nobody reads would grow all session."""
        while not self.hub.imu.empty():
            self.hub.imu.get_nowait()

    def _data_horizon_ns(self):
        """How far the swing sensor's data reaches: a miss is only a fact once it has passed the deadline."""
        return self._horizon_ns if self.pose_gyro is not None else self.hub.last_rx_ns

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
        hub_down = active and hub_stale and self.swing_source == "imu"   # the camera needs nothing from the hub
        game.set_pause("hub", hub_down, hub_since if hub_down else now)
        game.set_pause("pose", pose_lost, pose_since if pose_lost else now)

    # --- hub health -------------------------------------------------------------------------------------------
    def hub_status(self):
        """HUD text: ok | stale | reconnecting | lost | off (no hub at all)."""
        if getattr(self.hub, "absent", False):
            return "off"
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
        self.session.paddle_angle = self.imu.tilt_deg()
        return dataclasses.replace(self.session.hud_state(), swing_trace=trace, swing_scale=game.omega_hi,
                                   swing_threshold=game.judge.t_pk, hub_battery=self.hub.battery_pct(),
                                   hand_img=self._hand_img(),
                                   swing_label="CAMERA SWING" if self.swing_source == "pose" else "IMU SWING")

    def _hand_img(self):
        """Where the tracked hand is in the (mirrored) camera picture, as fractions across and down; None if unseen."""
        landmarks = getattr(self.vision, "last_landmarks", None)
        if landmarks is None:
            return None
        hx, hy = hand_xy(landmarks, getattr(self.vision, "hand", "right"))
        return (1.0 - hx, hy) if 0.0 <= hx <= 1.0 and 0.0 <= hy <= 1.0 else None

    def loop_stats(self):
        times = sorted(self._loop_ms)
        if not times:
            return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {"n": len(times), "p50_ms": times[len(times) // 2],
                "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))], "max_ms": times[-1]}


def assemble(*, hub, capture, landmarker, calibration, clock, tag_detector=None, mqtt_client=None, level=1,
             mode="survival", target=7, seed=1, source="live", scope=None, no_publish=False, no_motor=False,
             threaded=False, lag_s=None, gyro_per_dps=None, accel_per_g=None, fs_raw=None, stale_ms=None,
             to_image=None, record_dir=None, player="rafae", vision=None, recorder=None, spin_probs_fn=None,
             learner=None, pose_gyro=None, resume=False, overrides=None, latency=None, pose_model=None, hold_start=False,
             hit_mode="swing", flow=None, log=print):
    """Wire every piece into one LiveRig.  The real play.py and the fake rig both come through here,
    so the wiring that matters on hardware (haptic blank windows, phase-gated tag search, the pose
    lock, status lights) is exactly the wiring the tests run.

    The swing source is the calibration's: "imu" reads the hub's gyro, "pose" the camera's hand speed.  Live
    camera mode passes a PoseGyro (it makes the swing samples from the poses); a replay of a camera session
    passes none, because the recorded pose-derived samples arrive through the replay hub like hub samples.

    hold_start: the game also starts when the hub is held on the START button (live play; a replay starts its games
    where the recording says).  flow: the intro, title and menus in front of the game (live play; None: the plain lobby).
    hit_mode: "swing" (a swing the IMU sees meets the ball) or "contact" (the hand moving into the ball does, and the
    hub's flick is the spin); a replay passes the recorded one."""
    camera = calibration.swing.source == "pose"
    latency = latency or latency_mod.Latency.from_config()
    pose_model = pose_model or posemodel.PoseModel.default()
    if pose_gyro is not None and not camera:
        raise ValueError("a PoseGyro needs a camera calibration (swing source 'pose'), not a hub one")
    gpd = config.GYRO_PER_DPS if gyro_per_dps is None else gyro_per_dps
    apg = config.ACCEL_PER_G if accel_per_g is None else accel_per_g
    fs = config.HUB_FS_RAW if fs_raw is None else fs_raw
    if pose_gyro is not None:
        gpd, apg, fs = posegyro.GYRO_PER_DPS, posegyro.ACCEL_PER_G, posegyro.FS_RAW    # the units PoseGyro speaks
    params = calibration.swing_params(gpd, apg, fs)
    shake = ShakeMonitor(gyro_per_dps=gpd, rms_min_dps=calibration.swing.shake_rms_dps,       # motion smaller than this is tremor
                         **(POSE_SHAKE_SETTINGS if camera else {}))
    recorder = recorder or _start_recording(
        record_dir, log, source=source, player=player, seed=seed, level=level, mode=mode, target=target,
        scope=scope or config.RECORD_SCOPE, t0_ns=clock.now_ns(), calibration=calibration, gyro_per_dps=gpd,
        accel_per_g=apg, fs_raw=fs, lag_s=config.CAMERA_LAG_S if lag_s is None else lag_s,
        stale_ms=config.STALE_MS if stale_ms is None else stale_ms, no_motor=no_motor, learn=learner is not None,
        overrides=overrides, latency=dataclasses.asdict(latency), pose_model=pose_model.to_json(), hit_mode=hit_mode,
        clock=clock)
    tilt = (TiltEstimator(calibration.tilt, gpd, apg, nominal_hz=config.HUB_RATE_HZ or 64.0)
            if calibration.tilt is not None and not camera else None)
    imu = ImuWorker(hub.imu if pose_gyro is None else queue.SimpleQueue(), SwingDetector(params), shake=shake,
                    recorder=recorder, log=log, tilt=tilt, gyro_per_dps=None if camera else gpd)
    actuator = None                                      # no hub (--no-hub): no haptics, the sounds carry the cues
    if getattr(hub, "dev", None) is not None:
        # a pulse blanks the hub's gyro (the motors shake it); the camera does not feel the motors
        core = ActuatorCore(hub.dev, clock=clock, no_motor=no_motor, on_blank=None if camera else imu.blank)
        actuator = Actuator(core, log=log) if threaded else core
    publishing = mqtt_client is not None and not no_publish
    session = app.make_session(
        level=level, mode=mode, target=target, clock=clock, actuator=actuator,
        client=mqtt_client if publishing else None, source=source, scope=scope or config.RECORD_SCOPE,
        no_publish=no_publish, seed=seed, box=calibration.box, omega_lo=calibration.swing.omega_lo,
        omega_hi=calibration.swing.omega_hi, t_pk=params.t_pk, spin_probs_fn=spin_probs_fn, learner=learner,
        resume=resume, latency=latency, hand_model=pose_model.predictor, hold_start=hold_start, hit_mode=hit_mode, flow=flow,
        wrist_frame=None if camera or calibration.tilt is None else flick.wrist_frame(calibration.tilt.neutral,
                                                                                      calibration.tilt.axis),
        gyro_window=None if camera else imu.gyro_window)
    if vision is None:
        vision = VisionWorker(capture, landmarker, clock=clock, hand=calibration.hand, lag_s=lag_s,
                              tag_detector=tag_detector, phase_fn=lambda: session.game.phase,
                              body=BodyTracker(ref=calibration.shoulder_w or None), filter_params=pose_model.filter,
                              to_image=to_image, log=log)
    rig = LiveRig(session, hub=hub, imu=imu, vision=vision, actuator=actuator,
                  mqtt_client=mqtt_client if publishing else None, clock=clock, threaded=threaded,
                  stale_ms=stale_ms, recorder=recorder, pose_gyro=pose_gyro,
                  swing_source=calibration.swing.source, log=log)
    session.bind_status(hub=rig.hub_status, mqtt=mqtt_link.status_fn(mqtt_client) if publishing else None)
    if overrides:
        overrides_mod.apply(rig, overrides)
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
