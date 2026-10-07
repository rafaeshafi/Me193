"""A whole session on fake hardware: a scripted player plays through the REAL live pipeline.

Fake hub notifications -> the real legoeducation parser -> HubLink -> ImuWorker -> SwingDetector;
fake camera frames -> VisionWorker, with landmarks placed where the player's hand is;
a fake tag detector -> TagVoter; haptics -> ActuatorCore -> the fake hub; the score -> a fake
broker.  Everything between the sensors and the broker is production code wired by
live.assemble(), so this is what `./pp play --selftest` and tests/test_fakerig.py run.

Time is a FakeClock stepped at 240 Hz; the hub notifies at 66 Hz, the camera runs at 30 fps and the
game loop pumps at 60 Hz, like the real thing.  Blackout windows (seconds since the start) silence
the hub or hide the player to exercise the pause logic.
"""

import math
import random
import threading

from pingpong import live, posegyro
from pingpong.clock import FakeClock
from pingpong.events import ImuSample
from pingpong.hub import HubLink, NoHub
from pingpong.profile import Calibration
from pingpong.sources_fake import (FakeCamera, FakeDoubleMotor, FakeLandmarker, FakeMqttClient,
                                   FakeTagDetector)

S = 1_000_000_000
GPD = 10.0                    # raw gyro counts per deg/s of the fake hub (passed to assemble explicitly)
GRAVITY = 1000                # raw accelerometer counts at rest (z)
VIBRATION_DPS = 700.0         # a motor pulse "shakes" the fake IMU like a swing, unless it is blanked
SHAKE_DPS = 300.0             # amplitude of a deliberate shake (5 Hz)
WAVE_AMP = 0.35               # shoulder widths: amplitude of a deliberate hand wave (5 Hz), camera mode


def _smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def swing_offset(t, t0, dur, vpk, *, back=0.5, back_s=0.20, back_gap_s=0.22, recover_s=0.5):
    """How far (shoulder widths, along the swing direction) a hand scripted to swing has travelled at time t.

    A backswing (peak speed back*vpk), then the forward stroke from t0 for dur seconds (a half-sine speed pulse
    peaking at vpk shoulder widths per second, the shape the fake gyro pulses have), then a slow recovery to
    where the hand began, so consecutive swings do not drift.
    """
    total = 0.0
    for start, length, peak in ((t0 - back_gap_s, back_s, -back * vpk), (t0, dur, vpk)):
        if t > start:
            x = min(1.0, (t - start) / length)
            total += peak * length / math.pi * (1.0 - math.cos(math.pi * x))
    if t > t0 + dur:
        net = (vpk * dur - back * vpk * back_s) * 2.0 / math.pi
        total -= net * _smoothstep((t - t0 - dur) / recover_s)
    return total


class ScriptedPlayer:
    """Glides the hand to each ball's arrival point and swings so the stroke ends at the ball's t_c (the gyro peaks the
    judge's contact lag earlier)."""

    def __init__(self, box, *, w_pk=600.0, swing_s=0.15, timing_s=0.0, rest_uv=(0.0, -0.4), cards=(),
                 pose_motion=False, swing_dir=(1.0, 0.0)):
        self.box, self.w_pk, self.swing_s, self.timing_s = box, w_pk, swing_s, timing_s
        self.pose_motion, self.swing_dir = pose_motion, swing_dir    # camera mode: the hand itself makes the stroke
        self.wave_windows = []                                       # (start_s, stop_s) since rig start: a 5 Hz hand wave
        self._lock = threading.RLock()                               # the real-time rig has a thread per sensor
        self.cards = [(round(a * S), round(b * S), tag) for a, b, tag in cards]    # relative to rig start
        self.origin_ns = 0
        self._segments = [(0, 0, rest_uv, rest_uv)]          # (t0, t1, from_uv, to_uv)
        self._peaks = {}                                     # ball_id -> gyro peak time
        self._vibrations = []
        self.shake_windows = []                              # (start_s, stop_s) since rig start: 5 Hz on the y axis
        self._now = 0

    # --- decisions (called every step with the game state) -------------------------------------------
    def watch(self, game, now_ns):
        with self._lock:
            self._now = now_ns
            ball = game.incoming
            if ball is None or game.paused:
                return
            peak = ball.t_c_ns + round(self.timing_s * S) - round(game.judge.contact_lag_s * S)   # the stroke ends at t_c
            planned = self._peaks.get(ball.ball_id)
            if planned == peak:
                return
            if planned is None:                              # first sight of this ball: head for its arrival point
                # (finished before the backswing starts: the camera cannot tell a glide from a stroke that follows it closely)
                glide = min(0.43, 0.5 * max(0.0, (ball.t_c_ns - now_ns) / S - game.judge.contact_lag_s - 0.30))
                self._segments.append((now_ns, now_ns + round(glide * S), self._base_uv(now_ns),
                                       self.box.to_uv(ball.aim_ab[0], 0.5)))      # across the court, at the nominal depth
            self._peaks[ball.ball_id] = peak                 # (re)plan the swing: a pause moves t_c

    @property
    def n_swings(self):
        """Swings whose gyro peak has happened by now."""
        return sum(1 for peak in self._peaks.values() if peak <= self._now)

    def add_vibration(self, start_ns):
        with self._lock:
            if all(abs(start_ns - v) > 1_000_000 for v in self._vibrations):    # both motors of one pulse: once
                self._vibrations.append(start_ns)

    # --- what the sensors see ------------------------------------------------------------------------
    def _base_uv(self, t_ns):
        for t0, t1, a, b in reversed(self._segments):
            if t_ns >= t0:
                f = 1.0 if t1 <= t0 else _smoothstep((t_ns - t0) / (t1 - t0))
                return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
        return self._segments[0][2]

    def hand_uv(self, t_ns):
        with self._lock:
            u, v = self._base_uv(t_ns)
            if not self.pose_motion:
                return u, v
            travel = self._travel(t_ns)
            return u + travel * self.swing_dir[0], v + travel * self.swing_dir[1]

    def _travel(self, t_ns):
        """Camera mode: how far along the swing direction the hand is off its glide (strokes, then waves)."""
        t, vpk = t_ns / S, self.w_pk / posegyro.DPS_PER_SW_S
        total = sum(swing_offset(t, peak / S - self.swing_s / 2, self.swing_s, vpk)
                    for peak in self._peaks.values() if abs(t - peak / S) < 2.0)
        rel = (t_ns - self.origin_ns) / S
        for a, b in self.wave_windows:
            if a <= rel < b:
                total += WAVE_AMP * math.sin(2 * math.pi * 5.0 * (rel - a))
        return total

    def imu(self, t_ns):
        """Raw hub reading (ax, ay, az, gx, gy, gz): half-sine pulses on the forward (x) gyro axis."""
        with self._lock:
            gx = 0.0
            for peak in self._peaks.values():
                gx += self._pulse(t_ns, peak - self.swing_s / 2 * S, self.w_pk)
            for start in self._vibrations:
                gx += self._pulse(t_ns, start + 0.010 * S, VIBRATION_DPS)
            gy = 0.0
            rel = (t_ns - self.origin_ns) / S
            for a, b in self.shake_windows:
                if a <= rel < b:
                    gy += SHAKE_DPS * math.sin(2 * math.pi * 5.0 * (rel - a))
            return 0, 0, GRAVITY, round(gx * GPD), round(gy * GPD), 0

    def _pulse(self, t_ns, start_ns, peak_dps):
        x = (t_ns - start_ns) / (self.swing_s * S)
        return peak_dps * math.sin(math.pi * x) if 0.0 <= x <= 1.0 else 0.0

    def card_ids(self, t_ns):
        rel = t_ns - self.origin_ns
        return {tag for a, b, tag in self.cards if a <= rel < b}


class VibratingMotor(FakeDoubleMotor):
    """A fake hub whose motor pulses shake its own IMU, the way the real one does."""

    def __init__(self, player, clock, vibrate):
        super().__init__()
        self.player, self.clock, self.vibrate = player, clock, vibrate
        self.n_motor_pulses = 0

    def motor_run_for_time(self, time_ms, **kwargs):
        super().motor_run_for_time(time_ms, **kwargs)
        self.n_motor_pulses += 1
        if self.vibrate:
            self.player.add_vibration(self.clock.now_ns())


class FakeRig:
    def __init__(self, *, level=1, mode="survival", target=7, seed=1, calibration=None, source="live",
                 scope="record_session", w_pk=600.0, timing_s=0.0, cards=None, hz=66.0, fps=30.0, lag_s=0.10,
                 stale_ms=300.0, vibration=False, no_motor=False, record_dir=None, spin_probs_fn=None, learner=None,
                 swing_source="imu", no_hub=False, overrides=None, hand_motion=False):
        self.clock = FakeClock(start_ns=1_000_000_000)
        self.origin_ns = self.clock.now_ns()
        camera = swing_source == "pose"                                # the camera, not the hub's gyro, detects swings
        calibration = calibration or Calibration.default(swing_source)
        cards = [(0.6, 2.4, 0)] if cards is None else cards            # the START card, held 1.8 s
        self.player = ScriptedPlayer(calibration.box, w_pk=w_pk, timing_s=timing_s, cards=cards,
                                     pose_motion=camera or hand_motion)    # hand_motion: a hand that swings, hub gyro too
        self.player.origin_ns = self.origin_ns
        self.dev = None if no_hub else VibratingMotor(self.player, self.clock, vibration)
        self.client = FakeMqttClient()
        self.hub_blackouts, self.pose_blackouts = [], []
        self.shake_windows, self.wave_windows = self.player.shake_windows, self.player.wave_windows
        self.pause_reasons_seen = set()
        hub = NoHub() if no_hub else HubLink(self.dev, notify_ms=15, clock=self.clock)
        hub.connect()
        landmarker = FakeLandmarker(self._hand_if_visible, self.clock, lag_s=lag_s, hand=calibration.hand)
        tags = FakeTagDetector(self.player.card_ids, self.clock)
        self.rig = live.assemble(
            hub=hub, capture=FakeCamera(), landmarker=landmarker, calibration=calibration, clock=self.clock,
            tag_detector=tags, mqtt_client=self.client, level=level, mode=mode, target=target, seed=seed,
            source=source, scope=scope, no_motor=no_motor, threaded=False, lag_s=lag_s, gyro_per_dps=GPD,
            accel_per_g=1000.0, fs_raw=32767, stale_ms=stale_ms, to_image=lambda frame: frame,
            record_dir=record_dir, player="fake", spin_probs_fn=spin_probs_fn, learner=learner,
            pose_gyro=posegyro.PoseGyro() if camera else None, overrides=overrides, log=lambda *_: None)
        self.session, self.game = self.rig.session, self.rig.session.game
        self._dt = S // 240
        self._imu_period, self._frame_period, self._pump_period = round(S / hz), round(S / fps), S // 60
        t = self.clock.now_ns()
        self._next_imu, self._next_frame, self._next_pump = t + self._imu_period, t + self._frame_period, t

    # --- time ----------------------------------------------------------------------------------------------
    def now_s(self):
        return (self.clock.now_ns() - self.origin_ns) / S

    def _within(self, windows, t_ns):
        rel = (t_ns - self.origin_ns) / S
        return any(a <= rel < b for a, b in windows)

    def _hand_if_visible(self, t_ns):
        return None if self._within(self.pose_blackouts, t_ns) else self.player.hand_uv(t_ns)

    def step(self):
        self.clock.advance_s(self._dt / S)
        now = self.clock.now_ns()
        self.player.watch(self.game, now)
        while self._next_imu <= now:
            if self.dev is not None and not self._within(self.hub_blackouts, now):   # a silent hub delivers nothing
                self.dev.emit(imu=self.player.imu(now))
            self._next_imu += self._imu_period
        while self._next_frame <= now:
            self.rig.vision.step()
            self._next_frame += self._frame_period
        while self._next_pump <= now:
            self.rig.pump()
            self._next_pump += self._pump_period
        if self.game.paused:
            self.pause_reasons_seen |= self.game.pause_reasons

    def run(self, *, until=None, seconds=None, max_s=300.0):
        """Step until `until()` is true (AssertionError after max_s simulated seconds) or for `seconds`."""
        end = self.clock.now_ns() + round((max_s if seconds is None else seconds) * S)
        while self.clock.now_ns() < end:
            self.step()
            if until is not None and until():
                return True
        if until is not None:
            raise AssertionError(f"not reached within {max_s:.0f} simulated s: phase={self.game.phase} "
                                 f"streak={self.game.tracker.streak} paused={sorted(self.game.pause_reasons)}")
        return None

    def close(self):
        self.rig.close()


class CalibrationScript:
    """A scripted person doing the whole calibration, with the hub mounted at an arbitrary angle.

    Stand still, hold four reach corners, hold the hub upright and turn it side to side (two cycles of +-tilt_deg
    about tilt_axis, to the right first), then soft and full swings (each a backswing lobe followed by the forward
    stroke along u_true).  Everything is reproducible from `seed`.
    """

    def __init__(self, *, u_true=(0.35, 0.88, -0.32), shoulder_w=0.21, hand="right",
                 corners=((-1.1, 0.7), (1.1, 0.7), (1.1, -0.6), (-1.1, -0.6)),
                 soft=(330, 360, 300, 350, 320), full=(1100, 1050, 1200, 1150, 1000), back_ratios=None,
                 tilt_axis=(0.8, 0.6, 0.0), tilt_deg=40.0, gpd=GPD, seed=1, camera=False):
        norm = math.sqrt(sum(c * c for c in u_true))
        self.u_true = tuple(c / norm for c in u_true)
        tilt_norm = math.sqrt(sum(c * c for c in tilt_axis))
        self.tilt_axis, self.tilt_deg = tuple(c / tilt_norm for c in tilt_axis), tilt_deg
        self.camera = camera                                 # the hand itself swings (u_true is then a direction in u, v)
        self._swings = []                                    # camera mode: (forward start, duration, speed, back ratio)
        self.shoulder_w, self.hand, self.corners = shoulder_w, hand, [tuple(c) for c in corners]
        self.soft, self.full, self.gpd = tuple(soft), tuple(full), gpd
        peaks = self.soft + self.full
        back = back_ratios or (0.5,) * len(peaks)
        self.rng = random.Random(seed)
        rest = (0.0, -0.9)
        self._segments, self._lobes = [], []                 # hand moves (t0, t1, from, to); gyro lobes (t0, dur, peak)
        t, pos = 2.2, rest                                   # 2.2 s of standing still
        for corner in self.corners:
            self._segments.append((t, t + 0.6, pos, corner))
            pos, t = corner, t + 2.0                         # 0.6 s to get there, 1.4 s held still
        self._segments.append((t, t + 0.6, pos, rest))
        t += 1.1
        self._tilt_move = (t + 3.8, t + 7.8)                 # 3.8 s upright and still, then 4 s of turning side to side
        t += 7.8 + 0.8
        for peak, b in zip(peaks, back):
            t += 1.2
            self._lobes += [(t, 0.20, -b * peak), (t + 0.22, 0.15, peak)]
            self._swings.append((t + 0.22, 0.15, peak / posegyro.DPS_PER_SW_S, b))
            t += 0.72
        self.duration = t + 1.0

    def hand_uv(self, t):
        u, v = 0.0, -0.9
        for t0, t1, a, b in reversed(self._segments):
            if t >= t0:
                f = _smoothstep((t - t0) / (t1 - t0))
                u, v = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
                break
        if self.camera:
            travel = sum(swing_offset(t, t0, dur, vpk, back=back) for t0, dur, vpk, back in self._swings
                         if abs(t - t0) < 2.0)
            norm = math.hypot(self.u_true[0], self.u_true[1])
            u, v = u + travel * self.u_true[0] / norm, v + travel * self.u_true[1] / norm
        return u + self.rng.gauss(0, 0.004), v + self.rng.gauss(0, 0.004)

    def _tilt_deg(self, t):
        t0, t1 = self._tilt_move
        return self.tilt_deg * math.sin(2 * math.pi * 0.5 * (t - t0)) if t0 <= t < t1 else 0.0

    def imu_raw(self, t):
        rate = sum(peak * math.sin(math.pi * (t - t0) / dur) for t0, dur, peak in self._lobes if t0 <= t <= t0 + dur)
        t0, t1 = self._tilt_move
        turn = self.tilt_deg * math.pi * math.cos(2 * math.pi * 0.5 * (t - t0)) if t0 <= t < t1 else 0.0   # deg/s
        return tuple(round((rate * c + turn * k + self.rng.gauss(0, 2.0)) * self.gpd)
                     for c, k in zip(self.u_true, self.tilt_axis))

    def accel_raw(self, t):
        """Gravity as the hub reads it: along z, but turned the opposite way while the hub is turned about tilt_axis."""
        a, k = math.radians(-self._tilt_deg(t)), self.tilt_axis
        up = (0.0, 0.0, 1.0)
        cross = (k[1] * up[2] - k[2] * up[1], k[2] * up[0] - k[0] * up[2], k[0] * up[1] - k[1] * up[0])
        dot = sum(x * y for x, y in zip(k, up))
        return tuple(round(GRAVITY * (up[i] * math.cos(a) + cross[i] * math.sin(a) + k[i] * dot * (1 - math.cos(a))))
                     for i in range(3))


def drive_calibration(script, flow, *, on_step=None, on_note=None, stop_after_notes=(), imu_hz=66.0, pose_hz=30.0,
                      camera=False):
    """Feed a CalibrationFlow from a CalibrationScript in time order (240 Hz steps).

    camera=True: no hub at all; the swing samples are what PoseGyro makes of the (One-Euro filtered) poses."""
    from pingpong.events import PaddlePose
    from pingpong.oneeuro import OneEuro2D

    t0_ns = 5 * S
    last, t, next_imu, next_pose = None, 0.0, 0.0, 0.0
    smooth, gyro = OneEuro2D(), posegyro.PoseGyro()

    def notes():
        for note in flow.take_notes():
            if on_note:
                on_note(note)
            if any(tag in note for tag in stop_after_notes):
                return True
        return False

    while t <= script.duration:
        if flow.step != last:
            last = flow.step
            if on_step:
                on_step(last)
        if flow.finished():
            break
        now = t0_ns + round(t * S)
        if t >= next_imu and not camera:
            flow.feed_imu(ImuSample(t_ns=now, g=script.imu_raw(t), a=script.accel_raw(t)))
            next_imu += 1.0 / imu_hz
        if t >= next_pose:
            u, v = script.hand_uv(t)
            flow.feed_pose(now, u, v, 0.9, script.shoulder_w + script.rng.gauss(0, 0.002))
            if camera:
                fu, fv = smooth((u, v), now / S)
                sample = gyro.feed(PaddlePose(t_scene_ns=now, u=fu, v=fv, conf=0.9, hand=script.hand))
                if sample is not None:
                    flow.feed_imu(sample)
            next_pose += 1.0 / pose_hz
        if notes():
            return
        t += 1.0 / 240
    notes()
    if flow.step != last and on_step:
        on_step(flow.step)


def wave_u(t):
    """An irregular hand wave (two sines): the hand's horizontal position in shoulder widths."""
    return 0.8 * math.sin(2 * math.pi * 1.1 * t) + 0.5 * math.sin(2 * math.pi * 1.9 * t + 0.7)


def wave_speed(t):
    """How fast that hand is moving (what a gyro would feel), in shoulder widths per second."""
    return abs(0.8 * 2 * math.pi * 1.1 * math.cos(2 * math.pi * 1.1 * t)
               + 0.5 * 2 * math.pi * 1.9 * math.cos(2 * math.pi * 1.9 * t + 0.7))


def _lobe(t, t0, dur, peak):
    return peak * math.sin(math.pi * (t - t0) / dur) if t0 <= t <= t0 + dur else 0.0


def spin_swing_samples(label, rng, *, u_fwd=(1.0, 0.0, 0.0), u_roll=(0.0, 1.0, 0.0), u_up=(0.0, 0.0, 1.0), peak=700.0,
                       gpd=GPD, hz=66.0, t0_ns=5 * S):
    """Raw IMU samples of ONE swing of a spin class: 0.8 s of rest, a backswing, the forward stroke, 0.7 s of rest.

    flat pushes straight through; top rolls the wrist forward-up while lifting (rotation about u_roll,
    acceleration along +u_up); back rolls the other way while chopping down.  Every swing varies a little.
    """
    peak *= rng.uniform(0.85, 1.15)
    roll = {"flat": 0.0, "top": 0.5, "back": -0.5}[label] + rng.gauss(0, 0.08)
    lift_g = {"flat": 0.0, "top": 0.6, "back": -0.6}[label] + rng.gauss(0, 0.1)
    fwd_g = 0.5 + rng.gauss(0, 0.1)
    out = []
    for i in range(int(1.9 * hz)):
        t = i / hz
        fwd = _lobe(t, 1.02, 0.15, peak) - _lobe(t, 0.80, 0.20, 0.4 * peak)
        stroke = _lobe(t, 1.02, 0.15, 1.0)
        g = [fwd * u_fwd[k] + roll * peak * stroke * u_roll[k] + rng.gauss(0, 3.0) for k in range(3)]
        a = [(1000.0 if k == 2 else 0.0) + 1000.0 * stroke * (fwd_g * u_fwd[k] + lift_g * u_up[k]) for k in range(3)]
        out.append(ImuSample(t_ns=t0_ns + round(t * S), g=tuple(round(v * gpd) for v in g),
                             a=tuple(round(v) for v in a)))
    return out


def spin_dataset(n_per_class=12, seed=0, **kw):
    """(features, labels): simulated swings of each spin class, run through the REAL swing detector."""
    from pingpong.swing import SwingDetector, SwingParams

    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_per_class):
        for label in ("flat", "top", "back"):                     # interleaved, like the guided collection
            detector = SwingDetector(SwingParams(u_fwd=kw.get("u_fwd", (1.0, 0.0, 0.0)), gyro_per_dps=GPD, t_pk=250.0))
            impacts = [e for s in spin_swing_samples(label, rng, **kw) for e in detector.feed(s) if e.kind == "IMPACT"]
            if impacts:
                X.append(impacts[-1].feat)
                y.append(label)
    return X, y


class SpinScript:
    """A scripted player doing one spin swing after another, as a function of time (for the guided collection).

    `labels[k]` is the class of the k-th swing; swings are `spacing_s` apart after `lead_s` of rest (the
    detector needs half a second of quiet to warm up).  With `scramble=True` the swings come out in a
    shuffled order, so they no longer match what the collection tool asked for.
    """

    def __init__(self, labels, *, seed=0, spacing_s=2.4, lead_s=1.6, u_fwd=(1.0, 0.0, 0.0), u_roll=(0.0, 1.0, 0.0),
                 u_up=(0.0, 0.0, 1.0), peak=700.0, gpd=GPD, scramble=False):
        self.rng = random.Random(seed)
        self.labels = list(labels)
        if scramble:
            self.rng.shuffle(self.labels)
        self.spacing_s, self.lead_s, self.gpd = spacing_s, lead_s, gpd
        self.u_fwd, self.u_roll, self.u_up = u_fwd, u_roll, u_up
        self.params = [self._draw(label, peak) for label in self.labels]
        self.duration = lead_s + spacing_s * len(self.labels) + 1.0

    def _draw(self, label, peak):
        rng = self.rng
        return {"peak": peak * rng.uniform(0.85, 1.15), "fwd_g": 0.5 + rng.gauss(0, 0.1),
                "roll": {"flat": 0.0, "top": 0.5, "back": -0.5}[label] + rng.gauss(0, 0.08),
                "lift_g": {"flat": 0.0, "top": 0.6, "back": -0.6}[label] + rng.gauss(0, 0.1)}

    def imu_raw(self, t):
        """Raw (ax, ay, az, gx, gy, gz) at time t."""
        k = int((t - self.lead_s) // self.spacing_s) if t >= self.lead_s else -1
        g, a = [0.0, 0.0, 0.0], [0.0, 0.0, 1000.0]
        if 0 <= k < len(self.params):
            p, tt = self.params[k], (t - self.lead_s) - k * self.spacing_s
            fwd = _lobe(tt, 0.22, 0.15, p["peak"]) - _lobe(tt, 0.0, 0.20, 0.4 * p["peak"])
            stroke = _lobe(tt, 0.22, 0.15, 1.0)
            for i in range(3):
                g[i] = fwd * self.u_fwd[i] + p["roll"] * p["peak"] * stroke * self.u_roll[i]
                a[i] += 1000.0 * stroke * (p["fwd_g"] * self.u_fwd[i] + p["lift_g"] * self.u_up[i])
        g = [v + self.rng.gauss(0, 3.0) for v in g]
        return (round(a[0]), round(a[1]), round(a[2]), *(round(v * self.gpd) for v in g))


def spin_stream(labels, *, hz=66.0, t0_ns=5 * S, **kw):
    """Every IMU sample of a SpinScript, in order: what the hub would deliver."""
    script = SpinScript(labels, **kw)
    out = []
    for i in range(int(script.duration * hz)):
        t = i / hz
        raw = script.imu_raw(t)
        out.append(ImuSample(t_ns=t0_ns + round(t * S), g=raw[3:], a=raw[:3]))
    return out, script
