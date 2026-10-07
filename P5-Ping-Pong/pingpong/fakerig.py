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

from pingpong import live
from pingpong.clock import FakeClock
from pingpong.hub import HubLink
from pingpong.profile import Calibration
from pingpong.sources_fake import (FakeCamera, FakeDoubleMotor, FakeLandmarker, FakeMqttClient,
                                   FakeTagDetector)

S = 1_000_000_000
GPD = 10.0                    # raw gyro counts per deg/s of the fake hub (passed to assemble explicitly)
GRAVITY = 1000                # raw accelerometer counts at rest (z)
VIBRATION_DPS = 700.0         # a motor pulse "shakes" the fake IMU like a swing, unless it is blanked


def _smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class ScriptedPlayer:
    """Glides the hand to each ball's arrival point and swings so the gyro peaks at the ball's t_c."""

    def __init__(self, box, *, w_pk=600.0, swing_s=0.15, timing_s=0.0, rest_uv=(0.0, -0.4), cards=()):
        self.box, self.w_pk, self.swing_s, self.timing_s = box, w_pk, swing_s, timing_s
        self.cards = [(round(a * S), round(b * S), tag) for a, b, tag in cards]    # relative to rig start
        self.origin_ns = 0
        self._segments = [(0, 0, rest_uv, rest_uv)]          # (t0, t1, from_uv, to_uv)
        self._peaks = {}                                     # ball_id -> gyro peak time
        self._vibrations = []
        self._now = 0

    # --- decisions (called every step with the game state) -------------------------------------------
    def watch(self, game, now_ns):
        self._now = now_ns
        ball = game.incoming
        if ball is None or game.paused:
            return
        peak = ball.t_c_ns + round(self.timing_s * S)
        planned = self._peaks.get(ball.ball_id)
        if planned == peak:
            return
        if planned is None:                                  # first sight of this ball: head for its arrival point
            glide = min(0.43, 0.5 * max(0.0, (ball.t_c_ns - now_ns) / S))
            self._segments.append((now_ns, now_ns + round(glide * S), self.hand_uv(now_ns),
                                   self.box.to_uv(*ball.aim_ab)))
        self._peaks[ball.ball_id] = peak                     # (re)plan the swing: a pause moves t_c

    @property
    def n_swings(self):
        """Swings whose gyro peak has happened by now."""
        return sum(1 for peak in self._peaks.values() if peak <= self._now)

    def add_vibration(self, start_ns):
        if all(abs(start_ns - v) > 1_000_000 for v in self._vibrations):    # both motors of one pulse: once
            self._vibrations.append(start_ns)

    # --- what the sensors see ------------------------------------------------------------------------
    def hand_uv(self, t_ns):
        for t0, t1, a, b in reversed(self._segments):
            if t_ns >= t0:
                f = 1.0 if t1 <= t0 else _smoothstep((t_ns - t0) / (t1 - t0))
                return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
        return self._segments[0][2]

    def imu(self, t_ns):
        """Raw hub reading (ax, ay, az, gx, gy, gz): half-sine pulses on the forward (x) gyro axis."""
        gx = 0.0
        for peak in self._peaks.values():
            gx += self._pulse(t_ns, peak - self.swing_s / 2 * S, self.w_pk)
        for start in self._vibrations:
            gx += self._pulse(t_ns, start + 0.010 * S, VIBRATION_DPS)
        return 0, 0, GRAVITY, round(gx * GPD), 0, 0

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
                 stale_ms=300.0, vibration=False, no_motor=False):
        self.clock = FakeClock(start_ns=1_000_000_000)
        self.origin_ns = self.clock.now_ns()
        calibration = calibration or Calibration.default()
        cards = [(0.6, 2.4, 0)] if cards is None else cards            # the START card, held 1.8 s
        self.player = ScriptedPlayer(calibration.box, w_pk=w_pk, timing_s=timing_s, cards=cards)
        self.player.origin_ns = self.origin_ns
        self.dev = VibratingMotor(self.player, self.clock, vibration)
        self.client = FakeMqttClient()
        self.hub_blackouts, self.pose_blackouts = [], []
        self.pause_reasons_seen = set()
        hub = HubLink(self.dev, notify_ms=15, clock=self.clock)
        hub.connect()
        landmarker = FakeLandmarker(self._hand_if_visible, self.clock, lag_s=lag_s, hand=calibration.hand)
        tags = FakeTagDetector(self.player.card_ids, self.clock)
        self.rig = live.assemble(
            hub=hub, capture=FakeCamera(), landmarker=landmarker, calibration=calibration, clock=self.clock,
            tag_detector=tags, mqtt_client=self.client, level=level, mode=mode, target=target, seed=seed,
            source=source, scope=scope, no_motor=no_motor, threaded=False, lag_s=lag_s, gyro_per_dps=GPD,
            accel_per_g=1000.0, fs_raw=32767, stale_ms=stale_ms, to_image=lambda frame: frame,
            log=lambda *_: None)
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
            if not self._within(self.hub_blackouts, now):             # a silent hub delivers nothing at all
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
