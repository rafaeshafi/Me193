"""Where the time goes: every stage between a hand moving and the player seeing, hearing or feeling the answer.

    hand -> hub IMU -> BLE bursts -> arrival stamp          imu_s      (the hub sends no timestamps: a sample is stamped
                                                                       when it ARRIVES; the camera's poses are aligned to
                                                                       that clock by CAMERA_LAG_S, bench_cam measures it)
    peak of the gyro's rate -> end of the forward stroke    stroke_s   (where the player means the paddle to meet the ball;
                                                                       28 real swings: median 0.20 s, quartiles 0.14-0.24)
    frame drawn -> light from the screen                    display_s  (+ up to loop_s: the loop runs at ~60 Hz)
    sound written -> heard                                  audio_s
    motor command written -> the hub's motors move          haptic_s

Only the camera's lag is measured here (against the hub); the rest are typical values with a basis, tunable in
config_local.json or live with `--set latency.display_s=0.08`.  Compensating means placing every effect where it will
be PERCEIVED: the ball (a deterministic flight) is drawn where it will be when the light reaches the eye, the hand is
drawn where it will be then, the swing is dated by when the player did it, and the thump and the sound are sent early
so they arrive with the picture.
"""

import math
from dataclasses import dataclass

import config

MAX_SPEED_SW_S = 5.0        # the hand is never extrapolated faster than this (a glitch must not fling the paddle)
MAX_LEAD_S = 0.25           # ... nor further ahead than this (or than a trained model is asked for)
MAX_AGE_S = 0.4             # a reading older than this is not extrapolated at all
FIT_S = 0.15                # the hand's velocity is fitted over this much of its latest history
GAIN = 0.0                  # no extrapolation unless asked: on the player's recorded tracks, extrapolating 0.17 s ahead by the
                            # recent speed was no more accurate than holding the hand (+3% error) and shimmered 4x as much


@dataclass(frozen=True)
class Latency:
    imu_s: float = 0.040
    stroke_s: float = 0.20
    display_s: float = 0.050
    audio_s: float = 0.025
    haptic_s: float = 0.050
    loop_s: float = 1 / 60

    def __post_init__(self):
        if min(self.imu_s, self.stroke_s, self.display_s, self.audio_s, self.haptic_s, self.loop_s) < 0:
            raise ValueError("a delay cannot be negative")

    @classmethod
    def from_config(cls):
        return cls(imu_s=config.LAT_IMU_S, stroke_s=config.LAT_STROKE_S, display_s=config.LAT_DISPLAY_S,
                   audio_s=config.LAT_AUDIO_S, haptic_s=config.LAT_HAPTIC_S)

    @property
    def contact_lag_s(self):
        """From a swing's peak, as stamped on arrival, to the contact in the same clock: the stroke's length less the
        time the hub took to tell us (the stamp is already that late)."""
        return self.stroke_s - self.imu_s

    @property
    def view_ahead_s(self):
        """How far ahead of now a frame is drawn, so that when the light reaches the eye it shows the world as it is then."""
        return self.display_s + self.loop_s / 2


def predict_hand(poses, now_ns, lat, *, min_conf=0.5, gain=GAIN, model=None):
    """Where the hand will be when the frame being drawn now reaches the eye, from its latest readings: (u, v) or None.

    The newest reading is its own age old (the camera, the pose model and the loop), was taken imu_s before its stamp
    says (the stamps are on the hub's arrival clock) and will be seen display_s from now.  What to do about that gap:
    a trained `model` (posemodel.HandPredictor) predicts it; failing that `gain` > 0 extrapolates the hand's velocity over
    the last FIT_S, capped and damped; and by default (gain 0) the hand is drawn where it was last read."""
    good = [p for p in poses if p.conf >= min_conf]
    if not good:
        return None
    last = good[-1]
    age = (now_ns - last.t_scene_ns) / 1e9
    lead = min(MAX_LEAD_S, max(0.0, age) + lat.imu_s + lat.view_ahead_s)
    if model is not None and age <= MAX_AGE_S:
        answer = model.predict(good, lead)
        if answer is not None:
            return answer
    recent = [p for p in good if p.t_scene_ns >= last.t_scene_ns - round(FIT_S * 1e9)]
    if gain <= 0.0 or age > MAX_AGE_S or len(recent) < 2:
        return last.u, last.v
    ts = [(p.t_scene_ns - last.t_scene_ns) / 1e9 for p in recent]
    mean = sum(ts) / len(ts)
    spread = sum((t - mean) ** 2 for t in ts)
    if spread < 1e-9:
        return last.u, last.v
    vu = sum((t - mean) * p.u for t, p in zip(ts, recent)) / spread
    vv = sum((t - mean) * p.v for t, p in zip(ts, recent)) / spread
    speed = math.hypot(vu, vv)
    if speed > MAX_SPEED_SW_S:
        vu, vv = vu * MAX_SPEED_SW_S / speed, vv * MAX_SPEED_SW_S / speed
    return last.u + gain * vu * lead, last.v + gain * vv * lead
