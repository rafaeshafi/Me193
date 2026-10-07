"""PoseGyro: the camera as a virtual gyro, the swing source when the hub's IMU is too slow.

The Double Motor's gyro is the swing sensor.  If the bench finds it delivering fewer than 25 samples a
second (a swing lasts ~150 ms, so that is two or three samples) the very same swing detector can run on the
HAND'S VELOCITY from the camera instead: every accepted pose becomes one pseudo-IMU sample whose gyro vector is
the hand's velocity in the camera plane, scaled so one shoulder width per second reads as DPS_PER_SW_S "deg/s".
Everything downstream (signed forward axis from the calibration swings, FFT shake lock, judge, recorder) is the
code that runs on the hub's samples, and never knows the difference except through `src="pose"`.

What is lost: a swing toward the camera barely moves the hand in the picture, a fast sideways reposition looks
like a swing (the duration limits and the judge's timing and pose gates are what stop most of those), the frame
rate is 15-30 Hz, and there is no accelerometer, so no spin.  It is a fallback, not an equal.
"""

from pingpong.events import ImuSample

DPS_PER_SW_S = 100.0          # pseudo deg/s per (shoulder width / s) of hand speed
GYRO_PER_DPS = 10.0           # raw counts per pseudo deg/s: the hub's hypothesis, so the detector's limits carry over
ACCEL_PER_G = 1000.0
FS_RAW = 32767
DELAY_S = 0.03                # the filtered hand speed peaks about this long after the hand's real peak
MAX_GAP_S = 0.25              # tracking lost for longer than this: the jump across the gap is not a speed


class PoseGyro:
    def __init__(self, *, dps_per_sw_s=DPS_PER_SW_S, max_gap_s=MAX_GAP_S, delay_s=DELAY_S):
        self.scale = dps_per_sw_s * GYRO_PER_DPS
        self.max_gap_ns, self.delay_ns = round(max_gap_s * 1e9), round(delay_s * 1e9)
        self._prev = None                                 # (t_scene_ns, u, v)

    def feed(self, pose):
        """-> an ImuSample for this pose, or None if it is not newer than the last one."""
        prev, self._prev = self._prev, (pose.t_scene_ns, pose.u, pose.v)
        if prev is not None and pose.t_scene_ns <= prev[0]:
            self._prev = prev
            return None
        g = (0, 0, 0)
        if prev is not None and pose.t_scene_ns - prev[0] <= self.max_gap_ns:
            dt = (pose.t_scene_ns - prev[0]) / 1e9
            g = (self._raw((pose.u - prev[1]) / dt), self._raw((pose.v - prev[2]) / dt), 0)
        return ImuSample(t_ns=pose.t_scene_ns - self.delay_ns, g=g, a=(0, 0, round(ACCEL_PER_G)), src="pose")

    def _raw(self, speed_sw_s):
        return max(-FS_RAW, min(FS_RAW, round(speed_sw_s * self.scale)))
