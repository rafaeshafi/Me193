"""CalibrationFlow: the guided calibration as a pure state machine (the window lives in tools/).

    stand    stand still, arms down: shoulder width (feeds the one-player pose lock)
    corners  hold the paddle still at four corners of your comfortable reach (no keyboard: you
             are 1.8 m from the laptop, so a held hand is the "click")
    soft     5 soft swings   -> omega_lo (and, with T_PK = 0.7 * omega_lo, the weakest swing that counts)
    full     5 full swings   -> omega_hi (a full swing = top speed)
    done     forward axis (SVD of the peak vectors), strengths, reach box -> a Calibration

Swings are cut out of the raw gyro stream by magnitude, with 0.8 s of lead-in so the swing
detector's warm-up has data when the finished calibration is checked against its own takes.
The flow only ever sees samples and poses, so it runs the same on hardware, on a recording
and in tests.
"""

import math
from collections import deque
from statistics import median

import numpy as np

from pingpong import calibration as cal
from pingpong.paddle import ReachBox
from pingpong.profile import Calibration
from pingpong.shake import _reversals
from pingpong.swing import SwingDetector

S = 1_000_000_000
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


class CalibrationError(Exception):
    pass


class CalibrationFlow:
    def __init__(self, *, gyro_per_dps, hand="right", accel_per_g=1000.0, fs_raw=32767, n_soft=5, n_full=5,
                 stand_s=1.0, hold_s=0.8, still_sw=0.08, min_corner_gap_sw=0.35, min_span=(0.6, 0.4),
                 swing_start_dps=80.0, swing_end_dps=40.0, quiet_s=0.25, min_peak_dps=150.0, max_take_s=3.0,
                 lead_s=0.8, max_reversals=3, source="imu"):
        self.gpd, self.hand, self.accel_per_g, self.fs_raw = gyro_per_dps, hand, accel_per_g, fs_raw
        self.source = source                                     # "pose": the samples are the camera's hand speed
        self.n_soft, self.n_full = n_soft, n_full
        self.stand_ns, self.hold_ns, self.still_sw, self.min_gap = round(stand_s * S), round(hold_s * S), still_sw, min_corner_gap_sw
        self.min_span = min_span
        self.start_dps, self.end_dps, self.quiet_ns = swing_start_dps, swing_end_dps, round(quiet_s * S)
        self.min_peak, self.max_take_ns, self.lead_ns = min_peak_dps, round(max_take_s * S), round(lead_s * S)
        self.max_reversals = max_reversals
        self.step = "stand"
        self.corners, self.soft, self.full = [], [], []         # soft / full: [(peak_dps, [ImuSample, ...])]
        self.self_check = None                                  # (swings the new calibration detects, swings taken)
        self._shoulders, self._shoulder_w, self._window = [], None, deque()
        self._rest = None                                        # where the hand hangs: never a reach corner
        self._bias, self._lead, self._take = None, deque(), None
        self._take_start = self._last_loud = 0
        self._notes, self._calibration = [], None

    # --- what to tell the player ---------------------------------------------------------------------
    def prompt(self):
        if self.step == "stand":
            return "Stand still, arms down, facing the camera: measuring your shoulders"
        if self.step == "corners":
            name = CORNER_NAMES[len(self.corners)].upper()
            return f"Hold the hub at the {name} corner of where you can comfortably reach, and keep still"
        if self.step == "soft":
            return f"SOFT swing {len(self.soft) + 1} of {self.n_soft}: easy, like returning a gentle ball"
        if self.step == "full":
            return f"FULL swing {len(self.full) + 1} of {self.n_full}: as hard as you would smash"
        return "Calibration complete"

    def progress(self):
        return {"stand": (0, 1), "corners": (len(self.corners), 4), "soft": (len(self.soft), self.n_soft),
                "full": (len(self.full), self.n_full), "done": (1, 1)}[self.step]

    def take_notes(self):
        notes, self._notes = self._notes, []
        return notes

    def finished(self):
        return self.step == "done"

    def calibration(self):
        if self._calibration is None:
            raise CalibrationError(f"calibration is not finished yet (step: {self.step})")
        return self._calibration

    # --- poses: stand, corners ------------------------------------------------------------------------
    def feed_pose(self, t_ns, u, v, conf, shoulder_w):
        if conf < 0.6:
            return
        if self.step == "stand":
            self._stand(t_ns, u, v, shoulder_w)
        elif self.step == "corners":
            self._corner(t_ns, u, v)

    def _stand(self, t_ns, u, v, shoulder_w):
        if shoulder_w is None:
            return
        self._shoulders.append((t_ns, shoulder_w, u, v))
        self._shoulders = [row for row in self._shoulders if row[0] >= t_ns - 2 * self.stand_ns]
        recent = [row for row in self._shoulders if row[0] >= t_ns - self.stand_ns]
        if t_ns - self._shoulders[0][0] < self.stand_ns or len(recent) < 15:
            return
        widths = [row[1] for row in recent]
        mid = median(widths)
        if (max(widths) - min(widths)) / mid <= 0.15:            # standing still
            self._shoulder_w, self.step = mid, "corners"
            self._rest = (median(row[2] for row in recent), median(row[3] for row in recent))
            self._note(f"shoulders measured: {mid:.3f} of the frame width")

    def _corner(self, t_ns, u, v):
        self._window.append((t_ns, u, v))
        while self._window and self._window[0][0] < t_ns - self.hold_ns:
            self._window.popleft()
        if t_ns - self._window[0][0] < 0.9 * self.hold_ns:
            return
        us, vs = [p[1] for p in self._window], [p[2] for p in self._window]
        if max(us) - min(us) > self.still_sw or max(vs) - min(vs) > self.still_sw:
            return                                               # still moving
        cu, cv = sum(us) / len(us), sum(vs) / len(vs)
        taken = self.corners + ([self._rest] if self._rest is not None else [])
        if any(math.hypot(cu - a, cv - b) < self.min_gap for a, b in taken):
            return                                               # already taken, or just a hand hanging at rest
        name = CORNER_NAMES[len(self.corners)]
        self.corners.append((cu, cv))
        self._window.clear()
        self._note(f"corner {len(self.corners)}/4 captured ({name})")
        if len(self.corners) == 4:
            self._check_box()

    def _check_box(self):
        us, vs = [c[0] for c in self.corners], [c[1] for c in self.corners]
        width, height = max(us) - min(us), max(vs) - min(vs)
        if width < self.min_span[0] or height < self.min_span[1]:
            self._note(f"reach box too small ({width:.2f} x {height:.2f} shoulder widths): reach further out "
                       "to each corner")
            self.corners = []
        else:
            self.step = "soft"

    # --- IMU: swings ------------------------------------------------------------------------------------
    def feed_imu(self, sample):
        g = np.array(sample.g, dtype=float) / self.gpd
        if self._bias is None:
            self._bias = g
        mag = float(np.linalg.norm(g - self._bias))
        if mag < 25.0:                                           # resting: keep the zero-rate offset fresh
            self._bias = self._bias + 0.02 * (g - self._bias)
        self._lead.append(sample)
        while self._lead and self._lead[0].t_ns < sample.t_ns - self.lead_ns:
            self._lead.popleft()
        if self.step not in ("soft", "full"):
            self._take = None
            return
        t = sample.t_ns
        if self._take is None:
            if mag > self.start_dps:
                self._take, self._take_start, self._last_loud = list(self._lead), t, t
            return
        self._take.append(sample)
        if mag > self.end_dps:
            self._last_loud = t
        if t - self._take_start > self.max_take_ns:
            self._take = None
            self._note("that went on too long: one clean swing at a time, then rest")
        elif t - self._last_loud >= self.quiet_ns:
            self._finish_take()

    def _finish_take(self):
        samples, self._take = self._take, None
        vectors = np.array([s.g for s in samples], dtype=float) / self.gpd - self._bias
        norms = np.linalg.norm(vectors, axis=1)
        k = int(np.argmax(norms))
        peak = float(norms[k])
        if peak < self.min_peak:
            self._note(f"that swing was too gentle ({peak:.0f} dps): swing a bit firmer")
            return
        along = vectors @ (vectors[k] / peak)
        if _reversals(along, 0.25 * peak) > self.max_reversals:
            self._note("that was waving, not a swing: one clean swing at a time")
            return
        kind, takes, total = ("soft", self.soft, self.n_soft) if self.step == "soft" else ("full", self.full, self.n_full)
        takes.append((peak, samples))
        self._note(f"{kind} swing {len(takes)}/{total}: peak {peak:.0f} dps")
        if len(takes) == total:
            self._next_after(kind)

    def _next_after(self, kind):
        if kind == "soft":
            self.step = "full"
        else:
            self._finalize()

    def _finalize(self):
        takes = [samples for _, samples in self.soft + self.full]
        axis, _ = cal.fit_forward_axis(takes, self.gpd)
        # Strength = the FORWARD peak on the learned axis, not the biggest gyro magnitude: a real soft swing's
        # backswing is about as big as its stroke, so the biggest lobe can be the wrong one.
        n_soft = len(self.soft)
        forward = cal.forward_peaks(takes, axis, self.gpd, bias=self._bias)
        try:
            strengths = cal.swing_strengths(forward[:n_soft], forward[n_soft:])
        except ValueError as exc:
            self._note(f"{exc}; repeat the full swings, harder")
            self.full = []
            return
        calibration = Calibration(
            swing=cal.SwingCalibration(u_fwd=axis, omega_lo=strengths["omega_lo"], omega_hi=strengths["omega_hi"],
                                       source=self.source),
            box=ReachBox.fit(self.corners), shoulder_w=self._shoulder_w, hand=self.hand)
        self.self_check = (self._detected(calibration, takes), len(takes))
        self._calibration, self.step = calibration, "done"
        self._note(f"calibration done: the swing detector recognises {self.self_check[0]} of "
                   f"{self.self_check[1]} of your calibration swings")

    def _detected(self, calibration, takes):
        count = 0
        for samples in takes:
            detector = SwingDetector(calibration.swing_params(self.gpd, self.accel_per_g, self.fs_raw))
            count += any(e.kind == "IMPACT" for s in samples for e in detector.feed(s))
        return count

    def _note(self, text):
        self._notes.append(text)
