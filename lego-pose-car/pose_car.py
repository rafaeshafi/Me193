"""Drive a LEGO Education car with your arms.

A webcam feeds MediaPipe's PoseLandmarker; each arm's height above or below
the shoulder becomes a tank-drive speed for the motor on that side, which is
sent to the LEGO Double Motor over Bluetooth Low Energy.

    arm straight up    ->  that wheel full forward
    arm held out (T)   ->  that wheel stopped
    arm down at side   ->  that wheel full reverse

Both arms up = drive forward. One up, one down = spin in place.

Run `python pose_car.py --no-motor` to see the vision half work with no
hardware attached.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"

# --- Landmark indices we care about (MediaPipe's own numbering) -------------
# These are ANATOMICAL: LEFT_WRIST is the person's left wrist, whichever way
# the image is mirrored. The car drives away from the driver, so the driver's
# left arm and the car's left wheel are on the same side.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

# --- Tuning ----------------------------------------------------------------
MAX_SPEED = 70        # motor % at full arm extension; 100 is a lot in a hallway
DEADZONE = 0.15       # arm heights within this of shoulder level read as "stop"
FULL_SCALE = 0.85     # arm height (in shoulder-widths) that means MAX_SPEED
SMOOTHING = 0.35      # EMA weight on each new reading; lower = smoother, laggier
MIN_VISIBILITY = 0.5  # below this the landmark is a guess, so we stop instead
MIN_SHOULDER_FRAC = 0.08  # shoulders narrower than this fraction of the frame = bad read
LOST_POSE_GRACE = 0.4 # seconds a dropped pose is tolerated before stopping
SEND_INTERVAL = 0.08  # seconds between BLE writes (~12/s)
SEND_DELTA = 4        # don't spend a BLE write on a change smaller than this


def clamp(value, low, high):
    return max(low, min(high, value))


def arm_speed(shoulder, wrist, shoulder_width):
    """Map one arm's height relative to its shoulder onto a motor speed.

    Dividing by the shoulder width makes this scale-invariant: standing closer
    to the camera makes every pixel distance bigger, but their ratio holds.
    Image y grows downward, so a raised wrist has the SMALLER y.
    """
    offset = (shoulder[1] - wrist[1]) / shoulder_width
    if abs(offset) < DEADZONE:
        return 0.0
    span = FULL_SCALE - DEADZONE
    scaled = (abs(offset) - DEADZONE) / span
    return math.copysign(clamp(scaled, 0.0, 1.0) * MAX_SPEED, offset)


def read_arms(landmarks, width, height):
    """Return ((left_speed, right_speed), points) or None if the pose is unusable."""
    needed = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
    if any(landmarks[i].visibility < MIN_VISIBILITY for i in needed):
        return None

    # Normalized coords are fractions of width/height, so they must be scaled
    # back into pixels before any distance between them means anything.
    pts = {i: (landmarks[i].x * width, landmarks[i].y * height) for i in needed}

    # Every measurement below is divided by this, so a bad value doesn't just
    # add noise, it amplifies it. A driver standing side-on or far away has
    # near-overlapping shoulders, which would turn a level arm into full speed.
    shoulder_width = math.dist(pts[L_SHOULDER], pts[R_SHOULDER])
    if shoulder_width < MIN_SHOULDER_FRAC * width:
        return None  # side-on or too far away: refuse to guess

    left = arm_speed(pts[L_SHOULDER], pts[L_WRIST], shoulder_width)
    right = arm_speed(pts[R_SHOULDER], pts[R_WRIST], shoulder_width)
    return (left, right), pts


class Car:
    """The LEGO Double Motor, or a stand-in for it when --no-motor is used.

    Rate-limits BLE writes and sends them non-blocking, so the camera loop
    never waits on Bluetooth.
    """

    def __init__(self, *, enabled=True, card_color=None, card_serial=None):
        self.enabled = enabled
        self.card_color = card_color
        self.card_serial = card_serial
        self.motor = None
        self._last_sent = None
        self._last_send_time = 0.0

    def connect(self):
        if not self.enabled:
            print("--no-motor: running vision only, no Bluetooth.")
            return True

        import legoeducation as le

        # With several cars in one room a bare connect() grabs whichever motor
        # answers first, which may be someone else's. The Connection Card that
        # ships with the motor names one specific unit.
        target = {}
        if self.card_color is not None:
            # Only some LEGO colours appear on Connection Cards, so check
            # against that set rather than letting a typo raise mid-demo.
            valid = {le.LEGO_COLOR_NAME_MAP[c].removeprefix("LEGO_COLOR_").lower(): c
                     for c in le.CARD_COLORS}
            key = self.card_color.lower()
            if key not in valid:
                print(f"Unknown card colour {self.card_color!r}. "
                      f"Expected one of: {', '.join(sorted(valid))}")
                return False
            target["card_color"] = valid[key]
        if self.card_serial is not None:
            target["card_serial"] = self.card_serial

        described = ", ".join(f"{k}={v}" for k, v in target.items()) or "any"
        print(f"Scanning for a Double Motor over BLE ({described})...")
        self.motor = le.DoubleMotor()
        self.motor.connect(**target)
        if not self.motor.connected:
            print("Could not connect. Is the Double Motor powered on and nearby?")
            return False
        print("Connected.")
        return True

    def drive(self, left, right):
        left, right = int(round(left)), int(round(right))
        now = time.monotonic()
        stopping = (left, right) == (0, 0)

        if self._last_sent is not None:
            moved = max(abs(left - self._last_sent[0]), abs(right - self._last_sent[1]))
            # A stop always goes out immediately; everything else waits its turn.
            if not stopping and (moved < SEND_DELTA or now - self._last_send_time < SEND_INTERVAL):
                return
            if self._last_sent == (left, right):
                return

        self._last_sent = (left, right)
        self._last_send_time = now
        if self.motor is None:
            return
        # blocking=False: fire the command and get straight back to the camera.
        self.motor.movement_move_tank(left, right, blocking=False)

    def stop(self):
        self._last_sent = None  # force the next drive() call through
        if self.motor is not None:
            self.motor.movement_stop(blocking=False)

    def close(self):
        if self.motor is None:
            return
        for teardown in (self.motor.movement_stop, self.motor.disconnect):
            try:
                teardown()
            except Exception:
                pass  # link may already be gone; keep tearing down regardless


def draw_hud(frame, pts, left, right, armed, have_pose):
    height, width = frame.shape[:2]

    if pts is not None:
        for shoulder, elbow, wrist in ((L_SHOULDER, L_ELBOW, L_WRIST),
                                       (R_SHOULDER, R_ELBOW, R_WRIST)):
            chain = [tuple(map(int, pts[i])) for i in (shoulder, elbow, wrist)]
            cv2.line(frame, chain[0], chain[1], (0, 220, 255), 3)
            cv2.line(frame, chain[1], chain[2], (0, 220, 255), 3)
            for point in chain:
                cv2.circle(frame, point, 6, (255, 255, 255), -1)
        cv2.line(frame, tuple(map(int, pts[L_SHOULDER])),
                 tuple(map(int, pts[R_SHOULDER])), (0, 220, 255), 2)

    # Speed bars: left arm on the left edge, right arm on the right edge. The
    # preview is mirrored, so those line up with the driver's own arms.
    bar_top, bar_height, bar_width = 60, height - 140, 26
    mid = bar_top + bar_height // 2
    for x, speed, label in ((24, left, "L"), (width - 24 - bar_width, right, "R")):
        cv2.rectangle(frame, (x, bar_top), (x + bar_width, bar_top + bar_height),
                      (70, 70, 70), 1)
        cv2.line(frame, (x, mid), (x + bar_width, mid), (70, 70, 70), 1)
        filled = int(speed / MAX_SPEED * (bar_height // 2))
        color = (0, 200, 0) if speed >= 0 else (0, 120, 255)
        cv2.rectangle(frame, (x, mid), (x + bar_width, mid - filled), color, -1)
        cv2.putText(frame, f"{label}{int(speed):+4d}", (x - 6, bar_top + bar_height + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if armed:
        status, status_color = "DRIVING", (0, 220, 0)
    else:
        status, status_color = "disarmed - press d to drive", (0, 180, 255)
    cv2.putText(frame, status, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    if not have_pose:
        cv2.putText(frame, "no pose - stopped", (24, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
    else:
        cv2.putText(frame, "arms up = forward | out = stop | down = reverse",
                    (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-motor", action="store_true",
                        help="skip Bluetooth; just show what the car would do")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--card-color", metavar="COLOR",
                        help="Connection Card colour, e.g. azure (targets one specific motor)")
    parser.add_argument("--card-serial", metavar="SERIAL",
                        help="Connection Card serial, e.g. 3683")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Missing pose model: {MODEL_PATH}\nSee README.md for the download command.")
        return 1

    car = Car(enabled=not args.no_motor,
              card_color=args.card_color, card_serial=args.card_serial)
    if not car.connect():
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.")
        car.close()
        return 1

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,  # VIDEO mode tracks between frames
        num_poses=1,
    )

    armed = False
    left = right = 0.0
    last_pose_time = 0.0
    start = time.monotonic()

    print("Press 'd' to arm/disarm the motors, 'q' or Esc to quit.")
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Camera stopped delivering frames.")
                    break

                # Mirror so the preview behaves like a mirror for the driver.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                now = time.monotonic()
                result = landmarker.detect_for_video(image, int((now - start) * 1000))

                reading = None
                if result.pose_landmarks:
                    reading = read_arms(result.pose_landmarks[0], width, height)

                pts = None
                if reading is not None:
                    (target_left, target_right), pts = reading
                    last_pose_time = now
                elif now - last_pose_time < LOST_POSE_GRACE:
                    # Brief dropout (a blink of the detector) - hold the last speeds.
                    target_left, target_right = left, right
                else:
                    target_left = target_right = 0.0

                left += (target_left - left) * SMOOTHING
                right += (target_right - right) * SMOOTHING
                # Kill the EMA's long tail so "stop" actually reaches zero.
                if abs(left) < 1:
                    left = 0.0
                if abs(right) < 1:
                    right = 0.0

                if armed:
                    car.drive(left, right)

                have_pose = now - last_pose_time < LOST_POSE_GRACE
                draw_hud(frame, pts, left, right, armed, have_pose)
                cv2.imshow("pose car", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("d"):
                    armed = not armed
                    if not armed:
                        car.stop()
                    print("motors armed" if armed else "motors disarmed")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        car.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
