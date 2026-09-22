"""
P2: drive the trike back and forth until an AprilTag sits in the middle of
a camera's view -- a PD controller on the tag's position in the image.

    python3 PD.py [source] [--mode damped|spring] [--kp K] [--kd K]
                  [--sign 1|-1] [--dry-run]

<source> is a camera index (default 0), a video file, or a stream URL, e.g.
a phone camera app (http://<phone-ip>:8080/video). The same code covers both
set-ups -- the laptop camera watching a tag on the car, or a phone on the car
watching a fixed tag -- because at start-up it drives a short pulse and
measures which way the tag moves in the image (the "sign check") instead of
assuming it. --sign skips that check; --dry-run runs without the robot.

THE POLICY
----------
    e     = (tag_x - W/2) / (W/2)        -1 at left edge, 0 centre, +1 right edge
    de    = de/dt from frame timestamps, low-pass filtered
    speed = -sign * 100 * (KP*e + KD*de), clipped to +/-MAX_SPEED

P: speed proportional to distance from centre -- fast when far, slow when
close. D: opposes the tag's motion across the image, braking on the way in.
Speeds too small to turn the motors are raised to MIN_SPEED. Inside
CENTER_TOL with the tag nearly still, the car stops and holds; it restarts
only once the tag drifts past twice that band, so it does not twitch.

NO TAG: stop at once. If the tag was last seen near a frame edge it probably
drove out of view, so creep back toward that side for up to SEARCH_S, then
stop and wait. Derivative history is dropped, so a re-sighting cannot spike.

SPRING (--mode spring): stiff P, no D. With ~175 ms of Bluetooth lag the car
is still moving when it reaches the centre, overshoots, and P pulls it back:
a decaying oscillation, like a mass on a spring.

Video window keys: q quit, space pause/resume, m switch damped <-> spring.
"""

import argparse
import math
import time

import cv2
from cv2 import aruco

# --- Controller settings ------------------------------------------------
# (KP, KD) per mode. Gains act on the NORMALISED error, so they do not
# depend on camera resolution.
GAINS = {
    "damped": (1.0, 0.35),   # slows on approach, stops without overshoot
    "spring": (3.0, 0.0),    # overshoots the centre, swings back, settles
}
MAX_SPEED = 100       # motor command cap, 0-100
MIN_SPEED = 15        # below this the motors stall rather than creep
CENTER_TOL = 0.04     # |e| counted as centred: 4% of the half-width
STILL_TOL = 0.15      # |de| (half-widths per second) counted as stopped
D_FILTER = 0.5        # weight of each new derivative sample (1 = no filter)

EDGE = 0.75           # |e| beyond this when lost = "drove out of view"
SEARCH_SPEED = 20
SEARCH_S = 2.0
STALE_S = 0.3         # a gap longer than this resets the derivative

CMD_PERIOD_S = 0.1    # resend an unchanged command at most this often
CMD_STEP = 2          # ...or sooner if it changed by at least this much

SIGN_PULSE_SPEED = 30
SIGN_PULSE_S = 0.4
SIGN_MIN_PX = 8       # tag must move at least this far to trust the check

TAG_FAMILY = aruco.DICT_APRILTAG_36h11   # what scripts/make_apriltag.py makes
RED = (0, 0, 255)
WINDOW = "PD tag centring (q quit, space pause, m mode)"


class PDController:
    """Tag x-position in, motor speed out. No hardware or camera inside, so
    it can be reused as-is with any tag reader (e.g. the phone stream).

    sign: +1 if a POSITIVE motor command moves the tag RIGHT in the image,
          -1 if it moves it left. Found by sign_check() or given with --sign.
    """

    def __init__(self, kp, kd, sign=+1):
        self.kp, self.kd, self.sign = kp, kd, sign
        self.e = 0.0
        self.de = 0.0
        self.centred = False
        self._prev_t = None     # time of the last sighting
        self._prev_e = None

    def update(self, tag_x, frame_width, now):
        """Return (speed, state). tag_x is None when no tag is visible."""
        if tag_x is None:
            return self._lost(now)

        half = frame_width / 2.0
        e = (tag_x - half) / half
        if self._prev_t is None or now - self._prev_t > STALE_S:
            self.de = 0.0       # fresh sighting: no history worth trusting
        else:
            dt = max(now - self._prev_t, 1e-3)
            self.de += D_FILTER * ((e - self._prev_e) / dt - self.de)
        self._prev_t, self._prev_e, self.e = now, e, e

        tol = CENTER_TOL * (2 if self.centred else 1)   # hysteresis
        if abs(e) < tol and abs(self.de) < STILL_TOL:
            self.centred = True
            return 0.0, "centred"
        self.centred = False

        speed = -self.sign * 100.0 * (self.kp * e + self.kd * self.de)
        speed = max(-MAX_SPEED, min(MAX_SPEED, speed))
        if abs(speed) < MIN_SPEED:
            # Push through the motor deadband only when heading TOWARD the
            # centre; a small command the other way is D braking -- just stop.
            toward = -self.sign * e
            speed = math.copysign(MIN_SPEED, toward) if speed * toward > 0 else 0.0
        return speed, "tracking"

    def _lost(self, now):
        self.centred = False
        if self._prev_t is None:
            return 0.0, "no tag"
        if now - self._prev_t < SEARCH_S and abs(self.e) > EDGE:
            return -self.sign * math.copysign(SEARCH_SPEED, self.e), "searching"
        return 0.0, "lost"


def find_tag(detector, frame):
    """(corners 4x2, (cx, cy)) of the largest tag in view, or None."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    pts = max(corners, key=cv2.contourArea).reshape(4, 2)
    cx, cy = pts.mean(axis=0)
    return pts, (float(cx), float(cy))


def draw(frame, tag, speed, state, ctrl, mode):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
    band = int(CENTER_TOL * w / 2)
    cv2.rectangle(frame, (w // 2 - band, 0), (w // 2 + band, h), (0, 200, 0), 1)
    if tag is not None:
        pts, (cx, cy) = tag
        cv2.polylines(frame, [pts.astype(int)], True, RED, 3)
        cv2.circle(frame, (int(cx), int(cy)), 5, RED, -1)
        cv2.putText(frame, f"({cx:.0f}, {cy:.0f})", (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    lines = [f"{state}  speed {speed:+.0f}",
             f"e {ctrl.e:+.2f}  de/dt {ctrl.de:+.2f}",
             f"{mode}: kp {ctrl.kp}  kd {ctrl.kd}  sign {ctrl.sign:+d}"]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)


def watch(cap, detector, seconds):
    """Show the feed for `seconds`; return the last tag x seen (or None).

    Reading continuously, rather than sleeping, also drains the camera's
    frame buffer so the answer is not a stale frame.
    """
    x, end = None, time.monotonic() + seconds
    while time.monotonic() < end:
        ok, frame = cap.read()
        if not ok:
            break
        tag = find_tag(detector, frame)
        if tag is not None:
            x = tag[1][0]
            cv2.polylines(frame, [tag[0].astype(int)], True, RED, 3)
        cv2.putText(frame, "sign check...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)
    return x


def sign_check(cap, detector, trike):
    """Drive a short forward pulse and see which way the tag moves."""
    print("sign check: waiting for the tag ...")
    x0 = None
    while x0 is None:
        x0 = watch(cap, detector, 0.5)
    trike.drive_straight(SIGN_PULSE_SPEED)
    time.sleep(SIGN_PULSE_S)
    trike.drive_straight(0)
    x1 = watch(cap, detector, 0.8)   # let the car stop and the camera catch up
    if x1 is None or abs(x1 - x0) < SIGN_MIN_PX:
        raise SystemExit("sign check failed: the tag did not move visibly. "
                         "Check the car can drive, or pass --sign 1 / --sign -1.")
    sign = 1 if x1 > x0 else -1
    print(f"sign check: tag moved {x1 - x0:+.0f} px on a forward pulse -> sign {sign:+d}")
    return sign


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("source", nargs="?", default="0")
    p.add_argument("--mode", choices=GAINS, default="damped")
    p.add_argument("--kp", type=float)
    p.add_argument("--kd", type=float)
    p.add_argument("--sign", type=int, choices=(1, -1))
    p.add_argument("--dry-run", action="store_true",
                   help="camera and controller only; print speeds, no robot")
    args = p.parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source {args.source!r}")
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(TAG_FAMILY),
                                   aruco.DetectorParameters())

    trike = None
    if not args.dry_run:
        from trike import Trike     # only needs the LEGO library when driving
        # Straight-line driving only: leave the steering motor where it is.
        trike = Trike().connect(center=None, progress=print)

    mode = args.mode
    kp, kd = GAINS[mode]
    ctrl = PDController(args.kp if args.kp is not None else kp,
                        args.kd if args.kd is not None else kd)
    try:
        if args.sign is not None:
            ctrl.sign = args.sign
        elif trike is not None:
            ctrl.sign = sign_check(cap, detector, trike)

        paused, sent, sent_t = False, None, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            tag = find_tag(detector, frame)
            speed, state = ctrl.update(None if tag is None else tag[1][0],
                                       frame.shape[1], now)
            if paused:
                speed, state = 0.0, "paused"

            # Bluetooth is the bottleneck: send only what changed, plus a
            # periodic resend, rather than one command per frame.
            if (sent is None or abs(speed - sent) >= CMD_STEP
                    or now - sent_t > CMD_PERIOD_S):
                if trike is not None:
                    trike.drive_straight(speed)
                elif speed != sent:
                    print(f"{state:9} e {ctrl.e:+.2f}  speed {speed:+.0f}")
                sent, sent_t = speed, now

            draw(frame, tag, speed, state, ctrl, mode)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("m"):
                mode = "spring" if mode == "damped" else "damped"
                ctrl.kp, ctrl.kd = GAINS[mode]
    except KeyboardInterrupt:
        pass
    finally:
        if trike is not None:
            trike.stop(straighten=False)
            trike.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
