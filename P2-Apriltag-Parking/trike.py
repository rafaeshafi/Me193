"""
Hardware layer for the P2 trike (PD.py drives it). Run this file on its own
for an 11 s test drive.

Slip-free driving for a tricycle robot:
  - a Double Motor turning two fixed traction wheels (differential drive)
  - a Single Motor setting the steering angle of a third, free-spinning wheel

THE NO-SLIP CONDITION
---------------------
Put the drive axle at the origin, x forward, y left:

      (0, +W/2)  left drive wheel
         |
         |------ (L, 0)  steered wheel
         |
      (0, -W/2)  right drive wheel

The two drive wheels are fixed to the frame, so they can only roll along x.
That forces the instantaneous centre of rotation (ICR) onto their axle line,
at some point (0, R). Every wheel must circle that same ICR, or something
scrubs sideways.

The steered wheel sits at (L, 0), so its radius vector from the ICR is
(L, -R) and it must point perpendicular to that:

        tan(delta) = L / R          ->  R = L / tan(delta)

With the body rotating at omega about the ICR, the midpoint moves at
v = omega * R, and each drive wheel at its own radius:

        v_left  = omega * (R - W/2) = v * (1 - W*tan(delta) / (2L))
        v_right = omega * (R + W/2) = v * (1 + W*tan(delta) / (2L))

Written that way there is no singularity at delta = 0 (straight ahead), and
R never has to be computed. Sign convention: +delta steers LEFT, and the
left wheel correspondingly turns slower.

ROBOT GEOMETRY
--------------
Measured on this robot: track 86 mm, wheelbase 80 mm, drive wheels 62 mm
diameter. Only the RATIO W/2L enters the kinematics -- here 86/160 = 0.538.

GEARED DRIVE
------------
The drive motor no longer turns the wheels 1:1. A 36-tooth black double-bevel
gear on the motor drives a 20-tooth tan one on each wheel axle, so the wheels
turn 36/20 = 1.8x FASTER than the motor.

That ratio does NOT appear in the kinematics below, and this is worth being
clear about: both drive wheels share identical gearing, so it scales v_left and
v_right equally. The no-slip condition constrains only their RATIO, which is
unchanged. The gearing matters for ground speed and odometry, nothing else.

What DID change the kinematics is the geometry the gears forced: the pair of
gears pushed the drive wheels 16 mm further apart (two 1-stud-wide gears) and
the rebuild moved the steered wheel in to 80 mm. W/2L rose from 0.412 to 0.538,
so the two drive wheels now differ by ~30% more at the same steering angle.
"""

import math
import time

import legoeducation as le

# --- Hardware ------------------------------------------------------------
DRIVE_CARD_COLOR = le.LEGO_COLOR_YELLOW
DRIVE_CARD_SERIAL = 994

# Same connection card as the drive motor -- the library also filters on
# device type, so SingleMotor and DoubleMotor each find the right one.
STEER_CARD_COLOR = le.LEGO_COLOR_YELLOW
STEER_CARD_SERIAL = 994

# --- Geometry (measured on this robot) -----------------------------------
WHEELBASE_MM = 80.0     # L: drive axle -> steered wheel contact point
TRACK_MM = 86.0         # W: between the two drive wheel centres
DRIVE_WHEEL_DIA_MM = 62.0   # not used by the kinematics; kept for odometry

# Wheel revolutions per motor revolution: 36-tooth black double-bevel gear on
# the motor driving a 20-tooth tan one on the wheel. >1 means geared for SPEED.
# Cancels out of the kinematics (both wheels share it) -- it only sets how much
# ground speed a given motor command buys, so it lives here for odometry and
# for anyone wondering why the robot got quicker.
DRIVE_GEAR_RATIO = 36.0 / 20.0      # 1.8

# Motor degrees per degree of actual wheel steering. 1.0 if the motor drives
# the steering wheel directly; change it if there is gearing in between.
# Verified 1:1 on this robot, and +angle steers LEFT.
STEER_GEAR_RATIO = 1.0

# Absolute encoder reading with the steered wheel pointing STRAIGHT AHEAD.
# Absolute position survives disconnects and power cycles, so this only has
# to be measured once -- run read_abs.py with the wheel straight to re-check.
STEER_CENTER_ABS = 5

# The drive motor's positive direction relative to this module's body frame
# (x forward). A single external gear mesh REVERSES rotation, so adding the
# 36T->20T pair flipped this from -1 back to +1. This is the one constant here
# that is inferred rather than measured -- if forward and backward come out
# reversed on the first run, flip it back to -1 and nothing else.
DRIVE_SIGN = +1

# Set to True if a LEFT turn makes the robot swing RIGHT: it means the left
# and right drive outputs are mirrored relative to the steering. This is a
# port-mapping fact, not a direction fact -- the gears do not change which
# motor output reaches which side of the robot, so it stays True.
SWAP_DRIVE_WHEELS = True

MAX_STEER_DEG = 60.0    # beyond this the inner wheel speed blows up
STEER_SPEED = 100       # how briskly the steering motor moves (max)

# Ramp rates, 0-100. Maxed out so the robot reaches commanded speed as fast as
# the hardware allows -- with a ~175 ms Bluetooth command latency, a slow ramp
# on top of that makes it feel sluggish.
DRIVE_ACCEL = 100
DRIVE_DECEL = 100

# The motor's position controller stops a few degrees short of its target, and
# always toward centre. Left uncorrected that is several degrees of steering
# error, which is exactly the scrub this module exists to avoid -- so steering
# moves are closed-loop: command, read the encoder, correct, repeat.
# The motor has roughly 3 deg of deadband, so demanding better than that makes
# the trim loop oscillate. Correct with a damped gain and accept what is left:
# the drive side then matches the speeds to the MEASURED angle, so the residual
# error costs accuracy of heading, not traction.
STEER_TOLERANCE_DEG = 3.0
STEER_TRIM_GAIN = 0.6
STEER_MAX_TRIES = 3


def wheel_speeds(speed, steer_deg, wheelbase=WHEELBASE_MM, track=TRACK_MM):
    """Slip-free left/right drive speeds for a forward speed and steering angle.

    speed:     -100..100, the speed of the midpoint between the drive wheels
    steer_deg: +left / -right, the steering wheel's angle from straight ahead

    Returns (v_left, v_right), rescaled together if either would clip, so the
    ratio between them -- which is what keeps the wheels from scrubbing -- is
    preserved.
    """
    steer_deg = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steer_deg))
    ratio = track * math.tan(math.radians(steer_deg)) / (2.0 * wheelbase)

    v_left = speed * (1.0 - ratio)
    v_right = speed * (1.0 + ratio)

    # Clip by scaling BOTH, never independently: clamping one alone would
    # change the ratio and reintroduce slip.
    peak = max(abs(v_left), abs(v_right))
    if peak > 100.0:
        v_left *= 100.0 / peak
        v_right *= 100.0 / peak

    return v_left, v_right


def steer_for_radius(radius_mm, wheelbase=WHEELBASE_MM):
    """Steering angle that traces a given turn radius (+left / -right)."""
    if radius_mm == 0:
        raise ValueError("radius must be non-zero; this robot cannot spin in place")
    return math.degrees(math.atan2(wheelbase, radius_mm)) - (0.0 if radius_mm > 0 else 180.0)


# Reason codes bleak reports when the radio cannot be used, mapped to advice
# about the LAPTOP rather than the robot.
_BT_ADVICE = {
    "POWERED_OFF": """this computer's Bluetooth radio is switched OFF.
  Turn it on: Win+A and click the Bluetooth tile, or
  Settings -> Bluetooth & devices -> Bluetooth toggle.""",
    "NO_BLUETOOTH": """this computer has no usable Bluetooth adapter.
  Check Device Manager -> Bluetooth; a USB dongle may have dropped out.""",
    "NO_BLE_CENTRAL_ROLE": """the adapter does not support Bluetooth Low Energy.
  LEGO hubs are BLE-only, so this adapter cannot talk to them.""",
    "DENIED_BY_USER": """Bluetooth permission was denied for this app.
  Settings -> Privacy & security -> Bluetooth, and allow desktop apps.""",
    "DENIED_BY_SYSTEM": """Windows is blocking Bluetooth access for this app.
  Settings -> Privacy & security -> Bluetooth, and allow desktop apps.""",
}


def bluetooth_error():
    """Why the Bluetooth radio is unusable, as a sentence -- or None if it is fine.

    Worth the extra half-second at startup: the legoeducation library reports a
    switched-off RADIO and a switched-off MOTOR identically, as "could not find
    device". That sends you hunting around the robot for a fault that is on the
    laptop. Checking first turns the commonest failure into a sentence that says
    what to do.

    Never raises, and returns None whenever it cannot tell -- an unknown state
    must fall through to the normal connect path rather than block it.
    """
    try:
        import asyncio

        from bleak import BleakScanner
        from bleak.exc import BleakBluetoothNotAvailableError
    except Exception:
        return None

    async def probe():
        # Starting and immediately stopping a scan is the cheapest way to make
        # the backend actually touch the radio; nothing is discovered.
        scanner = BleakScanner()
        await scanner.start()
        await scanner.stop()

    try:
        asyncio.run(probe())
    except BleakBluetoothNotAvailableError as exc:
        reason = getattr(getattr(exc, "reason", None), "name", "")
        return "Bluetooth unavailable -- " + _BT_ADVICE.get(
            reason, "the radio is not available ({}).".format(exc))
    except RuntimeError:
        return None          # already inside an event loop; cannot probe
    except Exception:
        return None          # some other fault: let connect() report it
    return None


class Trike:
    """Double motor for drive, single motor for steering."""

    def __init__(self, wheelbase=WHEELBASE_MM, track=TRACK_MM):
        self.wheelbase = wheelbase
        self.track = track
        self.drive = le.DoubleMotor()
        self.steer = le.SingleMotor()
        self._steer_deg = 0.0

    def connect(self, center=True, progress=None, notify_ms=None):
        """Connect both motors.

        center:    centre the steering as part of connecting. Pass False from a
                   GUI -- centring issues BLOCKING motor moves, and if the wheel
                   cannot reach its target the call never returns.
                   None leaves the steering motor exactly where it is.
        progress:  optional callback(str) for stage reporting.
        notify_ms: IMU/motor notification interval. The library default is
                   100 ms (10 Hz), too slow for heading control; pass ~20 for
                   50 Hz. Minimum accepted by the library is 15.
        """
        say = progress or (lambda _m: None)
        # Check the radio BEFORE the motors, so a laptop-side fault is not
        # misreported as a missing robot.
        problem = bluetooth_error()
        if problem:
            raise ConnectionError(problem)
        say("connecting to drive motor ...")
        # le.DoubleMotor / le.SingleMotor do NOT raise when the device is not
        # found -- they print and return, leaving .connected False. Without
        # this check the caller drives a dead link forever.
        if notify_ms is None:
            self.drive.connect(card_color=DRIVE_CARD_COLOR, card_serial=DRIVE_CARD_SERIAL)
        else:
            self.drive.connect(card_color=DRIVE_CARD_COLOR, card_serial=DRIVE_CARD_SERIAL,
                               device_notification_delay=notify_ms)
        if not self.drive.connected:
            raise ConnectionError(
                f"double motor not found (card {DRIVE_CARD_SERIAL:04d}) -- is it powered on?")
        say("connecting to steering motor ...")
        self.steer.connect(card_color=STEER_CARD_COLOR, card_serial=STEER_CARD_SERIAL)
        if not self.steer.connected:
            self.drive.disconnect()
            raise ConnectionError(
                f"single motor not found (card {STEER_CARD_SERIAL:04d}) -- is it powered on?")
        self.drive.movement_set_acceleration(DRIVE_ACCEL, DRIVE_DECEL)
        self.steer.motor_set_speed(STEER_SPEED)
        self.steer.motor_set_acceleration(DRIVE_ACCEL, DRIVE_DECEL)
        # HOLD, not coast: the steering has to resist spring-back and the
        # cornering load, or the wheel drifts back toward centre and scrubs.
        self.steer.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
        if center:
            say("centring steering ...")
            self.center_steering(wait=True)
        elif center is not None:
            self.set_steer(0.0, wait=False)
        say("connected")
        return self

    def center_steering(self, wait=True):
        """Point the steered wheel straight ahead (absolute STEER_CENTER_ABS)."""
        return self.set_steer(0.0, wait=wait)

    def steer_position(self):
        """Current steering angle in degrees, read from the absolute encoder."""
        raw = self.steer.motor.absolutePosition   # re-read: notifications rebind
        delta = (raw - STEER_CENTER_ABS + 180) % 360 - 180
        return delta / STEER_GEAR_RATIO

    def set_steer(self, steer_deg, wait=False):
        """Point the steered wheel at an angle (+left / -right).

        Absolute positioning: the target is a fixed encoder value, not an
        offset from wherever the motor happens to be. Errors cannot accumulate
        across calls, and the zero survives a power cycle.
        """
        steer_deg = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steer_deg))
        target = int(round(STEER_CENTER_ABS + steer_deg * STEER_GEAR_RATIO)) % 360
        self.steer.motor_run_to_absolute_position(
            target, direction=le.MOTOR_MOVE_DIRECTION_SHORTEST, blocking=wait)
        self._steer_deg = steer_deg

        # Closed-loop trim. Only possible when we waited for the move, since
        # otherwise the encoder has not arrived yet.
        if wait:
            for _ in range(STEER_MAX_TRIES):
                err = steer_deg - self.steer_position()
                if abs(err) <= STEER_TOLERANCE_DEG:
                    break
                target = int(round(target + STEER_TRIM_GAIN * err * STEER_GEAR_RATIO)) % 360
                self.steer.motor_run_to_absolute_position(
                    target, direction=le.MOTOR_MOVE_DIRECTION_SHORTEST, blocking=True)
        return steer_deg

    def drive_at(self, speed, steer_deg, settle=True):
        """The main command: drive at `speed` while steering `steer_deg`.

        Steering is set FIRST and allowed to settle, because driving with the
        wheel still swinging is exactly when it scrubs.

        The wheel speeds are then computed from the angle the steered wheel
        ACTUALLY reached, not the one requested. The motor has a few degrees of
        deadband, and matching the speeds to the real angle is what keeps the
        wheels rolling rather than scrubbing -- a small heading error is much
        cheaper than slip.
        """
        turning = abs(steer_deg - self._steer_deg) > 1.0
        self.set_steer(steer_deg, wait=settle and turning)
        try:
            actual = self.steer_position()
            if abs(actual - steer_deg) < 15.0:   # sane reading, not a dropout
                steer_deg = actual
        except Exception:
            pass
        v_left, v_right = wheel_speeds(speed, steer_deg, self.wheelbase, self.track)
        out_l, out_r = DRIVE_SIGN * v_left, DRIVE_SIGN * v_right
        if SWAP_DRIVE_WHEELS:
            out_l, out_r = out_r, out_l
        self.drive.movement_move_tank(int(round(out_l)), int(round(out_r)),
                                      blocking=False)
        return v_left, v_right

    def drive_straight(self, speed):
        """Both drive wheels at `speed` (-100..100), steering left untouched.

        For fast control loops: drive_at() re-sends the steering target and
        reads the encoder on every call, two extra Bluetooth round trips.
        Centre the steering once (connect() does), then call this.
        """
        out = int(round(DRIVE_SIGN * max(-100.0, min(100.0, speed))))
        self.drive.movement_move_tank(out, out, blocking=False)

    # --- convenience commands -------------------------------------------
    def forward(self, speed=40):
        return self.drive_at(speed, 0.0)

    def turn(self, speed=40, steer_deg=30.0):
        return self.drive_at(speed, steer_deg)

    def turn_radius(self, speed=40, radius_mm=300.0):
        return self.drive_at(speed, steer_for_radius(radius_mm, self.wheelbase))

    # --- heading (yaw) --------------------------------------------------
    def heading(self):
        """Heading in degrees since the last reset_heading(), wrapped to +/-180.

        The hardware reports decidegrees, hence the /10. Sign matches the
        steering convention: POSITIVE is counter-clockwise / left.
        """
        raw = self.drive.imu_device.yaw      # re-read: notifications rebind this
        return ((float(raw) / 10.0) + 180.0) % 360.0 - 180.0

    def reset_heading(self):
        """Define the current direction as heading zero."""
        self.drive.imu_reset_yaw_axis(0)

    def stop(self, straighten=True):
        try:
            self.drive.movement_move_tank(0, 0, blocking=False)
            self.drive.movement_stop()
            if straighten:
                self.set_steer(0.0, wait=True)
        except Exception:
            pass

    def disconnect(self):
        for call in (self.drive.disconnect, self.steer.disconnect):
            try:
                call()
            except Exception:
                pass


def main():
    print("left/right wheel speeds at speed=50 (no hardware needed):")
    for d in (0, 10, 20, 30, 45, 60, -30):
        vl, vr = wheel_speeds(50, d)
        print(f"  steer {d:+4.0f} deg -> left {vl:6.1f}  right {vr:6.1f}")

    t = Trike().connect()
    print("\nconnected -- point the steered wheel straight before running this")
    try:
        print("forward 2 s")
        t.forward(40)
        time.sleep(2)

        print("left turn 3 s")
        t.turn(40, 30)
        time.sleep(3)

        print("right turn 3 s")
        t.turn(40, -30)
        time.sleep(3)

        print("wide left arc (400 mm radius) 3 s")
        t.turn_radius(40, 400)
        time.sleep(3)
    finally:
        t.stop()
        t.disconnect()
        print("done")


if __name__ == "__main__":
    main()