"""Run the trained policy on the real LEGO Double Motor over Bluetooth, so it walks straight.

    python real/drive_policy.py --check          # once: find the motor directions and speed (robot held in the air)
    python real/drive_policy.py                  # drive with the policy for 10 s
    python real/drive_policy.py --seconds 20
    python real/drive_policy.py --open-loop 50   # no policy: both beams at 50 %, to compare against
    python real/drive_policy.py --flip           # once, if the robot walks the wrong way: reverse it and save that

Before every drive: put the beams in the Onshape pose (both flat, left bar pointing forward, right bar
pointing back), set the robot down pointing where it should go, and keep it still until it starts.
Ctrl-C stops it. Every run is saved to data/real/ as a .csv.

The Connection Card is read from robot/params.json ("real": card_color, card_serial).
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "robot" / "params.json"
sys.path.append(str(Path(__file__).resolve().parent))
from policy_np import Policy, make_obs  # noqa: E402


def wrap_deg(angle):
    return (angle + 180) % 360 - 180


class Robot:
    """The Double Motor. Angles come back in the simulation's units and directions once --check has run."""

    def __init__(self, real):
        import legoeducation as le
        self.le = le
        color = getattr(le, f"LEGO_COLOR_{str(real.get('card_color', '')).upper()}", None)
        if color is None or not real.get("card_serial"):
            sys.exit('params.json: set "real": {"card_color": "...", "card_serial": "..."} from your Connection Card')
        self.dev = le.DoubleMotor()
        print(f"Looking for Double Motor {real['card_color']} {real['card_serial']}...")
        self.dev.connect(card_color=color, card_serial=str(real["card_serial"]))
        if not self.dev.connected:
            sys.exit("Could not connect. Is the Double Motor on and close by?")
        print("Connected.")

    def raw(self):
        """Left and right beam position (degrees) and heading (degrees), exactly as the hub reports them."""
        le, imu = self.le, self.dev.imu_device
        yaw = imu.yaw / 10                      # the hub reports tenths of a degree
        if abs(imu.roll) > 900:                 # past 90 degrees of tilt the hub flips yaw by half a turn
            yaw = wrap_deg(yaw + 180)
        return self.dev.motor[le.MOTOR_LEFT].position, self.dev.motor[le.MOTOR_RIGHT].position, yaw

    def zero(self):
        self.dev.motor_reset_relative_position(motor=self.le.MOTOR_BOTH)
        self.dev.imu_reset_yaw_axis(0)
        time.sleep(0.3)                         # wait for fresh readings

    def drive(self, left_percent, right_percent):
        self.dev.movement_move_tank(int(round(left_percent)), int(round(right_percent)), blocking=False)

    def close(self):
        for step in (self.dev.movement_stop, self.dev.disconnect):
            try:
                step()
            except Exception:
                pass


def check(robot, params):
    """Find which way everything counts, and how fast a beam turns per percent. Saves it in params.json."""
    input("Hold the robot in the AIR, beams free to spin, then press Enter...")
    robot.zero()
    left0, right0, _ = robot.raw()
    robot.drive(20, 20)
    time.sleep(2.0)
    left1, right1, _ = robot.raw()
    robot.drive(0, 0)
    d_left, d_right = left1 - left0, right1 - right0
    if min(abs(d_left), abs(d_right)) < 20:
        sys.exit(f"The beams barely moved (left {d_left:.0f}, right {d_right:.0f} degrees). Check the robot and try again.")
    answer = input("Did the TOP of the beams move toward the FRONT of the robot (the long end)? [y/n] ").strip().lower()
    forward = 1 if answer.startswith("y") else -1
    input("Put the robot on the table. Press Enter, then turn it a quarter turn to its LEFT "
          "(counter-clockwise seen from above) within 5 seconds...")
    robot.zero()
    time.sleep(5.0)
    yaw = robot.raw()[2]
    if abs(yaw) < 30:
        sys.exit(f"The heading only changed {yaw:.0f} degrees. Run --check again and turn it a full quarter turn.")
    params["real"].update({
        "forward_sign": forward,
        "left_count_sign": forward * (1 if d_left > 0 else -1),
        "right_count_sign": forward * (1 if d_right > 0 else -1),
        "yaw_sign": 1 if yaw > 0 else -1,
        "rad_s_per_percent": round(math.radians((abs(d_left) + abs(d_right)) / 2) / 2.0 / 20, 4),
    })
    PARAMS.write_text(json.dumps(params, indent=2) + "\n")
    print("Saved to robot/params.json:", params["real"])
    print(f"Full speed is about {params['real']['rad_s_per_percent'] * 100:.1f} rad/s "
          f"(the simulation assumes {params['max_speed_rad_s']}).")


def drive(robot, params, seconds, open_loop):
    real = params["real"]
    missing = [k for k in ("forward_sign", "left_count_sign", "right_count_sign", "yaw_sign", "rad_s_per_percent") if k not in real]
    if missing:
        sys.exit("Run  python real/drive_policy.py --check  first (missing: " + ", ".join(missing) + ")")
    policy = None if open_loop is not None else Policy()
    step = 1 / params["loop_hz"]
    percent_per_unit = params["max_speed_rad_s"] / real["rad_s_per_percent"]   # a command of 1.0, in percent

    # The hub's own "forward" may be the robot's backward. Then its left motor is also the robot's right one,
    # so the two sides swap as well as changing sign.
    swap = real["forward_sign"] < 0

    def read():
        left, right, yaw = robot.raw()
        left, right = math.radians(left) * real["left_count_sign"], math.radians(right) * real["right_count_sign"]
        return (right, left, math.radians(yaw) * real["yaw_sign"]) if swap else (left, right, math.radians(yaw) * real["yaw_sign"])

    input("Beams in the Onshape pose, robot on the table pointing straight. Press Enter to start...")
    robot.zero()
    out = ROOT / "data" / "real" / time.strftime("run_%Y%m%d_%H%M%S.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    last, action = read(), np.zeros(2)
    start = time.monotonic()
    with out.open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["t", "heading_deg", "left_rad", "right_rad", "left_cmd", "right_cmd"])
        tick = 0
        while (now := time.monotonic() - start) < seconds:
            now_read = read()
            speeds = [(now_read[i] - last[i]) / step for i in (0, 1)]
            turn_rate = math.remainder(now_read[2] - last[2], math.tau) / step
            if policy:
                obs = make_obs(now_read[2], turn_rate, speeds, now_read[:2], action)
                action = np.clip(policy.act(obs), -1, 1)
            else:
                action = np.full(2, open_loop / 100 * real["rad_s_per_percent"] * 100 / params["max_speed_rad_s"])
            percent = np.clip(action * percent_per_unit, -100, 100) * real["forward_sign"]
            robot.drive(*(percent[::-1] if swap else percent))
            log.writerow([f"{now:.3f}", f"{math.degrees(now_read[2]):.1f}", f"{now_read[0]:.3f}", f"{now_read[1]:.3f}",
                          f"{action[0]:.3f}", f"{action[1]:.3f}"])
            if tick % params["loop_hz"] == 0:
                print(f"t={now:4.1f}s  heading={math.degrees(now_read[2]):+6.1f} deg  command L={action[0]:+.2f} R={action[1]:+.2f}")
            last, tick = now_read, tick + 1
            time.sleep(max(0.0, start + tick * step - time.monotonic()))
    print(f"Done. Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trained policy on the real Double Motor.")
    parser.add_argument("--check", action="store_true", help="find the motor directions and speed, once")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to drive, 1 to 60 (default 10)")
    parser.add_argument("--flip", action="store_true", help="reverse the driving direction and save it, then exit")
    parser.add_argument("--open-loop", type=float, metavar="PERCENT", help="no policy: both beams at this percent, 1 to 100")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    if args.open_loop is not None and not 1 <= args.open_loop <= 100:
        parser.error("--open-loop must be between 1 and 100")
    params = json.loads(PARAMS.read_text())
    params.setdefault("real", {})
    if args.flip:
        for key in ("forward_sign", "left_count_sign", "right_count_sign"):
            if key not in params["real"]:
                sys.exit("Run  python real/drive_policy.py --check  first")
            params["real"][key] *= -1
        PARAMS.write_text(json.dumps(params, indent=2) + "\n")
        sys.exit(f"Reversed. forward_sign is now {params['real']['forward_sign']}.")
    robot = Robot(params["real"])
    try:
        check(robot, params) if args.check else drive(robot, params, args.seconds, args.open_loop)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.close()
