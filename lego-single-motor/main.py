"""Spin a single LEGO Education motor at whatever speed you type in.

Same hardware target as `single_motor.py` (CS & AI kit via the `legoeducation`
library). Enter a speed percentage each time you're prompted; the motor changes
to that speed immediately and keeps spinning until the next entry.
"""

import legoeducation as le


def read_speed() -> int | None:
    """Prompt for a speed in [-100, 100]. Return None to quit."""
    raw = input("speed % (-100 to 100, blank or 'q' to quit): ").strip()
    if raw.lower() in ("", "q", "quit"):
        return None
    try:
        speed = int(raw)
    except ValueError:
        print("  not a whole number — try again")
        return read_speed()
    if not -100 <= speed <= 100:
        print("  out of range — must be between -100 and 100")
        return read_speed()
    return speed


def main() -> int:
    motor = le.SingleMotor()
    motor.connect()

    if not motor.connected:
        print("Could not connect to a Single Motor. Is it powered on and nearby?")
        return 1

    print("Connected. Positive = clockwise, negative = counter-clockwise.")

    try:
        while True:
            speed = read_speed()
            if speed is None:
                break
            # Re-issuing motor_run with a new speed changes the running speed;
            # it also starts the motor if it was stopped.
            motor.motor_run(speed=speed)
            print(f"  running at {speed}%   (position: {motor.motor.position})")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
    finally:
        # Best-effort teardown: always try to stop the motor and drop the BLE
        # link, even on error or Ctrl-C — and don't let a failing teardown call
        # (e.g. the link already dropped) skip the rest.
        try:
            motor.motor_stop()
        except Exception:
            pass
        try:
            motor.disconnect()
        except Exception:
            pass

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
