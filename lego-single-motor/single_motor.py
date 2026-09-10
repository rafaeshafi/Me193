"""Spin a single LEGO Education motor (Computer Science & AI kit) over Bluetooth.

Uses the `legoeducation` library: https://github.com/LEGO/LEGOEducation
Pointing at THIS library (not the `spike` / `hub` modules) is what keeps the
code targeting the CS & AI kit hardware instead of SPIKE Prime.
"""

import time

import legoeducation as le


def main() -> int:
    motor = le.SingleMotor()

    # Scan over BLE and connect to the first Single Motor found.
    # To target one specific motor, use its Connection Card instead:
    #   motor.connect(card_color=le.LEGO_COLOR_AZURE, card_serial="3683")
    motor.connect()

    if not motor.connected:
        print("Could not connect to a Single Motor. Is it powered on and nearby?")
        return 1

    print("Connected. Spinning...")

    try:
        # 1) One full turn at 50% speed. Blocks until the move completes.
        motor.motor_run_for_degrees(360, speed=50)

        # 2) Free-run at 30% for ~2 seconds, printing live telemetry.
        motor.motor_reset_relative_position()
        motor.motor_run(speed=30)
        for _ in range(20):
            print(f"position: {motor.motor.position:>6}   speed: {motor.motor.speed}")
            time.sleep(0.1)
    except KeyboardInterrupt:
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
