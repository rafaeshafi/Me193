"""Drive a LEGO Education Double Motor in a straight line over Bluetooth.

Uses the `legoeducation` library's Movement functions, which run both motors
together in sync so the car tracks straight (no per-side tank speeds to match).

    python double_motor_straight.py                      # 720 deg forward at 50%
    python double_motor_straight.py --degrees 1440 --speed 30
    python double_motor_straight.py --backward
    python double_motor_straight.py --card-color azure --card-serial 3683

If the car goes the wrong way on "forward" (motor mounted flipped, as on the
pose/whistle cars), just add --backward.
"""

import argparse

import legoeducation as le


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrees", type=int, default=720,
                        help="motor rotation degrees to drive (distance), default 720")
    parser.add_argument("--speed", type=int, default=50, help="speed %% 1-100, default 50")
    parser.add_argument("--backward", action="store_true", help="drive backward instead")
    parser.add_argument("--card-color", help="Connection Card colour, e.g. azure")
    parser.add_argument("--card-serial", help="Connection Card serial, e.g. 3683")
    args = parser.parse_args()

    if not 1 <= args.speed <= 100:
        parser.error("--speed must be between 1 and 100")
    if args.degrees <= 0:
        parser.error("--degrees must be positive")

    # With several cars in one room, a bare connect() grabs whichever motor
    # answers first; the Connection Card targets one specific unit.
    target = {}
    if args.card_color and args.card_serial:
        color = getattr(le, f"LEGO_COLOR_{args.card_color.upper()}", None)
        if color is None:
            parser.error(f"unknown card colour: {args.card_color}")
        target = {"card_color": color, "card_serial": args.card_serial}

    motor = le.DoubleMotor()
    print("Scanning for a Double Motor over BLE...")
    motor.connect(**target)
    if not motor.connected:
        print("Could not connect to a Double Motor. Is it powered on and nearby?")
        return 1

    direction = (le.MOVEMENT_MOVE_DIRECTION_BACKWARD if args.backward
                 else le.MOVEMENT_MOVE_DIRECTION_FORWARD)
    print(f"Connected. Driving {'backward' if args.backward else 'forward'} "
          f"{args.degrees} deg at {args.speed}%...")

    try:
        # Blocks until both motors have turned the requested degrees.
        motor.movement_move_for_degrees(args.degrees, direction=direction, speed=args.speed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Best-effort teardown: always stop and drop the BLE link, even on
        # Ctrl-C, without letting one failing call skip the other.
        for teardown in (motor.movement_stop, motor.disconnect):
            try:
                teardown()
            except Exception:
                pass

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
