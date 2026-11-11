"""Q-learning to keep a LEGO Education Double Motor car driving straight.

The car reads its IMU yaw, bins the heading error into a state, and learns
which left/right speed differential brings the heading back to zero.

    python rl_straight.py --sim                 # train on a simulated car
    python rl_straight.py --episodes 10         # train on the real car
    python rl_straight.py --run                 # drive with the learned table
    python rl_straight.py --flip                # if "forward" drives backwards

Default Connection Card is red / 1131; override with --card-color/--card-serial.
The Q-table is saved next to this file as q_table.npy and reloaded each run.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np

# 1. Define things.
ALPHA = 0.1            # learning rate
GAMMA = 0.9            # discount factor
EPSILON_START = 1.0    # exploration rate
EPSILON_DECAY = 0.98   # multiplied in after every step
EPSILON_MIN = 0.05     # never stop exploring entirely while training

BASE_SPEED = 15                          # % both motors run at (--speed overrides)
ACTIONS = [-8, -5, -2, 0, 2, 5, 8]       # speed differential added left, taken right
YAW_EDGES = [-30, -15, -8, -3, 3, 8, 15, 30]  # degrees -> len+1 = 9 state bins
GOOD_YAW = 3           # |yaw| below this counts as "straight"
STEP_S = 0.1           # seconds between decisions
STEPS_PER_EPISODE = 40
START_OFFSET = 20      # each episode starts with a random yaw error up to this

Q_PATH = Path(__file__).with_name("q_table.npy")


def wrap(angle):
    """Wrap degrees into [-180, 180)."""
    return (angle + 180) % 360 - 180


def to_state(yaw):
    """2. Read the state: yaw error -> bin index."""
    return int(np.digitize(wrap(yaw), YAW_EDGES))


def reward(yaw):
    yaw = abs(wrap(yaw))
    return 1.0 if yaw < GOOD_YAW else -yaw / 10


def choose(q, state, epsilon):
    """3. Explore below epsilon, otherwise exploit the best-known action."""
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(q[state]))


class RealCar:
    """The Double Motor over BLE; yaw comes from its built-in IMU."""

    def __init__(self, card_color, card_serial, direction):
        import legoeducation as le
        color = getattr(le, f"LEGO_COLOR_{card_color.upper()}", None)
        if color is None:
            raise SystemExit(f"unknown card colour: {card_color}")
        self.direction = direction
        self.motor = le.DoubleMotor()
        print(f"Scanning for Double Motor {card_color} {card_serial}...")
        self.motor.connect(card_color=color, card_serial=card_serial)
        if not self.motor.connected:
            raise SystemExit("Could not connect. Is the Double Motor powered on and nearby?")
        print("Connected.")

    def reset_yaw(self, value):
        self.motor.imu_reset_yaw_axis(int(value * 10))  # IMU works in tenths of a degree
        time.sleep(0.2)  # let a fresh IMU notification arrive

    def yaw(self):
        # The motor body tumbles about its axle while driving. Each time pitch
        # passes +-90 the IMU's Euler angles flip to the equivalent form
        # (yaw+180, roll+180), so undo that to keep a continuous heading.
        imu = self.motor.imu_device
        yaw = imu.yaw / 10
        return wrap(yaw + 180) if abs(imu.roll) > 900 else yaw

    def drive(self, left, right):
        self.motor.movement_move_tank(int(left) * self.direction,
                                      int(right) * self.direction, blocking=False)

    def stop(self):
        self.motor.movement_stop()

    def close(self):
        for teardown in (self.motor.movement_stop, self.motor.disconnect):
            try:
                teardown()
            except Exception:
                pass


class SimCar:
    """A crude stand-in: speed difference turns the car, plus drift and noise."""

    TURN_GAIN = 1.5   # deg/s of yaw per % of left-right difference
    DRIFT = 8.0       # deg/s the car veers with equal speeds (mismatched motors)

    def __init__(self):
        self._yaw = 0.0
        self._rate = 0.0

    def reset_yaw(self, value):
        self._yaw = float(value)

    def yaw(self):
        self._yaw += (self._rate + self.DRIFT + random.gauss(0, 2)) * STEP_S
        return self._yaw

    def drive(self, left, right):
        self._rate = self.TURN_GAIN * (left - right)

    def stop(self):
        self._rate = 0.0

    def close(self):
        pass


def run_episode(car, q, epsilon, learn, base_speed):
    """One drive. Returns (updated epsilon, mean |yaw| over the episode)."""
    car.reset_yaw(random.uniform(-START_OFFSET, START_OFFSET))
    state = to_state(car.yaw())
    errors = []
    for _ in range(STEPS_PER_EPISODE):
        action = choose(q, state, epsilon if learn else 0.0)       # 3.
        diff = ACTIONS[action]
        car.drive(base_speed + diff, base_speed - diff)            # 4.
        time.sleep(STEP_S if isinstance(car, RealCar) else 0)
        yaw = car.yaw()                                            # 5.
        new_state, r = to_state(yaw), reward(yaw)
        if learn:                                                  # 6.
            best_next = np.max(q[new_state])
            q[state, action] += ALPHA * (r + GAMMA * best_next - q[state, action])
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)    # 7.
        errors.append(abs(wrap(yaw)))
        state = new_state                                          # 8.
    car.stop()
    return epsilon, float(np.mean(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", action="store_true", help="use the simulated car")
    parser.add_argument("--run", action="store_true", help="drive greedily, no learning")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fresh", action="store_true", help="ignore any saved Q-table")
    parser.add_argument("--speed", type=int, default=BASE_SPEED, help="base motor speed %%")
    parser.add_argument("--flip", action="store_true", help="reverse motor direction")
    parser.add_argument("--card-color", default="red")
    parser.add_argument("--card-serial", default="1131")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not max(ACTIONS) < args.speed <= 100 - max(ACTIONS):
        parser.error(f"--speed must be between {max(ACTIONS) + 1} and {100 - max(ACTIONS)}")

    shape = (len(YAW_EDGES) + 1, len(ACTIONS))
    if Q_PATH.exists() and not args.fresh:
        q = np.load(Q_PATH)
        if q.shape != shape:
            parser.error(f"{Q_PATH.name} has shape {q.shape}, expected {shape}; use --fresh")
        print(f"Loaded {Q_PATH.name}")
    else:
        q = np.zeros(shape)

    car = SimCar() if args.sim else RealCar(args.card_color, args.card_serial,
                                            -1 if args.flip else 1)
    epsilon = EPSILON_START
    learn = not args.run
    try:
        for episode in range(1, args.episodes + 1):
            if not args.sim:
                input(f"\nEpisode {episode}: place the car, then press Enter...")
            epsilon, mean_err = run_episode(car, q, epsilon, learn, args.speed)
            print(f"episode {episode:3d}   mean |yaw| {mean_err:6.1f} deg"
                  + (f"   epsilon {epsilon:.3f}" if learn else ""))
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
    finally:
        car.close()
        if learn:
            np.save(Q_PATH, q)
            print(f"Saved {Q_PATH}")

    print("\nGreedy action per yaw bin (diff %):")
    labels = ["<-30"] + [f"{a}..{b}" for a, b in zip(YAW_EDGES, YAW_EDGES[1:])] + [">30"]
    for label, row in zip(labels, q):
        print(f"  {label:>8}: {ACTIONS[int(np.argmax(row))]:+d}" if row.any() else f"  {label:>8}: (unvisited)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
