"""P3 Whistle Soccer: steer a LEGO Double Motor car by whistling.

Middle whistle drives straight, higher turns left, lower turns right, silence
stops. The car waits for "start" on MQTT. As the ball, it dies (publish + death
song) if the goalie reaches its colour sensor, and scores (publish + victory
song) on a held top-note whistle.

As the goalie this is the ROBOT laptop: its whistle slides the car back and
forth along the goal line (higher = forward, lower = backward, middle or
silence = stop; it never turns), and it also owns the glove's Single Motor, which it moves on commands that the GLOVE laptop
(glove.py) sends over MQTT. It reports the game state back to that laptop.

    python whistle_car.py --calibrate         # measure your whistle once
    python whistle_car.py --role ball         # game day
    python whistle_car.py --role goalie       # robot laptop (+ glove.py on the other)
    python whistle_car.py --no-motor --no-sensor --no-glove   # audio + display only

Keys in the plot window: s = start locally, r = reset, c / g = pretend
caught / goal (tests the messages and songs), q = quit.
Messages, songs, card and speeds are all in config.py.
"""

import argparse
import math
import queue
import time

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
import pyaudio

import config
import whistle_policy as wp
from live_audio import Display, Mic, hearing_lines, load_bands, measure_floor, run_calibration
from songs import Player

SEND_INTERVAL = 0.08  # seconds between BLE motor writes (~12/s)
SEND_DELTA = 4        # don't spend a BLE write on a change smaller than this
STATE_EVERY = 1.0     # s between state reports to the glove laptop


# --- LEGO hardware ----------------------------------------------------------

class Car:
    """Double Motor, Color Sensor and glove Single Motor, or stand-ins.

    Rate-limits BLE writes and sends them non-blocking, so the audio loop never
    waits on Bluetooth. (Same approach as P1-Pose-Race.)
    """

    def __init__(self, *, motor=True, sensor=True, glove=False):
        self.use_motor, self.use_sensor, self.use_glove = motor, sensor, glove
        self.motor = self.sensor = self.glove = None
        self.glove_angle = 0
        self._last_sent = None
        self._last_send_time = 0.0

    def connect(self):
        if not (self.use_motor or self.use_sensor or self.use_glove):
            return True
        import legoeducation as le

        valid = {le.LEGO_COLOR_NAME_MAP[c].removeprefix("LEGO_COLOR_").lower(): c
                 for c in le.CARD_COLORS}
        color = valid.get(config.CARD_COLOR.lower())
        if color is None:
            print(f"Unknown card colour {config.CARD_COLOR!r}. Expected one of: "
                  f"{', '.join(sorted(valid))}")
            return False
        card = dict(card_color=color, card_serial=config.CARD_SERIAL)
        print(f"Connecting over BLE to card {config.CARD_COLOR} {config.CARD_SERIAL}...")

        for wanted, attr, cls, name in ((self.use_motor, "motor", le.DoubleMotor, "Double Motor"),
                                        (self.use_sensor, "sensor", le.ColorSensor, "Color Sensor"),
                                        (self.use_glove, "glove", le.SingleMotor, "glove Single Motor")):
            if not wanted:
                continue
            device = cls()
            device.connect(**card)
            if not device.connected:
                print(f"{name} not found. Is it on and nearby?")
                return False
            setattr(self, attr, device)
            print(f"  {name} connected.")
        if self.glove is not None:
            # Wherever the glove points now is "centre", and it holds its angle
            # between commands instead of flopping around.
            self.glove.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
            self.glove.motor_reset_relative_position()
        return True

    def reflection(self):
        if self.sensor is None:
            return float("nan")
        return float(self.sensor.sensor.reflection)

    def drive(self, left, right):
        left, right = int(round(left)), int(round(right))
        now = time.monotonic()
        stopping = (left, right) == (0, 0)
        if self._last_sent is not None:
            moved = max(abs(left - self._last_sent[0]), abs(right - self._last_sent[1]))
            if not stopping and (moved < SEND_DELTA or now - self._last_send_time < SEND_INTERVAL):
                return
            if self._last_sent == (left, right):
                return
        self._last_sent = (left, right)
        self._last_send_time = now
        if self.motor is not None:
            if config.SWAP_SIDES:  # motor ports wired the other way round
                left, right = right, left
            self.motor.movement_move_tank(left * config.MOTOR_DIRECTION,
                                          right * config.MOTOR_DIRECTION, blocking=False)

    def move_glove(self, angle):
        angle = int(np.clip(angle, -config.GLOVE_MAX_DEG, config.GLOVE_MAX_DEG))
        if angle == self.glove_angle:
            return
        self.glove_angle = angle
        if self.glove is not None:
            self.glove.motor_run_to_relative_position(angle * config.GLOVE_DIRECTION,
                                                      speed=config.GLOVE_SPEED, blocking=False)

    def stop(self):
        if self.motor is not None:
            self.motor.movement_stop(blocking=False)
        self._last_sent = (0, 0)

    def close(self):
        for device, teardown in ((self.motor, "movement_stop"), (self.motor, "disconnect"),
                                 (self.sensor, "disconnect"), (self.glove, "disconnect")):
            if device is not None:
                try:
                    getattr(device, teardown)()
                except Exception:
                    pass  # link may already be gone; keep tearing down regardless


# --- MQTT -------------------------------------------------------------------

class Radio:
    """Subscribes to some topics; incoming (topic, text) pairs land in a queue."""

    def __init__(self, topics):
        self.topics = topics
        self.inbox = queue.Queue()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = lambda c, *_: [c.subscribe(t) for t in self.topics]
        self.client.on_message = lambda c, u, msg: self.inbox.put(
            (msg.topic, msg.payload.decode().strip()))

    def connect(self):
        try:
            self.client.connect(config.BROKER, config.PORT)
        except OSError as e:
            print(f"MQTT: could not reach {config.BROKER} ({e}). Press s to start locally.")
            return
        self.client.loop_start()
        print(f"MQTT: listening on {', '.join(self.topics)} at {config.BROKER}")

    def publish(self, text, topic=config.TOPIC, quiet=False):
        if not quiet:
            print(f"MQTT -> [{topic}] {text}")
        self.client.publish(topic, text)

    def messages(self):
        while not self.inbox.empty():
            yield self.inbox.get()

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


# --- Game -------------------------------------------------------------------

WAITING, DRIVING, WON, LOST = "WAITING FOR START", "DRIVING", "WON", "LOST"


class Game:
    def __init__(self, role, car, radio, player, policy):
        self.role, self.car, self.radio, self.player, self.policy = role, car, radio, player, policy
        self.state = WAITING
        self.result = ""
        self.baseline = 0.0
        self.high_since = None
        self.glove_heard = None  # time of the last glove command from the glove laptop

    def start(self):
        if self.state == DRIVING:
            return
        refl = self.car.reflection()
        self.baseline = 0.0 if math.isnan(refl) else refl
        self.high_since = None
        self.state, self.result = DRIVING, ""
        self.car.move_glove(0)
        print(f"START (sensor baseline reflection {self.baseline:.0f})")

    def reset(self):
        self.car.stop()
        self.state, self.result = WAITING, ""

    def finish(self, won, why, publish=None):
        self.car.stop()
        self.state, self.result = (WON if won else LOST), why
        if publish:
            self.radio.publish(publish)
        self.player.play(config.VICTORY_SONG if won else config.DEATH_SONG)
        print(f"{self.state}: {why}")

    def caught(self):
        if self.role == "ball":
            self.finish(False, "goalie reached our sensor", publish=config.MSG_CAUGHT)

    def goal(self):
        if self.role == "ball":
            self.finish(True, "goal whistle", publish=config.MSG_GOAL)

    def on_message(self, text):
        if text == config.MSG_START:
            self.start()
        elif self.role == "goalie" and self.state not in (WON, LOST):
            if text == config.MSG_CAUGHT:
                self.finish(True, "we caught the ball")
            elif text == config.MSG_GOAL:
                self.finish(False, "the ball scored")

    def on_team_message(self, text, now):
        """A glove command from our glove laptop: 'glove <degrees>'."""
        word, _, value = text.partition(" ")
        if self.role != "goalie" or word != config.GLOVE_CMD:
            return  # includes our own 'state ...' reports echoed back
        try:
            angle = float(value)
        except ValueError:
            print(f"Ignoring bad team message {text!r}")
            return
        self.glove_heard = now
        if self.state == DRIVING:  # the glove only moves while the game is on
            self.car.move_glove(angle)

    def sensor_tripped(self, now):
        refl = self.car.reflection()
        if self.role != "ball" or math.isnan(refl) or refl - self.baseline < config.REFLECT_DELTA:
            self.high_since = None
            return False
        self.high_since = self.high_since or now
        return now - self.high_since >= config.REFLECT_HOLD

    def update(self, decision, now):
        """Apply one policy decision. Motors only move while DRIVING."""
        if self.state != DRIVING:
            self.car.drive(0, 0)
            return
        if self.sensor_tripped(now):
            self.caught()
        elif decision.goal:
            self.goal()
        else:
            self.car.drive(*self.wheels(decision))

    def wheels(self, decision):
        """Left/right motor % for a decision.

        The ball steers like a car. The goalie only slides back and forth
        along the goal line: above the middle note = forward, below = backward,
        both wheels always equal so it never turns.
        """
        if self.role == "ball":
            return decision.left, decision.right
        speed = decision.steer * config.GOALIE_SPEED * config.GOALIE_DIRECTION
        return speed, speed


def status(game, det, decision, muted, now):
    """Headline, its colour, and the status lines for the display."""
    left, right = game.wheels(decision)
    if game.state == DRIVING:
        headline, colour = decision.label.split("  ")[0], "green"
        if decision.label.startswith("STOP"):
            headline, colour = "STOP", "red"
        elif game.role == "goalie":
            move = "FORWARD" if decision.steer > 0 else "BACKWARD"
            headline = f"{move} {abs(decision.steer):.0%}" if decision.steer else "HOLD"
        elif decision.goal_progress:
            colour = "purple"
    elif game.state == WAITING:
        headline, colour = "WAITING", "gray"
    else:
        headline, colour = game.state, ("green" if game.state == WON else "red")

    lines = [f"state    {game.state}  {game.result}", *hearing_lines(det, muted),
             f"decision {decision.label}   steer {decision.steer:+.2f}",
             f"motors   L {left:5.0f}%   R {right:5.0f}%"
             + ("" if game.state == DRIVING else "   (held at 0)")]
    if game.role == "ball":
        lines.append(f"sensor   reflection {game.car.reflection():.0f}  (baseline "
                     f"{game.baseline:.0f}, trip +{config.REFLECT_DELTA})")
    else:
        heard = ("never" if game.glove_heard is None
                 else f"{now - game.glove_heard:.1f} s ago")
        lines.append(f"glove    {game.car.glove_angle:+4d} deg   last command from glove "
                     f"laptop: {heard}")
    lines.append("keys     s start  r reset  c caught  g goal  q quit")
    return headline, colour, lines


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", choices=("ball", "goalie"), default="ball")
    parser.add_argument("--calibrate", action="store_true", help="measure your whistle and exit")
    parser.add_argument("--no-motor", action="store_true", help="don't connect the Double Motor")
    parser.add_argument("--no-sensor", action="store_true", help="don't connect the Color Sensor")
    parser.add_argument("--no-glove", action="store_true", help="don't connect the glove motor")
    args = parser.parse_args()
    if args.calibrate:
        run_calibration()
        return

    goalie = args.role == "goalie"
    bands = load_bands()
    policy = wp.Policy(bands, config.BASE_SPEED, config.TURN_GAIN, config.GOAL_HOLD)
    # The ball carries the colour sensor (the goalie has to reach it); the
    # goalie carries the glove.
    car = Car(motor=not args.no_motor, sensor=not goalie and not args.no_sensor,
              glove=goalie and not args.no_glove)
    pa = pyaudio.PyAudio()
    mic = Mic(pa)
    player = Player(pa)
    radio = Radio([config.TOPIC, config.TEAM_TOPIC] if goalie else [config.TOPIC])
    game = Game(args.role, car, radio, player, policy)

    try:
        if not car.connect():
            return
        radio.connect()
        floor = measure_floor(mic)
        labels = (("BACKWARD", "HOLD", "FORWARD", None) if goalie
                  else ("RIGHT", "STRAIGHT", "LEFT", "GOAL"))
        display = Display(bands, floor, f"Whistle Soccer ({args.role})", band_labels=labels)
        keys = queue.Queue()
        display.on_key(keys.put)
        display.close_on_ctrl_c()
        print(f"Ready as {args.role.upper()}. Waiting for '{config.MSG_START}' on "
              f"{config.TOPIC} (or press s).")
        last_state, last_report = None, 0.0

        while display.alive():
            now = time.monotonic()
            for topic, text in radio.messages():
                if topic == config.TEAM_TOPIC:
                    game.on_team_message(text, now)
                else:
                    print(f"MQTT <- [{topic}] {text}")
                    game.on_message(text)
            while not keys.empty():
                key = keys.get()
                if key == "q":
                    display.close()
                elif key == "s":
                    game.start()
                elif key == "r":
                    game.reset()
                elif key in ("c", "g"):
                    # Pretend it happened: the ball publishes and sings, the
                    # goalie reacts as if the ball's message had arrived.
                    if args.role == "ball":
                        game.caught() if key == "c" else game.goal()
                    else:
                        game.on_message(config.MSG_CAUGHT if key == "c" else config.MSG_GOAL)
            if not display.alive():
                break

            block = mic.block()
            det = wp.detect_pitch(block, floor)
            muted = player.playing
            # Self-mute: while our own song plays, the mic would hear the speaker.
            decision = policy.step(None if muted else det.pitch, now)
            game.update(decision, now)

            # Keep the glove laptop's screen in sync: report every change, plus
            # a heartbeat so it knows we are still alive.
            if goalie and (game.state != last_state or now - last_report >= STATE_EVERY):
                radio.publish(f"{config.STATE_CMD} {game.state}", topic=config.TEAM_TOPIC,
                              quiet=game.state == last_state)
                last_state, last_report = game.state, now

            display.update(block, det, *status(game, det, decision, muted, now))
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        car.close()
        radio.close()
        mic.close()
        player.wait()
        pa.terminate()


if __name__ == "__main__":
    main()
