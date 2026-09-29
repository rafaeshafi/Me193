"""P3 Whistle Soccer, GLOVE laptop: aim the goalie's glove by whistling.

The glove is a Single Motor standing upright on the goalie robot, with a big
LEGO piece on it. This laptop connects to that motor over Bluetooth and turns
its own microphone's whistle into glove angles. The ROBOT laptop
(whistle_car.py --role goalie) drives the wheels; the two talk over MQTT:
the robot laptop sends the game state (the glove only moves while it says
DRIVING), and this laptop reports the glove angle back.

Same whistle bands as driving: middle note = glove to centre, higher = swing
left, lower = swing right (further from the middle = further round), silence =
hold where it is.

    python glove.py --calibrate     # measure your whistle once, on this laptop
    python glove.py                 # game day
    python glove.py --solo          # practise: ignore the robot laptop's state
    python glove.py --no-motor      # no Bluetooth: audio + display only

Keys in the plot window: c = centre the glove, q = quit.
"""

import argparse
import queue
import time

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

import config
import whistle_policy as wp
from live_audio import Display, Mic, hearing_lines, load_bands, measure_floor, run_calibration
from songs import Player
from whistle_car import DRIVING, LOST, WON, Radio, lego_card

REPORT_EVERY = 1.0   # s between glove-angle reports to the robot laptop
ROBOT_TIMEOUT = 3.0  # s without a state report before we treat the robot as gone


class GloveMotor:
    """The glove's Single Motor in position mode, or a stand-in."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.motor = None
        self.angle = 0

    def connect(self):
        if not self.enabled:
            print("--no-motor: no Bluetooth, glove angles are only shown.")
            return True
        import legoeducation as le

        card = lego_card(le)
        if card is None:
            return False
        self.motor = le.SingleMotor()
        self.motor.connect(**card)
        if not self.motor.connected:
            print("Glove Single Motor not found. Is it on and nearby?")
            self.motor = None
            return False
        # Wherever the glove points now is "centre", and it holds its angle
        # between commands instead of flopping around.
        self.motor.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
        self.motor.motor_reset_relative_position()
        print("  Glove Single Motor connected.")
        return True

    def move(self, angle):
        angle = int(np.clip(angle, -config.GLOVE_MAX_DEG, config.GLOVE_MAX_DEG))
        if angle == self.angle:
            return
        self.angle = angle
        if self.motor is not None:
            self.motor.motor_run_to_relative_position(angle * config.GLOVE_DIRECTION,
                                                      speed=config.GLOVE_SPEED, blocking=False)

    def close(self):
        if self.motor is None:
            return
        for teardown in (self.motor.motor_stop, self.motor.disconnect):
            try:
                teardown()
            except Exception:
                pass  # link may already be gone; keep tearing down regardless


def glove_target(decision):
    """Policy decision -> glove angle, or None to hold where it is.

    Rounded to GLOVE_STEP so tiny pitch wobbles don't twitch the motor.
    """
    if decision.label.startswith("STOP") or decision.goal_progress:
        return None  # silence (or a stray goal-band note): the glove holds
    step = config.GLOVE_STEP
    return int(round(decision.steer * config.GLOVE_MAX_DEG / step) * step)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calibrate", action="store_true", help="measure your whistle and exit")
    parser.add_argument("--no-motor", action="store_true", help="don't connect the glove motor")
    parser.add_argument("--solo", action="store_true",
                        help="practise: move the glove without waiting for the robot laptop")
    args = parser.parse_args()
    if args.calibrate:
        run_calibration()
        return

    bands = load_bands()
    policy = wp.Policy(bands, config.BASE_SPEED, config.TURN_GAIN, config.GOAL_HOLD)
    glove = GloveMotor(enabled=not args.no_motor)
    pa = pyaudio.PyAudio()
    mic = Mic(pa)
    player = Player(pa)
    radio = Radio([config.TEAM_TOPIC])
    robot_state, robot_heard = "unknown", None

    try:
        if not glove.connect():
            return
        radio.connect()
        floor = measure_floor(mic)
        display = Display(bands, floor, "Whistle Soccer (glove)",
                          band_labels=("SWING RIGHT", "CENTRE", "SWING LEFT", None))
        keys = queue.Queue()
        display.on_key(keys.put)
        display.close_on_ctrl_c()
        print(f"Ready. Talking to the robot laptop on {config.TEAM_TOPIC}.")
        reported, reported_at = None, -1e9

        while display.alive():
            now = time.monotonic()
            for _, text in radio.messages():
                word, _, value = text.partition(" ")
                if word != config.STATE_CMD:
                    continue  # our own glove reports echoed back
                if value != robot_state:
                    print(f"robot is now {value}")
                    if value == DRIVING:
                        glove.move(0)  # kick-off: start from the centre
                    elif value in (WON, LOST):  # celebrate / mourn with the robot
                        player.play(config.VICTORY_SONG if value == WON else config.DEATH_SONG)
                robot_state, robot_heard = value, now
            while not keys.empty():
                key = keys.get()
                if key == "q":
                    display.close()
                elif key == "c":
                    glove.move(0)
            if not display.alive():
                break

            block = mic.block()
            det = wp.detect_pitch(block, floor)
            muted = player.playing
            decision = policy.step(None if muted else det.pitch, now)
            target = glove_target(decision)

            robot_alive = robot_heard is not None and now - robot_heard < ROBOT_TIMEOUT
            game_on = args.solo or (robot_alive and robot_state == DRIVING)
            if target is not None and game_on:
                glove.move(target)

            # Report where the glove is: on every change, plus a heartbeat.
            if glove.angle != reported or now - reported_at >= REPORT_EVERY:
                radio.publish(f"{config.GLOVE_CMD} {glove.angle}", topic=config.TEAM_TOPIC,
                              quiet=True)
                reported, reported_at = glove.angle, now

            if not game_on:
                headline, colour = (("NO ROBOT", "red") if not robot_alive
                                    else (robot_state.split()[0], "gray"))
            elif target is None:
                headline, colour = f"HOLD {glove.angle:+d}°", "gray"
            else:
                headline, colour = f"GLOVE {glove.angle:+d}°", "green"
            heard = "never" if robot_heard is None else f"{now - robot_heard:.1f} s ago"
            lines = [
                f"robot    {robot_state}   (last report {heard})"
                + ("   [--solo: ignored]" if args.solo else ""),
                *hearing_lines(det, muted),
                f"decision {decision.label}   steer {decision.steer:+.2f}",
                f"glove    at {glove.angle:+d} deg  (+-{config.GLOVE_MAX_DEG} max, "
                f"silence = hold)" + ("" if game_on else "   frozen until the robot says DRIVING"),
                "keys     c centre glove  q quit",
            ]
            display.update(block, det, headline, colour, lines)
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        glove.close()
        radio.close()
        mic.close()
        player.wait()
        pa.terminate()


if __name__ == "__main__":
    main()
