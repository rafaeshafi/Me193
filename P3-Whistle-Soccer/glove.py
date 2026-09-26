"""P3 Whistle Soccer, GLOVE laptop: aim the goalie's glove by whistling.

The glove is a Single Motor standing upright on the goalie robot. That motor is
connected over Bluetooth to the ROBOT laptop (whistle_car.py --role goalie);
this laptop only listens to its own microphone and sends glove angles to the
robot laptop over MQTT. The robot laptop sends the game state back.

Same whistle bands as driving: middle note = glove to centre, higher = swing
left, lower = swing right (further from the middle = further round), silence =
hold where it is.

    python glove.py --calibrate     # measure your whistle once, on this laptop
    python glove.py                 # game day

Keys in the plot window: c = centre the glove, q = quit.
"""

import argparse
import queue
import time

import matplotlib.pyplot as plt
import pyaudio

import config
import whistle_policy as wp
from live_audio import Display, Mic, hearing_lines, load_bands, measure_floor, run_calibration
from songs import Player
from whistle_car import DRIVING, LOST, WON, Radio

RESEND = 0.5        # s: repeat the current angle this often while whistling
ROBOT_TIMEOUT = 3.0  # s without a state report before we warn the robot is gone


class GloveAim:
    """Policy decision -> glove angle to send, or None when nothing to send."""

    def __init__(self):
        self.sent = None
        self.sent_at = -1e9

    def step(self, decision, now):
        steering = not decision.label.startswith("STOP") and not decision.goal_progress
        if not steering:
            return None  # silence (or a stray goal-band note): the glove holds
        step = config.GLOVE_STEP
        angle = int(round(decision.steer * config.GLOVE_MAX_DEG / step) * step)
        # Send on a real change, or periodically so a lost message heals itself.
        if angle != self.sent or now - self.sent_at >= RESEND:
            self.sent, self.sent_at = angle, now
            return angle
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calibrate", action="store_true", help="measure your whistle and exit")
    args = parser.parse_args()
    if args.calibrate:
        run_calibration()
        return

    bands = load_bands()
    policy = wp.Policy(bands, config.BASE_SPEED, config.TURN_GAIN, config.GOAL_HOLD)
    aim = GloveAim()
    pa = pyaudio.PyAudio()
    mic = Mic(pa)
    player = Player(pa)
    radio = Radio([config.TEAM_TOPIC])
    robot_state, robot_heard = "unknown", None

    try:
        radio.connect()
        floor = measure_floor(mic)
        display = Display(bands, floor, "Whistle Soccer (glove)",
                          band_labels=("SWING RIGHT", "CENTRE", "SWING LEFT", None))
        keys = queue.Queue()
        display.on_key(keys.put)
        display.close_on_ctrl_c()
        print(f"Ready. Sending glove angles on {config.TEAM_TOPIC}.")

        while display.alive():
            now = time.monotonic()
            for _, text in radio.messages():
                word, _, value = text.partition(" ")
                if word != config.STATE_CMD:
                    continue  # our own glove commands echoed back
                if value != robot_state:
                    print(f"robot is now {value}")
                    if value in (WON, LOST):  # celebrate / mourn with the robot
                        player.play(config.VICTORY_SONG if value == WON else config.DEATH_SONG)
                robot_state, robot_heard = value, now
            while not keys.empty():
                key = keys.get()
                if key == "q":
                    display.close()
                elif key == "c":
                    radio.publish(f"{config.GLOVE_CMD} 0", topic=config.TEAM_TOPIC)
                    aim.sent, aim.sent_at = 0, now
            if not display.alive():
                break

            block = mic.block()
            det = wp.detect_pitch(block, floor)
            muted = player.playing
            decision = policy.step(None if muted else det.pitch, now)
            angle = aim.step(decision, now)
            if angle is not None:
                radio.publish(f"{config.GLOVE_CMD} {angle}", topic=config.TEAM_TOPIC,
                              quiet=True)

            robot_alive = robot_heard is not None and now - robot_heard < ROBOT_TIMEOUT
            if not robot_alive:
                headline, colour = "NO ROBOT", "red"
            elif robot_state != DRIVING:
                headline, colour = robot_state.split()[0], "gray"
            elif decision.label.startswith("STOP"):
                headline, colour = "HOLD", "gray"
            else:
                headline, colour = f"GLOVE {aim.sent or 0:+d}°", "green"
            heard = "never" if robot_heard is None else f"{now - robot_heard:.1f} s ago"
            lines = [
                f"robot    {robot_state}   (last report {heard})",
                *hearing_lines(det, muted),
                f"decision {decision.label}   steer {decision.steer:+.2f}",
                f"glove    sent {aim.sent if aim.sent is not None else '-'} deg  "
                f"(+-{config.GLOVE_MAX_DEG} max, silence = hold)"
                + ("" if robot_state == DRIVING else "   robot ignores it until start"),
                "keys     c centre glove  q quit",
            ]
            display.update(block, det, headline, colour, lines)
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        radio.close()
        mic.close()
        player.wait()
        pa.terminate()


if __name__ == "__main__":
    main()
