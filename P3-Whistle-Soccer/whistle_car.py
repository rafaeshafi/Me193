"""P3 Whistle Soccer: steer a LEGO Double Motor car by whistling.

Middle whistle drives straight, higher turns left, lower turns right, silence
stops. The car waits for "start" on MQTT. As the ball, it dies (publish + death
song) if the goalie reaches its colour sensor, and scores (publish + victory
song) on a held top-note whistle. As the goalie, it plays the opposite songs.

    python whistle_car.py --calibrate         # measure your whistle once
    python whistle_car.py --role ball         # game day
    python whistle_car.py --role goalie
    python whistle_car.py --no-motor --no-sensor   # audio + display only

Keys in the plot window: s = start locally, r = reset, c / g = pretend
caught / goal (tests the messages and songs), q = quit.
Messages, songs, card and speeds are all in config.py.
"""

import argparse
import json
import math
import queue
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
import pyaudio

import config
import whistle_policy as wp
from songs import Player

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
SEND_INTERVAL = 0.08  # seconds between BLE motor writes (~12/s)
SEND_DELTA = 4        # don't spend a BLE write on a change smaller than this
PLOT_MAX_HZ = 4000


# --- Microphone -------------------------------------------------------------

class Mic:
    """PyAudio input stream that always holds the most recent BLOCK samples.

    The callback runs on PyAudio's thread; the main loop just takes whatever
    is newest, so a slow redraw never builds up a backlog of old audio.
    """

    def __init__(self, pa):
        self.latest = np.zeros(wp.BLOCK, dtype=np.float32)
        self.stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=wp.SAMPLE_RATE,
                              input=True, frames_per_buffer=wp.BLOCK // 2,
                              stream_callback=self._callback)

    def _callback(self, data, frames, time_info, status):
        chunk = np.frombuffer(data, dtype=np.float32)
        self.latest = np.concatenate((self.latest[len(chunk):], chunk))
        return None, pyaudio.paContinue

    def block(self):
        return self.latest.copy()

    def collect(self, seconds):
        """Grab non-overlapping blocks for `seconds` (used for calibration)."""
        blocks, end = [], time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(wp.BLOCK / wp.SAMPLE_RATE)
            blocks.append(self.block())
        return blocks

    def close(self):
        self.stream.stop_stream()
        self.stream.close()


def measure_floor(mic):
    print("Measuring room noise for 1 s -- stay quiet...")
    time.sleep(0.2)  # let the buffer fill with real audio
    return wp.noise_floor(mic.collect(1.0))


# --- LEGO hardware ----------------------------------------------------------

class Car:
    """Double Motor + Color Sensor, or stand-ins when disabled.

    Rate-limits BLE writes and sends them non-blocking, so the audio loop never
    waits on Bluetooth. (Same approach as P1-Pose-Race.)
    """

    def __init__(self, *, motor=True, sensor=True):
        self.use_motor, self.use_sensor = motor, sensor
        self.motor = self.sensor = None
        self._last_sent = None
        self._last_send_time = 0.0

    def connect(self):
        if not (self.use_motor or self.use_sensor):
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

        if self.use_motor:
            self.motor = le.DoubleMotor()
            self.motor.connect(**card)
            if not self.motor.connected:
                print("Double Motor not found. Is it on and nearby?")
                return False
            print("  Double Motor connected.")
        if self.use_sensor:
            self.sensor = le.ColorSensor()
            self.sensor.connect(**card)
            if not self.sensor.connected:
                print("Color Sensor not found. Is it on and nearby?")
                return False
            print("  Color Sensor connected.")
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
            self.motor.movement_move_tank(left * config.MOTOR_DIRECTION,
                                          right * config.MOTOR_DIRECTION, blocking=False)

    def stop(self):
        if self.motor is not None:
            self.motor.movement_stop(blocking=False)
        self._last_sent = (0, 0)

    def close(self):
        for device, teardown in ((self.motor, "movement_stop"), (self.motor, "disconnect"),
                                 (self.sensor, "disconnect")):
            if device is not None:
                try:
                    getattr(device, teardown)()
                except Exception:
                    pass  # link may already be gone; keep tearing down regardless


# --- MQTT -------------------------------------------------------------------

class Radio:
    """Subscribes to the game topic; incoming messages land in a queue."""

    def __init__(self):
        self.inbox = queue.Queue()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = lambda c, *_: c.subscribe(config.TOPIC)
        self.client.on_message = lambda c, u, msg: self.inbox.put(msg.payload.decode().strip())

    def connect(self):
        try:
            self.client.connect(config.BROKER, config.PORT)
        except OSError as e:
            print(f"MQTT: could not reach {config.BROKER} ({e}). Press s to start locally.")
            return
        self.client.loop_start()
        print(f"MQTT: listening on '{config.TOPIC}' at {config.BROKER}")

    def publish(self, text):
        print(f"MQTT -> [{config.TOPIC}] {text}")
        self.client.publish(config.TOPIC, text)

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

    def start(self):
        if self.state == DRIVING:
            return
        refl = self.car.reflection()
        self.baseline = 0.0 if math.isnan(refl) else refl
        self.high_since = None
        self.state, self.result = DRIVING, ""
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
            self.car.drive(decision.left, decision.right)


# --- Live display -----------------------------------------------------------

class Display:
    def __init__(self, bands, floor_db, role):
        plt.ion()
        self.fig, (self.ax_wave, self.ax_spec, self.ax_text) = plt.subplots(
            3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2, 1]})
        self.fig.canvas.manager.set_window_title(f"Whistle Soccer ({role})")

        t = np.arange(wp.BLOCK) / wp.SAMPLE_RATE * 1000
        (self.wave,) = self.ax_wave.plot(t, np.zeros(wp.BLOCK), lw=0.8)
        self.ax_wave.set(ylim=(-0.5, 0.5), xlim=(0, t[-1]), xlabel="ms",
                         ylabel="amplitude", title="Microphone signal")

        shown = wp.FREQS <= PLOT_MAX_HZ
        self.shown = shown
        b = bands
        for lo, hi, colour, label in (
                (b.f_min, b.f_center - b.dead_band, "tab:blue", "RIGHT"),
                (b.f_center - b.dead_band, b.f_center + b.dead_band, "tab:green", "STRAIGHT"),
                (b.f_center + b.dead_band, b.f_max, "tab:orange", "LEFT"),
                (b.f_goal, PLOT_MAX_HZ, "tab:purple", "GOAL")):
            self.ax_spec.axvspan(lo, hi, color=colour, alpha=0.15, label=label)
        for edge in wp.BAND:
            self.ax_spec.axvline(edge, color="gray", ls=":", lw=1)
        self.ax_spec.plot(wp.FREQS[shown], (floor_db + wp.MIN_SNR_DB)[shown], color="red",
                          lw=1, ls="--", label="noise floor + SNR gate")
        (self.spec,) = self.ax_spec.plot(wp.FREQS[shown], np.full(shown.sum(), -120.0), lw=1,
                                         color="black", label="spectrum")
        (self.peak,) = self.ax_spec.plot([], [], "o", ms=10)
        self.ax_spec.set(xlim=(0, PLOT_MAX_HZ), ylim=(-110, 0), xlabel="frequency (Hz)",
                         ylabel="dB", title="Spectrum (shaded = decision bands)")
        self.ax_spec.legend(loc="upper right", fontsize=8, ncol=3)

        self.ax_text.axis("off")
        self.status = self.ax_text.text(0.01, 0.95, "", va="top", family="monospace",
                                        fontsize=11, transform=self.ax_text.transAxes)
        self.big = self.ax_text.text(0.99, 0.5, "", ha="right", va="center", fontsize=26,
                                     weight="bold", transform=self.ax_text.transAxes)
        self.fig.tight_layout()

    def update(self, block, det, decision, game, reflection, muted):
        self.wave.set_ydata(block)
        db = wp.spectrum_db(block)
        self.spec.set_ydata(db[self.shown])
        if det.pitch is not None:
            self.peak.set_data([det.pitch], [db[np.argmin(np.abs(wp.FREQS - det.pitch))]])
            self.peak.set_color("green")
        else:
            self.peak.set_data([], [])

        if game.state == DRIVING:
            headline, colour = decision.label.split("  ")[0], "green"
            if decision.label.startswith("STOP"):
                colour = "red"
            elif decision.goal_progress:
                colour = "purple"
        elif game.state == WAITING:
            headline, colour = "WAITING", "gray"
        else:
            headline, colour = game.state, ("green" if game.state == WON else "red")
        self.big.set_text(headline)
        self.big.set_color(colour)

        pitch = f"{det.pitch:6.0f} Hz" if det.pitch else "  none   "
        heard = "muted (song playing)" if muted else det.reason
        lines = [
            f"state    {game.state}  {game.result}",
            f"pitch    {pitch}   peak {det.peak_hz:5.0f} Hz  -> {heard}",
            f"gates    rms {det.rms:.3f}  snr {det.snr_db:5.1f} dB  tonal {det.tonality_db:5.1f} dB",
            f"decision {decision.label}   steer {decision.steer:+.2f}",
            f"motors   L {decision.left:5.0f}%   R {decision.right:5.0f}%"
            + ("" if game.state == DRIVING else "   (held at 0)"),
            f"sensor   reflection {reflection:.0f}  (baseline {game.baseline:.0f}, "
            f"trip +{config.REFLECT_DELTA})",
            "keys     s start  r reset  c caught  g goal  q quit",
        ]
        self.status.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def alive(self):
        return plt.fignum_exists(self.fig.number)


# --- Calibration ------------------------------------------------------------

def load_bands():
    values = dict(f_min=config.F_MIN, f_center=config.F_CENTER, f_max=config.F_MAX,
                  f_goal=config.F_GOAL)
    if CALIBRATION_FILE.exists():
        values.update(json.loads(CALIBRATION_FILE.read_text()))
        print(f"Using calibration from {CALIBRATION_FILE.name}: {values}")
    else:
        print("No calibration.json -- using config.py defaults. Run --calibrate for your whistle.")
    return wp.Bands(dead_band=config.DEAD_BAND, **values)


def calibrate(mic):
    floor = measure_floor(mic)
    measured = {}
    for key, what in (("f_min", "your LOWEST comfortable whistle (hard right)"),
                      ("f_center", "your MIDDLE whistle (straight)"),
                      ("f_max", "your HIGHEST steering whistle (hard left)"),
                      ("goal", "the GOAL whistle: highest you can, well above the last one")):
        input(f"\nPress Enter, then whistle {what} for 2 s...")
        pitches = [d.pitch for d in (wp.detect_pitch(b, floor) for b in mic.collect(2.0))
                   if d.pitch is not None]
        if len(pitches) < 5:
            raise SystemExit("Heard almost no whistle. Whistle louder / closer, then retry.")
        measured[key] = float(np.median(pitches))
        print(f"  heard {measured[key]:.0f} Hz ({len(pitches)} blocks)")

    lo, mid, hi, goal = (measured[k] for k in ("f_min", "f_center", "f_max", "goal"))
    if not lo < mid - config.DEAD_BAND < mid + config.DEAD_BAND < hi < goal - 150:
        raise SystemExit(f"Notes too close together ({lo:.0f} / {mid:.0f} / {hi:.0f} / "
                         f"{goal:.0f} Hz). Spread them further apart and retry.")
    result = dict(f_min=round(lo), f_center=round(mid), f_max=round(hi),
                  f_goal=round((hi + goal) / 2))  # goal threshold halfway up the gap
    CALIBRATION_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved {CALIBRATION_FILE.name}: {result}")


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", choices=("ball", "goalie"), default="ball")
    parser.add_argument("--calibrate", action="store_true", help="measure your whistle and exit")
    parser.add_argument("--no-motor", action="store_true", help="don't connect the Double Motor")
    parser.add_argument("--no-sensor", action="store_true", help="don't connect the Color Sensor")
    args = parser.parse_args()

    pa = pyaudio.PyAudio()
    mic = Mic(pa)
    if args.calibrate:
        try:
            calibrate(mic)
        finally:
            mic.close()
            pa.terminate()
        return

    bands = load_bands()
    policy = wp.Policy(bands, config.BASE_SPEED, config.TURN_GAIN, config.GOAL_HOLD)
    # Only the ball needs the colour sensor: it is what the goalie has to reach.
    car = Car(motor=not args.no_motor, sensor=args.role == "ball" and not args.no_sensor)
    player = Player(pa)
    radio = Radio()
    game = Game(args.role, car, radio, player, policy)

    try:
        if not car.connect():
            return
        radio.connect()
        floor = measure_floor(mic)
        display = Display(bands, floor, args.role)
        keys = queue.Queue()
        display.fig.canvas.mpl_connect("key_press_event", lambda e: keys.put(e.key))
        print(f"Ready as {args.role.upper()}. Waiting for '{config.MSG_START}' on "
              f"{config.TOPIC} (or press s).")

        while display.alive():
            now = time.monotonic()
            for text in radio.messages():
                print(f"MQTT <- [{config.TOPIC}] {text}")
                game.on_message(text)
            while not keys.empty():
                key = keys.get()
                if key == "q":
                    plt.close(display.fig)
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
            display.update(block, det, decision, game, car.reflection(), muted)
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
