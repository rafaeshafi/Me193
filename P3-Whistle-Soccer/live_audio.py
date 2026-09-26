"""Shared by both laptops: microphone stream, live display, calibration."""

import json
import signal
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

import config
import whistle_policy as wp

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
PLOT_MAX_HZ = 4000


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
    print(f"Measuring room noise for {config.NOISE_SECONDS:g} s -- stay quiet...")
    time.sleep(0.2)  # let the buffer fill with real audio
    return wp.noise_floor(mic.collect(config.NOISE_SECONDS))


class Display:
    """Waveform, spectrum with the decision bands shaded, and a status panel."""

    def __init__(self, bands, floor_db, title, band_labels=("RIGHT", "STRAIGHT", "LEFT", "GOAL")):
        plt.ion()
        self.fig, (self.ax_wave, self.ax_spec, self.ax_text) = plt.subplots(
            3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2, 1]})
        self.fig.canvas.manager.set_window_title(title)

        t = np.arange(wp.BLOCK) / wp.SAMPLE_RATE * 1000
        (self.wave,) = self.ax_wave.plot(t, np.zeros(wp.BLOCK), lw=0.8)
        self.ax_wave.set(ylim=(-0.5, 0.5), xlim=(0, t[-1]), xlabel="ms",
                         ylabel="amplitude", title="Microphone signal")

        self.shown = wp.FREQS <= PLOT_MAX_HZ
        b = bands
        spans = ((b.f_min, b.f_center - b.dead_band, "tab:blue"),
                 (b.f_center - b.dead_band, b.f_center + b.dead_band, "tab:green"),
                 (b.f_center + b.dead_band, b.f_max, "tab:orange"),
                 (b.f_goal, PLOT_MAX_HZ, "tab:purple"))
        for (lo, hi, colour), label in zip(spans, band_labels):
            if label:
                self.ax_spec.axvspan(lo, hi, color=colour, alpha=0.15, label=label)
        for edge in wp.BAND:
            self.ax_spec.axvline(edge, color="gray", ls=":", lw=1)
        self.ax_spec.plot(wp.FREQS[self.shown], (floor_db + wp.MIN_SNR_DB)[self.shown],
                          color="red", lw=1, ls="--", label="noise floor + SNR gate")
        (self.spec,) = self.ax_spec.plot(wp.FREQS[self.shown], np.full(self.shown.sum(), -120.0),
                                         lw=1, color="black", label="spectrum")
        (self.peak,) = self.ax_spec.plot([], [], "o", ms=10, color="green")
        self.ax_spec.set(xlim=(0, PLOT_MAX_HZ), ylim=(-110, 0), xlabel="frequency (Hz)",
                         ylabel="dB", title="Spectrum (shaded = decision bands)")
        self.ax_spec.legend(loc="upper right", fontsize=8, ncol=3)

        self.ax_text.axis("off")
        self.status = self.ax_text.text(0.01, 0.95, "", va="top", family="monospace",
                                        fontsize=11, transform=self.ax_text.transAxes)
        self.big = self.ax_text.text(0.99, 0.98, "", ha="right", va="top", fontsize=24,
                                     weight="bold", transform=self.ax_text.transAxes)
        self.fig.tight_layout()

    def close_on_ctrl_c(self):
        """Ctrl+C in the terminal closes the window, so the program shuts down
        cleanly (matplotlib's event loop would otherwise swallow it)."""
        signal.signal(signal.SIGINT, lambda *_: self.close())

    def on_key(self, callback):
        self.fig.canvas.mpl_connect("key_press_event", lambda e: callback(e.key))

    def update(self, block, det, headline, colour, lines):
        self.wave.set_ydata(block)
        db = wp.spectrum_db(block)
        self.spec.set_ydata(db[self.shown])
        if det.pitch is not None:
            self.peak.set_data([det.pitch], [db[np.argmin(np.abs(wp.FREQS - det.pitch))]])
        else:
            self.peak.set_data([], [])
        self.big.set_text(headline)
        self.big.set_color(colour)
        self.status.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)

    def alive(self):
        return plt.fignum_exists(self.fig.number)


def hearing_lines(det, muted):
    """The two status lines both laptops show about what the mic heard."""
    pitch = f"{det.pitch:6.0f} Hz" if det.pitch else "  none   "
    heard = "muted (song playing)" if muted else det.reason
    return [
        f"pitch    {pitch}   peak {det.peak_hz:5.0f} Hz  -> {heard}",
        f"gates    rms {det.rms:.3f} (need {wp.MIN_RMS:g})  snr {det.snr_db:5.1f} dB "
        f"(need {wp.MIN_SNR_DB:g})  tonal {det.tonality_db:5.1f} dB",
    ]


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
                      ("f_center", "your MIDDLE whistle (straight / centre)"),
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


def run_calibration():
    pa = pyaudio.PyAudio()
    mic = Mic(pa)
    try:
        calibrate(mic)
    finally:
        mic.close()
        pa.terminate()
