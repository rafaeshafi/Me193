"""Record from the microphone and show the sound as a spectrogram.

Usage:
    python Spectrogram.py [seconds]      # default 3 seconds

Whistle, hum, or talk while it records. The plot shows frequency (Hz)
against time, with color as loudness in dB.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100   # Hz, plenty for a whistle (a few kHz at most)
MAX_PLOT_HZ = 5000    # zoom the plot onto the range whistles live in
COLORMAP = "turbo"    # try "nipy_spectral", "rainbow", "viridis", "magma"
DB_RANGE = (-100, -20)  # quiet -> loud, in dB; lower vmin for a darker plot


def record(seconds):
    print(f"Recording {seconds:g} s -- make some noise...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print("Done.")
    return audio[:, 0]


def plot_spectrogram(audio):
    plt.specgram(audio, NFFT=1024, Fs=SAMPLE_RATE, noverlap=512,
                 cmap=COLORMAP, vmin=DB_RANGE[0], vmax=DB_RANGE[1])
    plt.ylim(0, MAX_PLOT_HZ)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    audio = record(seconds)
    peak_bin = np.abs(np.fft.rfft(audio)).argmax()
    print(f"Loudest frequency: {np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[peak_bin]:.0f} Hz")
    plot_spectrogram(audio)
