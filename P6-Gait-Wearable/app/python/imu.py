"""Decode the sketch's IMU batches and keep a short ring buffer of samples.

Wire format (see sketch.ino): byte 0 sequence number, byte 1 sample count, byte 2 FIFO reset counter (a change
means the samples are not contiguous with the previous batch), then 12 bytes per sample (ax ay az gx gy gz as
big-endian int16, raw). Full scale is +-8 g and +-1000 deg/s.
"""
import struct
from collections import deque

RATE = 100
HEADER = 3
BYTES_PER_SAMPLE = 12
ACCEL_G_PER_LSB = 8.0 / 32768
GYRO_DPS_PER_LSB = 1000.0 / 32768


def decode(batch):
    """-> (seq, resets, [(ax, ay, az, gx, gy, gz), ...]) in g and deg/s. Raises ValueError on a malformed batch."""
    if isinstance(batch, list):
        batch = bytes(batch)
    if len(batch) < HEADER:
        raise ValueError("batch shorter than its header")
    seq, n, resets = batch[0], batch[1], batch[2]
    if len(batch) != HEADER + n * BYTES_PER_SAMPLE:
        raise ValueError("batch says %d samples but carries %d bytes" % (n, len(batch) - HEADER))
    raw = struct.unpack(">%dh" % (6 * n), batch[HEADER:])
    samples = [
        (raw[i] * ACCEL_G_PER_LSB, raw[i + 1] * ACCEL_G_PER_LSB, raw[i + 2] * ACCEL_G_PER_LSB,
         raw[i + 3] * GYRO_DPS_PER_LSB, raw[i + 4] * GYRO_DPS_PER_LSB, raw[i + 5] * GYRO_DPS_PER_LSB)
        for i in range(0, 6 * n, 6)
    ]
    return seq, resets, samples


class Ring:
    """The last `seconds` of samples; counts sequence-number gaps (a batch or more went missing)."""

    def __init__(self, seconds=5.0, rate=RATE):
        self.rate = rate
        self._buf = deque(maxlen=int(seconds * rate))
        self._seq = None
        self.total = 0
        self.dropped = 0

    def push(self, seq, samples):
        if self._seq is not None and seq != (self._seq + 1) % 256:
            self.dropped += 1
        self._seq = seq
        self._buf.extend(samples)
        self.total += len(samples)

    def clear(self):
        """Forget the buffered samples (the sensor stream restarted); counters are kept."""
        self._buf.clear()

    def window(self, seconds=2.0):
        """The newest `seconds` of samples as a list, or None until that many have arrived."""
        n = int(round(seconds * self.rate))
        if len(self._buf) < n:
            return None
        return list(self._buf)[-n:]
