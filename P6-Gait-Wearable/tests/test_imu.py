import struct

import pytest

from imu import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB, Ring, decode


def batch(seq, samples, resets=0):
    """samples: list of 6 raw int16 tuples -> the sketch's wire format."""
    body = b"".join(struct.pack(">6h", *s) for s in samples)
    return bytes([seq, len(samples), resets]) + body


def test_decode_scales_to_g_and_dps():
    seq, resets, samples = decode(batch(7, [(4096, 0, -4096, 0, 16384, 0)], resets=3))
    assert seq == 7 and resets == 3 and len(samples) == 1
    ax, ay, az, gx, gy, gz = samples[0]
    assert ax == pytest.approx(4096 * ACCEL_G_PER_LSB) == pytest.approx(1.0)
    assert az == pytest.approx(-1.0)
    assert gy == pytest.approx(16384 * GYRO_DPS_PER_LSB) == pytest.approx(500.0)


def test_decode_accepts_list_of_ints_and_empty_batch():
    seq, resets, samples = decode(list(batch(3, [])))
    assert seq == 3 and resets == 0 and samples == []


def test_decode_rejects_malformed():
    with pytest.raises(ValueError):
        decode(b"\x01\x00")
    with pytest.raises(ValueError):
        decode(bytes([0, 2, 0]) + b"\x00" * 12)  # says 2 samples, carries 1


def test_ring_window_and_dropped_batches():
    ring = Ring(seconds=1.0, rate=10)
    assert ring.window(0.5) is None
    for seq in range(3):
        ring.push(seq, [(seq, 0, 0, 0, 0, 0)] * 4)  # 12 samples into a 10-sample ring
    assert ring.total == 12 and ring.dropped == 0
    win = ring.window(0.5)  # 5 newest samples
    assert len(win) == 5 and win[-1][0] == 2
    ring.push(5, [(5, 0, 0, 0, 0, 0)])  # seq 3 and 4 never came
    assert ring.dropped == 1
    assert ring.window(0.5)[-1][0] == 5
    ring.clear()                        # the sketch said the stream restarted
    assert ring.window(0.5) is None and ring.total == 13


def test_ring_sequence_wrap_and_clear_keep_counters():
    ring = Ring(seconds=1.0, rate=10)
    ring.push(255, [(0,) * 6])
    ring.push(0, [(0,) * 6])
    assert ring.dropped == 0
    ring.push(2, [(0,) * 6])
    assert ring.dropped == 1
    ring.clear()
    assert ring.dropped == 1 and ring.total == 3
