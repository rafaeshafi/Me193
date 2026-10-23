import struct
from pathlib import Path

from imu import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB
from models import Tree
from pipeline import Gait
from recorder import Recorder
from synth import synth

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def to_batch(seq, samples, resets=0):
    raw = []
    for ax, ay, az, gx, gy, gz in samples:
        raw.append(struct.pack(">6h", int(ax / ACCEL_G_PER_LSB), int(ay / ACCEL_G_PER_LSB), int(az / ACCEL_G_PER_LSB),
                               int(gx / GYRO_DPS_PER_LSB), int(gy / GYRO_DPS_PER_LSB), int(gz / GYRO_DPS_PER_LSB)))
    return bytes([seq & 0xFF, len(samples), resets]) + b"".join(raw)


def make(tmp_path, clk):
    return Gait(Tree.load(ROOT / "models" / "model1.json"), Tree.load(ROOT / "models" / "model2.json"),
                Recorder(tmp_path, clock=clk, rolling=False), clock=clk)


def feed(g, clk, kind, seconds):
    sig = synth(kind, seconds=seconds)
    for i in range(0, len(sig), 10):
        clk.t += 0.1
        g.on_batch(to_batch(i // 10, sig[i:i + 10]))
        g.tick()


def test_no_imu_until_samples_arrive(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    assert g.tick() is True and g.state["act"] == "---"          # nothing heard yet
    clk.t += 0.6
    g.on_batch(to_batch(0, []))                                  # heartbeat, empty: the sketch has no IMU
    g.tick()
    assert g.state["act"] == "NO IMU"
    assert "act=NO IMU" in g.payload()


def test_classifies_walk_then_bike_with_seed_models(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 3.0)
    assert g.state["act"] == "WALK" and g.state["m1"] == "WALK"
    assert 90 < g.state["cad"] < 130                              # 1.8 Hz steps -> 108 per minute
    feed(g, clk, "BIKE", 3.5)
    assert g.state["act"] == "BIKE" and g.state["m1"] in ("STAND", "WALK")
    assert "agree=0" in g.payload()


def test_fifo_reset_clears_the_window(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 2.5)
    assert g.ring.window(2.0) is not None
    clk.t += 0.1
    g.on_batch(to_batch(99, synth("WALK", seconds=0.1), resets=1))   # the sketch restarted the FIFO
    assert g.ring.window(2.0) is None and g.ring.window(0.1) is not None


def test_wave_bytes_are_downsampled_to_50_hz(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    g.on_batch(to_batch(0, synth("STAND", seconds=0.1)))          # 10 samples
    w = g.take_wave()
    assert len(w) == 5 and g.take_wave() == b""


def test_recorder_gets_every_sample_and_prompt_reaches_payload(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    g.recorder.start_single("STAND", seconds=5)
    clk.t += 6.0
    g.on_batch(to_batch(0, synth("STAND", seconds=0.1)))
    assert g.recorder.rows == 10
    assert "prompt=STAND" in g.payload()


def test_snapshot_is_json_friendly(tmp_path):
    import json
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "RUN", 2.5)
    snap = json.loads(json.dumps(g.snapshot()))
    assert snap["act"] == "RUN" and snap["recorder"]["active"] is False
    assert len(snap["wave"]) <= 150 and isinstance(snap["features"]["peak_x"], float)


def test_votes_clear_after_an_imu_gap(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 3.0)
    assert g.state["act"] == "WALK"
    clk.t += 2.0                                   # silence: NO IMU, votes dropped
    g.tick()
    assert g.state["act"] == "NO IMU"
    sig = synth("BIKE", seconds=2.5)
    for i in range(0, len(sig), 10):
        clk.t += 0.1
        g.on_batch(to_batch(100 + i // 10, sig[i:i + 10], resets=1))
        g.tick()
    assert g.state["act"] == "BIKE"               # the first classified window after the gap wins outright


def test_no_imu_after_samples_then_silence(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "STAND", 2.5)
    assert g.state["act"] == "STAND"
    clk.t += 1.6
    g.tick()
    assert g.state["act"] == "NO IMU" and g.state["features"] == {}


def test_on_batch_from_a_thread_while_ticking(tmp_path):
    import threading
    clk = Clock()
    g = make(tmp_path, clk)
    sig = synth("RUN", seconds=4.0)
    errors = []

    def feeder():
        try:
            for i in range(0, len(sig), 10):
                clk.t += 0.1
                g.on_batch(to_batch(i // 10, sig[i:i + 10]))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    th = threading.Thread(target=feeder)
    th.start()
    while th.is_alive():
        g.tick()
        snap = g.snapshot()
        assert snap["act"] in ("---", "NO IMU", "STAND", "WALK", "RUN", "BIKE")
    th.join()
    assert not errors
    g.tick()
    assert g.state["act"] == "RUN"
