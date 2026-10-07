import math
import queue
import time

import pytest

from pingpong.events import ImuSample
from pingpong.imu_worker import ImuWorker
from pingpong.swing import SwingDetector, SwingParams

T0 = 5_000_000_000
HZ = 66.0
GPD = 10.0


def swing_samples(start=0.6, peak=800.0, dur=0.15, total=2.0):
    out = []
    for i in range(int(total * HZ) + 1):
        t = i / HZ
        s = peak * math.sin(math.pi * (t - start) / dur) if start <= t <= start + dur else 0.0
        out.append(ImuSample(t_ns=T0 + int(t * 1e9), g=(round(s * GPD), 0, 0), a=(0, 0, 1000)))
    return out


def params(t_pk=250.0):
    return SwingParams(u_fwd=(1, 0, 0), gyro_per_dps=GPD, t_pk=t_pk)


def make(**kw):
    q = queue.SimpleQueue()
    return ImuWorker(q, SwingDetector(params(**kw))), q


def feed(q, samples):
    for s in samples:
        q.put_nowait(s)


def test_queued_samples_become_swing_events():
    worker, q = make()
    feed(q, swing_samples())
    assert worker.step() > 100
    kinds = [e.kind for e in worker.poll()]
    assert kinds.count("IMPACT") == 1 and kinds[0] == "SWING_START"
    assert worker.poll() == []                       # events are handed out once


def test_a_blank_window_makes_the_detector_ignore_that_stretch():
    worker, q = make()
    worker.blank(T0 + int(0.4e9), T0 + int(1.2e9))
    feed(q, swing_samples())
    worker.step()
    assert worker.poll() == []


def test_new_calibration_params_swap_the_detector_atomically():
    worker, q = make(t_pk=250.0)
    worker.set_params(params(t_pk=2000.0))           # the swing is now far too weak to count
    feed(q, swing_samples())
    worker.step()
    assert [e for e in worker.poll() if e.kind == "IMPACT"] == []


def test_step_processes_at_most_the_samples_available_and_reports_zero_when_empty():
    worker, q = make()
    assert worker.step() == 0


def test_the_thread_turns_samples_into_events_and_stops_cleanly():
    worker, q = make()
    worker.start()
    feed(q, swing_samples())
    deadline = time.monotonic() + 3.0
    got = []
    while not any(e.kind == "IMPACT" for e in got) and time.monotonic() < deadline:
        got += worker.poll()
        time.sleep(0.01)
    worker.stop()
    assert any(e.kind == "IMPACT" for e in got)
    assert not worker._thread.is_alive()


def test_blank_can_be_called_from_another_thread_while_samples_flow():
    import threading

    worker, q = make()
    worker.start()
    stop = threading.Event()

    def blanker():
        k = 0
        while not stop.is_set():
            worker.blank(T0 + k, T0 + k + 1000)
            k += 1000

    t = threading.Thread(target=blanker)
    t.start()
    for _ in range(20):
        feed(q, swing_samples(total=1.0))
        time.sleep(0.01)
    stop.set()
    t.join()
    worker.stop()                                    # reaching here without an exception is the test
