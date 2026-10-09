"""FramePipe: the next picture is drawn on a thread of its own while the main thread sits in the window's wait."""

import threading
import time

import pytest

from pingpong.framepipe import FramePipe


def test_nothing_is_drawn_until_something_is_submitted():
    pipe = FramePipe(lambda x: x * 2)
    try:
        assert pipe.wait(0.01) is None
        pipe.submit(21)
        assert pipe.wait(1.0) == 42
        assert pipe.wait(0.01) is None                             # a frame is handed over once
    finally:
        pipe.close()


def test_wait_blocks_until_the_picture_being_drawn_is_finished():
    gate = threading.Event()

    def draw(x):
        gate.wait(2.0)
        return x

    pipe = FramePipe(draw)
    try:
        pipe.submit("a")
        assert pipe.wait(0.05) is None                             # still being drawn, and the wait ran out
        gate.set()
        assert pipe.wait(1.0) == "a"
    finally:
        gate.set()
        pipe.close()


def test_the_newest_submission_wins_when_the_drawer_is_behind():
    started, gate, drawn = threading.Event(), threading.Event(), []

    def draw(x):
        drawn.append(x)
        started.set()
        gate.wait(2.0)
        return x

    pipe = FramePipe(draw)
    try:
        pipe.submit(1)
        assert started.wait(1.0)                                   # 1 is being drawn
        pipe.submit(2)
        pipe.submit(3)                                             # 2 was never wanted
        gate.set()
        assert pipe.wait(2.0) == 3
        assert drawn == [1, 3]
    finally:
        gate.set()
        pipe.close()


def test_drawing_overlaps_with_the_main_threads_own_waiting():
    def draw(x):
        time.sleep(0.02)                                           # a picture takes 20 ms ...
        return x

    pipe = FramePipe(draw)
    try:
        t0, shown = time.perf_counter(), []
        for k in range(10):
            frame = pipe.wait(1.0)
            if frame is not None:
                shown.append(frame)
            pipe.submit(k)
            time.sleep(0.02)                                       # ... and so does the window's wait
        elapsed = time.perf_counter() - t0
        assert len(shown) == 9
        assert elapsed < 0.30                                      # one after the other it would be 0.40 s
    finally:
        pipe.close()


def test_an_error_while_drawing_comes_out_of_wait_once_and_the_pipe_goes_on():
    def draw(x):
        if x == "bad":
            raise ValueError("cannot draw this")
        return x

    pipe = FramePipe(draw)
    try:
        pipe.submit("bad")
        with pytest.raises(ValueError, match="cannot draw this"):
            pipe.wait(1.0)
        assert pipe.wait(0.01) is None
        pipe.submit("good")
        assert pipe.wait(1.0) == "good"
    finally:
        pipe.close()


def test_close_stops_the_thread_and_later_calls_do_nothing_bad():
    pipe = FramePipe(lambda x: x)
    pipe.submit(1)
    assert pipe.wait(1.0) == 1
    pipe.close()
    assert not any(t.name == "frames" and t.is_alive() for t in threading.enumerate())
    pipe.submit(2)                                                 # after close: ignored, not a crash
    assert pipe.wait(0.01) is None
    pipe.close()
