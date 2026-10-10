"""The scoreboard app that runs on the UNO Q (uno_q/pingpong-board): it listens for the game, and draws what it is sent on the matrix."""

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "uno_q" / "pingpong-board"
sys.path.insert(0, str(APP / "python"))

import scoreboard  # noqa: E402

from pingpong import boardlink, matrix  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


def board(clock=None, fail=None):
    drawn = []

    def draw(frame):
        if fail and fail[0]:
            raise RuntimeError("the MCU did not answer")
        drawn.append(frame)

    return scoreboard.Scoreboard(draw, now=clock or FakeClock()), drawn


FRAME = "0" * 40 + "7" * 4 + "0" * 60


def line(frame):
    return b"F" + frame.encode() + b"\n"


# --- the talk -----------------------------------------------------------------------------------------------------------------------------
def test_a_frame_is_a_line_of_f_and_a_hundred_and_four_digits():
    assert scoreboard.parse(line(FRAME)) == FRAME
    assert scoreboard.parse(line(FRAME)[:-1] + b"\r\n") == FRAME


@pytest.mark.parametrize("garbage", [b"", b"\n", b"hello\n", b"F" + b"0" * 103 + b"\n", b"F" + b"0" * 105 + b"\n", b"F" + b"8" * 104 + b"\n",
                                     b"G" + b"0" * 104 + b"\n", b"F" + b"x" * 104 + b"\n"])
def test_anything_else_is_ignored(garbage):
    assert scoreboard.parse(garbage) is None
    b, drawn = board()
    b.hear(garbage)
    b.tick()
    assert drawn == [scoreboard.STANDBY]                                            # still nothing but the standby dots


def test_what_the_mcu_is_sent_is_a_hundred_and_four_bytes_one_level_each():
    levels = scoreboard.levels(FRAME)
    assert isinstance(levels, bytes) and len(levels) == scoreboard.PIXELS
    assert list(levels[38:46]) == [0, 0, 7, 7, 7, 7, 0, 0] and max(scoreboard.levels(scoreboard.STANDBY)) == 2


def test_the_apps_greeting_and_port_are_the_ones_the_game_expects():
    assert scoreboard.HELLO == boardlink.HELLO + b"\n" and scoreboard.PORT == boardlink.BOARD_PORT
    assert scoreboard.STANDBY == boardlink.message(matrix.frame(("standby",), 0.0))[1:-1].decode()
    assert len(scoreboard.STANDBY) == scoreboard.PIXELS == matrix.ROWS * matrix.COLS


# --- what it draws --------------------------------------------------------------------------------------------------------------------------
def test_the_standby_dots_are_drawn_until_a_game_is_heard_and_then_the_game():
    b, drawn = board()
    b.tick()
    assert drawn == [scoreboard.STANDBY]
    b.hear(line(FRAME))
    b.tick()
    assert drawn == [scoreboard.STANDBY, FRAME]


def test_a_picture_is_drawn_once_not_every_turn():
    b, drawn = board()
    b.hear(line(FRAME))
    for _ in range(5):
        b.tick()
    assert drawn == [FRAME]


def test_after_eight_seconds_of_silence_the_standby_dots_come_back():
    clock = FakeClock()
    b, drawn = board(clock)
    b.hear(line(FRAME))
    b.tick()
    clock.t += scoreboard.STANDBY_AFTER_S - 0.5
    b.tick()
    assert drawn == [FRAME]
    clock.t += 1.0
    b.tick()
    assert drawn == [FRAME, scoreboard.STANDBY]
    b.hear(line(FRAME))                                                              # the game is back
    b.tick()
    assert drawn[-1] == FRAME


def test_a_draw_that_fails_is_tried_again_a_moment_later():
    clock = FakeClock()
    fail = [True]
    b, drawn = board(clock, fail)
    b.hear(line(FRAME))
    b.tick()
    b.tick()
    assert drawn == []
    fail[0] = False
    b.tick()
    assert drawn == []                                                               # not before the pause
    clock.t += scoreboard.RETRY_S + 0.1
    b.tick()
    assert drawn == [FRAME]


def test_the_newest_frame_wins():
    b, drawn = board()
    b.hear(line(FRAME))
    other = "7" + "0" * 103
    b.hear(line(other))
    b.tick()
    assert drawn == [other]


# --- the server -----------------------------------------------------------------------------------------------------------------------------
def test_a_game_that_connects_is_greeted_and_what_it_sends_is_drawn():
    b, drawn = board(clock=time.monotonic)
    server = scoreboard.serve(b, port=0, host="127.0.0.1")
    try:
        with socket.create_connection(server.server_address, timeout=2) as sock:
            reader = sock.makefile("rb")
            assert reader.readline() == scoreboard.HELLO
            sock.sendall(line(FRAME))
            deadline = time.time() + 2
            while time.time() < deadline:
                b.tick()
                if FRAME in drawn:
                    break
                time.sleep(0.01)
            assert FRAME in drawn
    finally:
        server.shutdown()
        server.server_close()


# --- the app's files ------------------------------------------------------------------------------------------------------------------------
def test_the_app_says_what_it_is_and_publishes_its_port():
    yaml = (APP / "app.yaml").read_text()
    assert "name:" in yaml and f"ports: [{scoreboard.PORT}]" in yaml and "bricks: []" in yaml


def test_the_sketch_provides_the_draw_the_python_side_calls_in_the_way_arduinos_own_matrix_example_does():
    sketch = (APP / "sketch" / "sketch.ino").read_text()
    main = (APP / "python" / "main.py").read_text()
    assert 'Bridge.provide("draw", draw)' in sketch and "std::vector<uint8_t>" in sketch and 'Bridge.call("draw"' in main
    assert "setGrayscaleBits(3)" in sketch and "normalizeFrame" in sketch
    assert (APP / "sketch" / "sketch.yaml").read_text().count("arduino:zephyr:unoq") == 1


@pytest.mark.skipif(shutil.which("clang++") is None, reason="no C++ compiler on this machine")
def test_the_c_plus_plus_codec_keeps_the_levels_and_makes_anything_else_safe_to_draw(tmp_path):
    driver = tmp_path / "t.cpp"
    driver.write_text('''
#include <cassert>
#include <cstring>
#include <vector>
#include "frame_codec.h"
int main() {
  uint8_t levels[pingpong::kPixels];
  std::memset(levels, 99, sizeof levels);
  std::vector<uint8_t> frame(104, 5);
  for (int i = 0; i < 8; ++i) frame[i] = i;
  frame[8] = 200;                                             // too bright to mean anything: the brightest there is
  pingpong::normalizeFrame(frame.data(), frame.size(), levels);
  for (int i = 0; i < 8; ++i) assert(levels[i] == i);
  assert(levels[8] == 7 && levels[9] == 5 && levels[103] == 5);
  std::vector<uint8_t> shorter = {1, 2, 3};                   // too short: dark where it says nothing, and nothing read beyond it
  pingpong::normalizeFrame(shorter.data(), shorter.size(), levels);
  assert(levels[0] == 1 && levels[2] == 3 && levels[3] == 0 && levels[103] == 0);
  pingpong::normalizeFrame(nullptr, 0, levels);
  assert(levels[0] == 0);
  return 0;
}
''')
    exe = tmp_path / "t"
    build = subprocess.run(["clang++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{APP / 'sketch' / 'src'}", str(driver),
                            str(APP / "sketch" / "src" / "frame_codec.cpp"), "-o", str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    assert subprocess.run([str(exe)], capture_output=True).returncode == 0
