"""tools/reset_hub.py: shake a ghost connection loose (a killed run leaves the hub 'connected' for ~24 s)."""

import pytest

from pingpong.sources_fake import FakeEnv
from tools import reset_hub


def test_a_hub_that_is_found_is_stopped_beeped_and_released():
    env = FakeEnv()
    out = []
    assert reset_hub.run(env, card={}, attempts=3, wait_s=0.0, out=out.append) == 0
    names = [c[0] for c in env.hub_device.calls]
    assert "beep" in names and names[-2:] == ["motor_stop", "disconnect"]


def test_a_hub_hidden_by_a_ghost_connection_is_found_once_the_ghost_times_out():
    env = FakeEnv(hub_found=False)
    attempts = {"n": 0}
    original = env.sleep

    def sleep(seconds):
        original(seconds)
        attempts["n"] += 1
        if attempts["n"] >= 2:
            env.hub_device.fail_connect = False              # the ghost session timed out

    env.sleep = sleep
    out = []
    assert reset_hub.run(env, card={}, attempts=5, wait_s=5.0, out=out.append) == 0
    assert sum("not found" in line for line in out) == 2


def test_a_hub_that_never_appears_gives_the_checklist():
    env = FakeEnv(hub_found=False)
    out = []
    assert reset_hub.run(env, card={}, attempts=3, wait_s=0.0, out=out.append) == 1
    text = " ".join(out).lower()
    assert "wake" in text and "card" in text and "scan_hubs" in text


def test_a_missing_card_is_reported_before_any_scanning(capsys):
    assert reset_hub.main([]) == 2
    assert "scan_hubs" in capsys.readouterr().err


def test_the_selftest_is_green():
    assert reset_hub.main(["--selftest"]) == 0
