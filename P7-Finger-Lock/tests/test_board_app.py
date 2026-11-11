"""app/python/main.py with the Arduino modules and the camera stubbed: a frame goes in, the MCU gets a picture."""
import sys
import types

import numpy as np
import pytest

import synth


class FakeBridge:
    calls, provided = [], {}

    @classmethod
    def call(cls, name, *args, **kw):
        cls.calls.append((name, args))

    @classmethod
    def provide(cls, name, fn):
        cls.provided[name] = fn


@pytest.fixture
def main(monkeypatch, tmp_path):
    app_utils = types.ModuleType("arduino.app_utils")
    app_utils.App, app_utils.Bridge = types.SimpleNamespace(run=lambda user_loop=None: None), FakeBridge
    cam_mod = types.ModuleType("arduino.app_peripherals.camera")
    cam_mod.Camera = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no camera in the test"))
    for name, mod in (("arduino", types.ModuleType("arduino")), ("arduino.app_utils", app_utils),
                      ("arduino.app_peripherals", types.ModuleType("arduino.app_peripherals")),
                      ("arduino.app_peripherals.camera", cam_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "web", types.SimpleNamespace(serve=lambda safe, port=0: None))
    sys.modules.pop("main", None)
    import main as m
    return m


def test_frame_drives_policy_and_mcu(main):
    FakeBridge.calls.clear()
    assert "button" in FakeBridge.provided
    safe = main.safe
    safe.tick()
    assert any(n == "draw" and len(a[0]) == 104 for n, a in FakeBridge.calls)  # the idle picture went to the matrix
    assert FakeBridge.provided["button"](1) is True and safe.lock.state == "LISTENING"
    safe.tick()
    assert ("leds", (3,)) in FakeBridge.calls  # both LEDs on while listening
    assert ("screen", (1, b"")) in FakeBridge.calls  # the TFT shows LISTENING with no digits yet
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:128, :128] = synth.hand(2, np.random.default_rng(0))
    safe.on_frame(frame)
    s = safe.snapshot()
    assert len(s["probs"]) == 6 and s["frames"] == 1 and s["engine"] == "numpy" and "pos" in s
    assert safe.motor("move", 50) == {"pos": None} and ("move", (50,)) in FakeBridge.calls  # the page's motor panel
    assert safe.stats()["frames"] == 1


def test_loop_without_camera_keeps_ticking(main):
    main.loop()  # tries to open the camera (fails in the test), still ticks the policy
    assert main.camera is None and main.safe.camera_ok is False
