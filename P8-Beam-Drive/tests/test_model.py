"""Checks on the robot model. Run from P8-Beam-Drive:  .venv/bin/python -m pytest tests -v"""
import json
import math
import sys
from pathlib import Path

import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "robot"), str(ROOT / "sim")]

import from_onshape  # noqa: E402
import open_loop  # noqa: E402

PARAMS = json.loads((ROOT / "robot" / "params.json").read_text())


def tilt_deg(data):
    """Angle between the robot's up axis and the world's, in degrees."""
    w, x, y, z = data.qpos[3:7]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def test_xml_file_matches_params():
    spec, _ = from_onshape.build_spec(from_onshape.load_params())
    spec.compile()
    assert (ROOT / "robot" / "beam_bot.xml").read_text() == spec.to_xml()


def test_cad_is_the_real_robot_size():
    _, sizes = from_onshape.build_spec(from_onshape.load_params())
    assert sizes["body_length"] == pytest.approx(0.080, abs=0.002)   # 10 studs
    assert sizes["axle_from_rear"] == pytest.approx(0.016, abs=0.002)  # 2 studs
    assert sizes["beam_reach"] > sizes["axle_height"]                # or the beams never reach the floor


def test_names_the_task_relies_on():
    model, _ = open_loop.make_model()
    for joint in ("left_wheel", "right_wheel"):
        assert model.joint(joint).type == mujoco.mjtJoint.mjJNT_HINGE
        assert model.actuator(joint).trnid[0] == model.joint(joint).id
    assert model.body("chassis").id > 0


def test_total_mass():
    model, _ = open_loop.make_model()
    assert model.body_subtreemass[model.body("chassis").id] == pytest.approx(PARAMS["total_mass"], rel=0.01)


def test_rests_still():
    model, data = open_loop.make_model()
    while data.time < 4.0:   # dropped from 1 cm, it rocks for about 3 s first
        mujoco.mj_step(model, data)
    x1, y1, _ = open_loop.pose(data)
    while data.time < 6.0:
        mujoco.mj_step(model, data)
    x2, y2, _ = open_loop.pose(data)
    assert math.hypot(x2 - x1, y2 - y1) < 0.002
    assert tilt_deg(data) < 20


def test_open_loop_goes_forward_and_fairly_straight():
    x, y, yaw = open_loop.run(percent=50, seconds=5)
    assert x > 0.10
    assert abs(y) < 0.3 * x
    assert abs(math.degrees(yaw)) < 30


def test_beams_heavier_than_the_robot_are_rejected(tmp_path):
    bad = dict(PARAMS, beam_mass=PARAMS["total_mass"])
    path = tmp_path / "params.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="beam"):
        from_onshape.load_params(path)
