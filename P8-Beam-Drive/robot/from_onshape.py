"""Turn the Onshape export (onshape/robot.xml) into the robot the simulation uses (robot/beam_bot.xml).

    python robot/from_onshape.py

Run it again after every new export. The shapes and the axle positions come from the CAD. What the CAD does
not know comes from robot/params.json: weights, friction and the motors.

What it changes:
- keeps the part that carries the two wheel joints and names it "chassis"; loose extra parts are dropped
- turns the robot the right way up if the CAD has it upside down (params "onshape_upside_down")
- moves the chassis frame to the middle of the body: x forward, y left, z up
- makes a positive command on either beam drive the robot forward
- sets the weights and friction, and swaps the export's position motors for speed motors
"""
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent / "onshape" / "robot.xml"
JOINTS = ("left_wheel", "right_wheel")
REQUIRED = [
    "total_mass", "beam_mass", "beam_friction", "body_friction",
    "max_speed_rad_s", "motor_kv", "motor_torque_limit", "loop_hz",
]
FLIP = np.array([0.0, 1.0, 0.0, 0.0])  # half a turn about x


def load_params(path=HERE / "params.json"):
    p = json.loads(Path(path).read_text())
    for key in REQUIRED:
        if not isinstance(p.get(key), (int, float)) or p[key] <= 0:
            raise ValueError(f"params.json: '{key}' must be a positive number, got {p.get(key)!r}")
    if 2 * p["beam_mass"] >= p["total_mass"]:
        raise ValueError("params.json: the two beams weigh more than total_mass")
    return p


def _rotate(quat, vec):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(vec, dtype=float), quat)
    return out


def _times(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, np.asarray(b, dtype=float))
    return out


def _mesh_points(model, data, geom_id):
    """Every corner of a mesh geom, in world coordinates."""
    mesh = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh]
    verts = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    return data.geom_xpos[geom_id] + verts @ data.geom_xmat[geom_id].reshape(3, 3).T


def _box(model, data, body_id):
    """Lowest and highest corner of one body's collision meshes."""
    pts = np.vstack([_mesh_points(model, data, g) for g in range(model.ngeom)
                     if model.geom_bodyid[g] == body_id and model.geom_contype[g]])
    return pts.min(0), pts.max(0)


def build_spec(p, export=EXPORT):
    """The finished robot as an MjSpec, plus the sizes measured from the CAD (metres)."""
    spec = mujoco.MjSpec.from_file(str(export))
    roots = [b for b in spec.worldbody.bodies if any(j.name in JOINTS for c in b.bodies for j in c.joints)]
    if len(roots) != 1:
        raise ValueError(f"{export}: need one part that carries both joints {JOINTS}, found {len(roots)}. "
                         "Name the revolute mates dof_left_wheel and dof_right_wheel.")
    chassis = roots[0]
    dropped = [b.name for b in spec.worldbody.bodies if b.name != chassis.name]
    for body in [b for b in spec.worldbody.bodies if b.name != chassis.name]:
        spec.delete(body)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    chassis.name = "chassis"
    links = {j.name: b for b in chassis.bodies for j in b.joints}
    if sorted(links) != sorted(JOINTS):
        raise ValueError(f"{export}: expected joints {JOINTS} on the chassis, found {sorted(links)}")

    # put the chassis frame at the middle of the body, the right way up
    turn = FLIP if p.get("onshape_upside_down") else np.array([1.0, 0.0, 0.0, 0.0])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = _box(model, data, model.body("chassis").id)
    centre = (low + high) / 2
    for item in list(chassis.geoms) + list(chassis.bodies):
        item.pos = _rotate(turn, np.asarray(item.pos) - centre)
        item.quat = _times(turn, item.quat)

    for name, link in links.items():
        joint = link.joints[0]
        if _rotate(np.asarray(link.quat, dtype=float), joint.axis)[1] < 0:   # make +command roll forward
            joint.axis = -np.asarray(joint.axis)
        joint.frictionloss, joint.armature, joint.damping = 0.0, 0.0002, [0.0005, 0, 0]
        actuator = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.set_to_velocity(kv=p["motor_kv"])
        actuator.forcelimited, actuator.forcerange = True, [-p["motor_torque_limit"], p["motor_torque_limit"]]

    # weights: the CAD parts have no material, so share out the weighed masses
    for body, mass, friction in [(chassis, p["total_mass"] - 2 * p["beam_mass"], p["body_friction"])] + [
            (link, p["beam_mass"], p["beam_friction"]) for link in links.values()]:
        body.explicitinertial = False
        solid = [g for g in body.geoms if g.contype]
        for geom in body.geoms:
            geom.density = 0.0
        for geom in solid:
            geom.mass = mass / len(solid)
            geom.friction, geom.priority = [friction, 0.005, 0.0001], 1

    # measure the finished robot, then start it just above the floor whatever angle the beams are at
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = _box(model, data, model.body("chassis").id)
    axle = data.xanchor[model.joint("left_wheel").id]
    beam_low, beam_high = _box(model, data, model.body(links["left_wheel"].name).id)
    reach = max(np.hypot(c[0] - axle[0], c[2] - axle[2]) for c in (beam_low, beam_high, [beam_low[0], 0, beam_high[2]], [beam_high[0], 0, beam_low[2]]))
    sizes = {
        "body_length": high[0] - low[0], "body_width": high[1] - low[1], "body_height": high[2] - low[2],
        "axle_from_rear": axle[0] - low[0], "axle_height": axle[2] - low[2],
        "beam_reach": reach, "dropped_parts": dropped,
    }
    chassis.pos = [0, 0, max(-low[2], reach - axle[2]) + 0.01]
    spec.meshdir = "../onshape/assets"
    return spec, sizes


if __name__ == "__main__":
    spec, sizes = build_spec(load_params())
    spec.compile()
    out = HERE / "beam_bot.xml"
    out.write_text(spec.to_xml())
    print(f"wrote {out}")
    for key, value in sizes.items():
        print(f"  {key}: {value if isinstance(value, list) else f'{value * 1000:.1f} mm'}")
