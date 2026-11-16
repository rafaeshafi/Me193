"""Drive both beams at one fixed speed (no policy) and film it. The simulated twin of the real open-loop test.

    python sim/open_loop.py            # 50 % speed for 5 s, writes open_loop.mp4
    python sim/open_loop.py 80 10      # 80 % for 10 s

Prints where the robot is each second: x forward, y sideways, heading. Works on the laptop and with runIt.py.
"""
import json
import math
import os
import sys
from pathlib import Path

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")  # the cluster has no screen

import mujoco

_HERE = Path(__file__).resolve().parent
ROBOT_DIR = next(p for p in (_HERE / "robot", _HERE.parent / "robot") if p.exists())
TIMESTEP = 0.0025


def make_model():
    """The robot on a floor, free to move. Returns (model, data) with the robot settled on the ground."""
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "beam_bot.xml"))
    spec.body("chassis").add_freejoint()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.35, 0.4, 0.45, 1])
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
    spec.option.timestep = TIMESTEP
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def pose(data):
    """x (m), y (m), heading (rad) of the chassis."""
    w, x, y, z = data.qpos[3:7]
    return data.qpos[0], data.qpos[1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(percent=50.0, seconds=5.0, settle=4.0, video=None):
    """Sit still for `settle` seconds, then drive both beams at `percent` of full speed. Returns the final pose."""
    params = json.loads((ROBOT_DIR / "params.json").read_text())
    speed = params["max_speed_rad_s"] * percent / 100
    model, data = make_model()
    renderer, frames, camera = None, [], None
    if video:
        renderer = mujoco.Renderer(model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = model.body("chassis").id
        camera.distance, camera.azimuth, camera.elevation = 0.6, 140, -25
    next_print = settle + 1.0
    while data.time < settle + seconds:
        data.ctrl[:] = speed if data.time >= settle else 0.0
        mujoco.mj_step(model, data)
        if renderer and data.time >= len(frames) / 30:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
        if data.time >= next_print:
            x, y, yaw = pose(data)
            print(f"t={data.time - settle:4.1f}s  x={x:+.3f} m  y={y:+.3f} m  heading={math.degrees(yaw):+6.1f} deg")
            next_print += 1.0
    if renderer:
        import imageio
        renderer.close()
        imageio.mimwrite(video, frames, fps=30)
        print(f"wrote {video}: {len(frames)} frames")
    return pose(data)


if __name__ == "__main__":
    percent = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    if not 0 < percent <= 100 or not 0 < seconds <= 60:
        sys.exit("usage: open_loop.py [percent 1-100] [seconds 1-60]")
    run(percent, seconds, video="open_loop.mp4")
