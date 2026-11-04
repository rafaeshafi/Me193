"""The camera hood for the finger lock, modelled in SolidCraft (open-source parametric CAD) from this script.

    enclosure/camera_box.py            # writes enclosure/camera_box.json and runs SolidCraft's CLI on it:
                                       #   out/camera_box.sc (open it in SolidCraft), .step, .stl, iso/top/front .png

A four-sided hood, open at the front and the bottom, that stands on a base plate. The roof is flat on top; a block
under it, inside the hood, gives it the depth for a cavity cut down from the roof surface, and the whole webcam
(70 x 18.5 x 25 mm body, lens on the 70 x 18.5 face) sits in that cavity lens-down, its back `sink` below the roof,
looking at the desk through a window in the cavity floor. The cavity is a press fit: its walls clear the camera by a
hair and ten thin crush ribs on them stand proud by rib_p, so the ribs squash on the way in and grip whatever the
printer did to the walls (too tight: shave the ribs or lower rib_p; loose: raise it). A chamfer at the mouth leads the
camera in, the ribs start a little below it, and a full-depth notch at each end is both the finger hold to pull the
camera out and the way out for its cable. Two slots beside the cavity take a strap over it as a belt-and-braces, and a
1/4-20 clearance hole in front is there for a camera with a tripod thread instead. The base plate carries the engraved
square the net is trained on, sized from the lens height and the camera's field of view, centred under the lens; a
notch at the top of the back wall passes the cable. Every number below is a design parameter in the .sc, so the hood
rebuilds in SolidCraft when one changes.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CLI = "/Applications/SolidCraft.app/Contents/MacOS/solidcraft-cli"

# ---- parameters (mm, degrees) ----
P = dict(
    cam_w=70.0, cam_h=18.5, cam_d=25.0,     # the webcam body: width, height of the lens face, depth (front to back)
    fit=0.3,                                # cavity wall clearance on width and height (total; the ribs do the gripping)
    rib_w=1.5, rib_p=0.4, rib_n=3,          # crush ribs: width, how far they stand into the cavity, count per long wall
    lead=1.0,                               # mouth chamfer; the ribs also stop this far below the mouth
    sink=0.5,                               # the camera's back sits this far below the roof surface
    notch_w=10.0, notch_d=6.0,              # finger/cable notch at each end of the cavity: along the wall, into the block
    lip=2.5,                                # cavity floor left under the camera on each long side (the lens looks past it)
    inner=320.0,                            # hood inside width and depth
    wall=4.0,                               # wall and roof thickness
    lens_h=306.0,                           # lens face above the base plate top
    boss_margin=10.0,                       # block wall around the cavity
    fov_h=70.0,                             # the webcam's horizontal field of view (degrees); sets the square
    roi=0.55,                               # the square's side as a fraction of the frame height (config.json "roi")
    line=2.0, engrave=0.6,                  # the engraved square's line width and depth
    strap_w=4.0, strap_l=14.0, strap_gap=6.0,  # strap slots beside the cavity
    tripod_d=6.6, tripod_y=45.0,            # 1/4-20 clearance hole, this far in front of the lens
    cable_w=16.0, cable_h=20.0,             # notch in the top of the back wall
    base_t=4.0, corner_r=8.0,
)


def derive(p):
    d = dict(p)
    d["outer"] = p["inner"] + 2 * p["wall"]
    d["pocket_w"], d["pocket_d"] = p["cam_w"] + p["fit"], p["cam_h"] + p["fit"]          # plan: X by Y
    d["window_w"], d["window_d"] = p["cam_w"] - 2 * 4.0, d["pocket_d"] - 2 * p["lip"]      # the lens looks through this
    d["floor_t"] = 2.0                                                                     # cavity floor thickness
    d["depth"] = p["cam_d"] + p["sink"]                                                    # cavity depth from the roof top
    d["hood_h"] = p["lens_h"] + d["depth"]                                                 # roof top; the lens face is depth below it
    d["boss_w"], d["boss_d"] = d["pocket_w"] + 2 * p["boss_margin"], d["pocket_d"] + 2 * p["boss_margin"]
    d["boss_h"] = d["depth"] + d["floor_t"]                                                # the block, roof top to floor underside
    frame_w = 2 * p["lens_h"] * math.tan(math.radians(p["fov_h"] / 2))                     # frame width on the desk
    d["square"] = round(p["roi"] * 480 / 640 * frame_w, 1)
    return d


def script(d):
    o, h, w, t = d["outer"], d["hood_h"], d["wall"], d["base_t"]
    half = o / 2
    c = []

    def add(command, **params):
        c.append({"command": command, "params": params})

    for name in ("inner", "wall", "lens_h", "cam_w", "cam_h", "cam_d", "fit", "rib_w", "rib_p", "lead", "lip", "boss_margin", "square", "line",
                 "engrave", "strap_w", "strap_l", "strap_gap", "tripod_d", "tripod_y", "cable_w", "cable_h", "base_t", "corner_r",
                 "notch_w", "sink"):
        add("parameters.change", name=name, expression=f"{d[name]} mm")
    # the hood: a box, rounded vertical edges, then hollowed from the front (+Y) and the bottom
    add("solid.box", length="inner + 2*wall", width="inner + 2*wall", height=h, center=[0, 0, h / 2], body_name="Hood")
    add("solid.fillet", edges=[[sx * half, sy * half, h / 2] for sx in (1, -1) for sy in (1, -1)], radius="corner_r")
    add("solid.shell", faces=[[0, half, h / 2], [0, 0, 0]], thickness="wall", direction="inside")
    # the block under the roof, the cavity cut down into it from the roof top, and the press fit: a chamfer round the
    # mouth, crush ribs on all four walls (rooted 0.5 mm into the wall, rising from the floor to lead below the mouth),
    # and a full-depth notch at each end for a finger and the cable
    top, pw, pd = h, d["pocket_w"], d["pocket_d"]
    add("sketch.create", plane="XY", offset=h - w, name="Block")  # the roof underside
    add("sketch.rectangle.center", center=[0, 0], corner=[d["boss_w"] / 2, d["boss_d"] / 2])
    add("sketch.finish")
    add("solid.extrude", distance=d["boss_h"] - w, direction="negative", operation="join", targets=["Hood"])
    add("sketch.create", plane="XY", offset=h, name="Cavity")
    add("sketch.rectangle.center", center=[0, 0], corner=["(cam_w + fit)/2", "(cam_h + fit)/2"])
    add("sketch.finish")
    add("solid.extrude", distance="cam_d + sink", direction="negative", operation="cut", targets=["Hood"])
    add("solid.chamfer", edges=[[0, pd / 2, top], [0, -pd / 2, top], [pw / 2, 0, top], [-pw / 2, 0, top]], distance="lead",
        body="Hood")
    rib_h = d["depth"] - d["lead"] + 0.5
    rib_z = d["lens_h"] - 0.5 + rib_h / 2
    for i in range(d["rib_n"]):
        x = (i - (d["rib_n"] - 1) / 2) * pw * 0.68 / max(d["rib_n"] - 1, 1)
        for sy in (1, -1):
            add("solid.box", length="rib_w", width="rib_p + 0.5", height=rib_h,
                center=[x, sy * (pd / 2 + 0.25 - d["rib_p"] / 2), rib_z], operation="join", targets=["Hood"])
    for sx in (1, -1):  # two per end wall, clear of the notch between them
        for sy in (1, -1):
            add("solid.box", length="rib_p + 0.5", width="rib_w", height=rib_h,
                center=[sx * (pw / 2 + 0.25 - d["rib_p"] / 2), sy * (pd / 2 - d["rib_w"] / 2 - 1.5), rib_z],
                operation="join", targets=["Hood"])
    for sx in (1, -1):
        add("solid.box", length=d["notch_d"] + 0.5, width="notch_w", height=d["depth"] + 0.5,
            center=[sx * (pw / 2 + d["notch_d"] / 2 - 0.25), 0, top - d["depth"] / 2 + 0.25], operation="cut", targets=["Hood"])
    add("sketch.create", plane="XY", offset="lens_h", name="Window")  # the cavity floor
    add("sketch.rectangle.center", center=[0, 0], corner=[d["window_w"] / 2, "(cam_h + fit)/2 - lip"])
    add("sketch.finish")
    add("solid.extrude", distance=d["floor_t"] + 1, direction="negative", operation="cut", targets=["Hood"])
    # strap slots either side of the cavity, the tripod hole in front of it, the cable notch at the back
    add("sketch.create", plane="XY", offset=h, name="Slots")
    for sx in (1, -1):
        x = sx * (d["boss_w"] / 2 + d["strap_gap"] + d["strap_w"] / 2)
        add("sketch.rectangle.center", center=[x, 0], corner=[x + d["strap_w"] / 2, d["strap_l"] / 2])
    add("sketch.circle.center", center=[0, "tripod_y"], diameter="tripod_d")
    add("sketch.rectangle.two_point", p0=["-cable_w/2", -half - 1], p1=["cable_w/2", -half + w + 1])
    add("sketch.finish")
    add("solid.extrude", distance="cable_h", direction="negative", operation="cut", targets=["Hood"])
    # the base plate with the engraved square, centred under the lens
    add("solid.box", length="inner + 2*wall", width="inner + 2*wall", height="base_t", center=[0, 0, -t / 2],
        operation="new", body_name="Base")
    add("solid.fillet", edges=[[sx * half, sy * half, -t / 2] for sx in (1, -1) for sy in (1, -1)], radius="corner_r",
        body="Base")
    add("sketch.create", plane="XY", offset=0, name="Square")
    add("sketch.rectangle.center", center=[0, 0], corner=["square/2 + line/2", "square/2 + line/2"])
    add("sketch.rectangle.center", center=[0, 0], corner=["square/2 - line/2", "square/2 - line/2"])
    add("sketch.finish")
    add("solid.extrude", distance="engrave", direction="negative", operation="cut", targets=["Base"])
    return {"commands": c}


def main():
    d = derive(P)
    OUT.mkdir(exist_ok=True)
    path = HERE / "camera_box.json"
    path.write_text(json.dumps(script(d), indent=1))
    print(f"hood {d['outer']:.0f} x {d['outer']:.0f} x {d['hood_h']:.0f} mm; cavity {d['pocket_w']} x {d['pocket_d']} x "
          f"{d['depth']} (ribs {d['rib_n'] * 2 + 4} x {d['rib_p']} mm proud -> grip {d['cam_w'] - (d['pocket_w'] - 2 * d['rib_p']):.2f} / "
          f"{d['cam_h'] - (d['pocket_d'] - 2 * d['rib_p']):.2f} mm interference); window {d['window_w']} x {d['window_d']}; "
          f"square {d['square']} mm; script {path.name}")
    if "--no-run" in sys.argv:
        return
    run = [CLI, "run", str(path), "--save", str(OUT / "camera_box.sc"), "--out", str(OUT / "camera_box.step"),
           "--out", str(OUT / "camera_box.stl"), "--param-check", "warn"]
    subprocess.run(run, check=True)
    for view in ("iso", "front", "top"):
        subprocess.run([CLI, "snapshot", "--in", str(OUT / "camera_box.sc"), "--out", str(OUT / f"camera_box_{view}.png"),
                        "--view", view, "--width", "1200", "--height", "900", "--no-grid"], check=True)
    subprocess.run([CLI, "eval", "--in", str(OUT / "camera_box.sc"), "--json-out", str(OUT / "camera_box_measure.json")], check=True)
    print("written:", ", ".join(p.name for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    main()
