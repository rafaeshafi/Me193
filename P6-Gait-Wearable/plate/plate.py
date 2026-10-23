"""Laser-cut 3 mm acrylic mounting plate for the P6 gait wearable (UNO Q + MPU6050) on the thigh.

Front view, hip at the top (+Y), origin at the plate centre. The UNO Q sits at the top turned 90 degrees so its USB-C
faces the bottom (knee) edge; the MPU6050 sits right of it near its bottom, long side along the thigh. The upper strap
crosses the UNO Q's hole-free middle, the lower strap the empty band below it. Boards stand on M3 standoffs (UNO Q
10 mm, MPU 5 mm with its pins pointing up). Two 25.4 mm elastic straps run across the FRONT face under the boards: in
through a left slit, out through the right slit, buckle behind the thigh. check() keeps every standoff out of both
strap bands. Values marked VERIFY want a caliper check before cutting.

Run:  ~/whoop-hr/.venv-cad/bin/python plate/plate.py   -> plate/out/plate.dxf (laser) and plate/out/plate.svg (preview)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import ezdxf

OUT = Path(__file__).resolve().parent / "out"


@dataclass(frozen=True)
class P:
    width: float = 100.0
    height: float = 109.0
    thickness: float = 3.0
    corner_r: float = 6.0
    margin: float = 3.5            # board edge to plate edge
    min_web: float = 2.0           # acrylic left between a hole and the plate edge
    standoff_r: float = 2.75       # M3 standoff, 5.5 mm across flats: straps stay this far from every hole centre
    # UNO Q (UNO R3 outline) turned 90 degrees: USB-C on the bottom edge. Holes from the board's lower-left corner.
    uno_w: float = 53.34           # across the thigh
    uno_h: float = 68.58           # along the thigh
    uno_holes: tuple = ((50.80, 13.97), (2.54, 15.24), (45.72, 66.04), (17.78, 66.04))
    uno_hole_d: float = 3.2
    uno_strap_y: float = 40.6      # lower strap band centre above the UNO Q's bottom edge: between its hole rows
    # MPU6050 GY-521 breakout, long (header) side along the thigh so the printed X arrow runs down the thigh (VERIFY the
    # arrow on your board); its two holes sit at the corners of the long edge opposite the header (VERIFY both numbers)
    mpu_w: float = 16.0            # across the thigh
    mpu_h: float = 21.0            # along the thigh
    mpu_hole_dx: float = 15.5      # hole spacing along the long side
    mpu_hole_off: float = 6.0      # holes this far left of the board centreline (header on the right, pins up)
    mpu_hole_d: float = 3.2
    mpu_y: float = 14.0            # MPU centre above the UNO Q's bottom edge
    # straps: VELCRO 90441, 25.4 mm wide, ~1.5 mm thick
    strap_w: float = 25.4
    slit_len: float = 27.0
    slit_w: float = 4.0
    slit_inset: float = 3.0        # plate edge to slit


def layout(p: P = P()) -> dict:
    """Centres of everything, in plate coordinates (mm). Also used by the checks."""
    uno_cx = -p.width / 2 + p.margin + p.uno_w / 2
    uno_top = p.height / 2 - p.margin
    uno_ll = (uno_cx - p.uno_w / 2, uno_top - p.uno_h)
    uno_cy = uno_top - p.uno_h / 2
    mpu_cx = uno_ll[0] + p.uno_w + 2.0 + p.mpu_w / 2   # 2 mm right of the UNO Q
    mpu_cy = uno_ll[1] + p.mpu_y
    strap_hi_y = uno_ll[1] + p.uno_strap_y
    strap_lo_y = uno_ll[1] - 4.0 - p.strap_w / 2
    slit_x = p.width / 2 - p.slit_inset - p.slit_w / 2
    return {
        "uno": (uno_cx, uno_cy),
        "uno_holes": [(uno_ll[0] + hx, uno_ll[1] + hy) for hx, hy in p.uno_holes],
        "mpu": (mpu_cx, mpu_cy),
        "mpu_holes": [(mpu_cx - p.mpu_hole_off, mpu_cy - p.mpu_hole_dx / 2),
                      (mpu_cx - p.mpu_hole_off, mpu_cy + p.mpu_hole_dx / 2)],
        "straps": [strap_hi_y, strap_lo_y],
        "slits": [(sx * slit_x, y) for sx in (-1, 1) for y in (strap_hi_y, strap_lo_y)],
    }


def check(p: P, lay: dict) -> list[str]:
    """Plain-English clearance report; a non-empty list means do not cut."""
    problems = []
    half_w, half_h = p.width / 2, p.height / 2
    holes = [(name, x, y, d) for name, d in (("uno", p.uno_hole_d), ("mpu", p.mpu_hole_d))
             for x, y in lay[name + "_holes"]]
    # boards inside the plate
    for name, (cx, cy), (w, h) in (("UNO Q", lay["uno"], (p.uno_w, p.uno_h)),
                                  ("MPU6050", lay["mpu"], (p.mpu_w, p.mpu_h))):
        if abs(cx) + w / 2 > half_w - p.margin + 0.01 or abs(cy) + h / 2 > half_h - p.margin + 0.01:
            problems.append(f"{name} runs off the plate")
    ux, uy = lay["uno"]
    mx, my = lay["mpu"]
    if mx - p.mpu_w / 2 < ux + p.uno_w / 2 and abs(my - uy) < (p.uno_h + p.mpu_h) / 2:
        problems.append("MPU6050 overlaps the UNO Q")
    # holes clear of the plate edge
    for name, x, y, d in holes:
        if abs(x) > half_w - d / 2 - p.min_web or abs(y) > half_h - d / 2 - p.min_web:
            problems.append(f"{name} hole at ({x:.1f}, {y:.1f}) is too close to the edge")
    # slits clear of the edge, and no hole or the MPU footprint in a slit
    for x, y in lay["slits"]:
        if abs(x) + p.slit_w / 2 > half_w - 1.5:
            problems.append(f"slit at x={x:.1f} breaks the edge")

    def hits_slit(x, y, half_w_, half_h_):
        return any(abs(x - sx) < half_w_ + p.slit_w / 2 and abs(y - sy) < half_h_ + p.slit_len / 2
                   for sx, sy in lay["slits"])

    for name, x, y, d in holes:
        if hits_slit(x, y, d / 2 + 1.0, d / 2 + 1.0):
            problems.append(f"{name} hole at ({x:.1f}, {y:.1f}) runs into a strap slit")
    if hits_slit(mx, my, p.mpu_w / 2, p.mpu_h / 2):
        problems.append("MPU6050 footprint sits over a strap slit")
    # no standoff inside either strap band (the straps run flat under the boards)
    for band_y in lay["straps"]:
        for name, x, y, d in holes:
            if abs(y - band_y) < p.strap_w / 2 + p.standoff_r:
                problems.append(f"{name} standoff at ({x:.1f}, {y:.1f}) sits in the strap band at y={band_y:.1f}")
    return problems


def plate(p: P = P()) -> cq.Workplane:
    lay = layout(p)
    w = cq.Workplane("XY").rect(p.width, p.height).extrude(p.thickness).edges("|Z").fillet(p.corner_r)
    w = w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["uno_holes"]).hole(p.uno_hole_d)
    w = w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["mpu_holes"]).hole(p.mpu_hole_d)
    w = (w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["slits"])
         .slot2D(p.slit_len, p.slit_w, angle=90).cutThruAll())
    return w


def export(body: cq.Workplane, p: P) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    dxf = OUT / "plate.dxf"
    # section() cuts relative to the chain's current workplane (the top face), so start from a fresh XY plane
    cq.exporters.export(cq.Workplane("XY").add(body.val()).section(p.thickness / 2), str(dxf))
    entities = len(list(ezdxf.readfile(str(dxf)).modelspace()))
    if entities < 10:
        raise RuntimeError(f"{dxf} has only {entities} entities: the section missed the solid")
    svg = OUT / "plate.svg"
    cq.exporters.export(body, str(svg), opt={"projectionDir": (0, 0, 1), "width": 800, "height": 1100, "showAxes": False})
    return dxf, svg


def main() -> int:
    p = P()
    lay = layout(p)
    problems = check(p, lay)
    for line in problems:
        print("PROBLEM:", line)
    dxf, svg = export(plate(p), p)
    print(f"plate {p.width} x {p.height} x {p.thickness} mm; "
          f"UNO Q centre ({lay['uno'][0]:.1f}, {lay['uno'][1]:.1f}) USB-C down, MPU centre ({lay['mpu'][0]:.1f}, "
          f"{lay['mpu'][1]:.1f}), strap bands at y = {lay['straps'][0]:.1f} and {lay['straps'][1]:.1f}")
    print("wrote", dxf, "and", svg)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
