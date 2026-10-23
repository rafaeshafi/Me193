"""Laser-cut 3 mm acrylic mounting plate for the P6 gait wearable (UNO Q + 3.2" ILI9341 TFT + MPU6050) on the thigh.

Front view, hip at the top (+Y), origin at the plate centre. The TFT spans the top, the UNO Q sits bottom-left and the
MPU6050 just right of it with its long (header) side along the thigh so its X axis runs down the thigh. Boards stand on M3 standoffs (TFT 20 mm so its rear header housings
clear, UNO Q 10 mm, MPU 5 mm). Two 25 mm elastic straps run across the FRONT face under the boards: in through a left
slit, out through a right slit, buckle behind the thigh. Values marked VERIFY want a caliper check before cutting.

Run:  ~/whoop-hr/.venv-cad/bin/python plate/plate.py   -> plate/out/plate.dxf (laser) and plate/out/plate.svg (preview)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).resolve().parent / "out"


@dataclass(frozen=True)
class P:
    width: float = 100.0
    height: float = 130.0
    thickness: float = 3.0
    corner_r: float = 6.0
    margin: float = 3.5            # board edge to plate edge
    # 3.2" TFT module (measured for whoop-hr)
    tft_w: float = 93.0
    tft_h: float = 60.0
    tft_hole_dx: float = 87.4
    tft_hole_dy: float = 54.4
    tft_hole_d: float = 3.2
    # UNO form factor (UNO Q), holes from the board's lower-left corner
    uno_w: float = 68.58
    uno_h: float = 53.34
    uno_holes: tuple = ((13.97, 2.54), (15.24, 50.80), (66.04, 7.62), (66.04, 35.56))
    uno_hole_d: float = 3.2
    gap: float = 8.0               # TFT bottom edge to UNO Q top edge
    # MPU6050 GY-521 breakout mounted with its long (header) side along the thigh, so the printed X arrow runs down the
    # thigh (VERIFY the arrow on your board); its two holes sit along that long side (VERIFY spacing)
    mpu_w: float = 16.0            # across the thigh
    mpu_h: float = 21.0            # along the thigh
    mpu_hole_dx: float = 15.5
    mpu_hole_d: float = 3.2
    # strap slits for 25.4 mm wide, ~1.5 mm thick elastic (VELCRO 90441)
    slit_len: float = 27.0
    slit_w: float = 4.0
    slit_inset: float = 3.0        # plate edge to slit


def layout(p: P = P()) -> dict:
    """Centres of everything, in plate coordinates (mm). Also used by the checks."""
    tft_cy = p.height / 2 - p.margin - p.tft_h / 2
    uno_cx = -p.width / 2 + p.margin + p.uno_w / 2
    uno_cy = tft_cy - p.tft_h / 2 - p.gap - p.uno_h / 2
    mpu_cx = uno_cx + p.uno_w / 2 + 2.0 + p.mpu_w / 2   # 2 mm clear of the UNO Q, clear of the right slit
    uno_ll = (uno_cx - p.uno_w / 2, uno_cy - p.uno_h / 2)
    slit_x = p.width / 2 - p.slit_inset - p.slit_w / 2
    return {
        "tft": (0.0, tft_cy),
        "tft_holes": [(sx * p.tft_hole_dx / 2, tft_cy + sy * p.tft_hole_dy / 2) for sx in (-1, 1) for sy in (-1, 1)],
        "uno": (uno_cx, uno_cy),
        "uno_holes": [(uno_ll[0] + hx, uno_ll[1] + hy) for hx, hy in p.uno_holes],
        "mpu": (mpu_cx, uno_cy),
        "mpu_holes": [(mpu_cx, uno_cy - p.mpu_hole_dx / 2), (mpu_cx, uno_cy + p.mpu_hole_dx / 2)],
        "slits": [(sx * slit_x, y) for sx in (-1, 1) for y in (tft_cy, uno_cy)],
    }


def check(p: P, lay: dict) -> list[str]:
    """Plain-English clearance report; a non-empty list means do not cut."""
    problems = []
    half_w, half_h = p.width / 2, p.height / 2
    uno_cx, uno_cy = lay["uno"]
    mpu_cx, _ = lay["mpu"]
    if uno_cy - p.uno_h / 2 < -half_h + p.margin:
        problems.append("UNO Q runs off the bottom of the plate: increase height")
    if uno_cx + p.uno_w / 2 + 2 > mpu_cx - p.mpu_w / 2:
        problems.append("UNO Q and MPU6050 overlap: increase width")
    for name in ("tft_holes", "uno_holes", "mpu_holes"):
        for x, y in lay[name]:
            if abs(x) > half_w - 2.5 or abs(y) > half_h - 2.5:
                problems.append(f"{name} hole at ({x:.1f}, {y:.1f}) is too close to the edge")
    for x, y in lay["slits"]:
        if abs(x) + p.slit_w / 2 > half_w - 1.5:
            problems.append(f"slit at x={x:.1f} breaks the edge")

    def hits_slit(x, y, half_w_, half_h_):
        return any(abs(x - sx) < half_w_ + p.slit_w / 2 and abs(y - sy) < half_h_ + p.slit_len / 2
                   for sx, sy in lay["slits"])

    for name, d in (("tft_holes", p.tft_hole_d), ("uno_holes", p.uno_hole_d), ("mpu_holes", p.mpu_hole_d)):
        for x, y in lay[name]:
            if hits_slit(x, y, d / 2 + 1.0, d / 2 + 1.0):
                problems.append(f"{name} hole at ({x:.1f}, {y:.1f}) runs into a strap slit")
    if hits_slit(mpu_cx, uno_cy, p.mpu_w / 2, p.mpu_h / 2):
        problems.append("MPU6050 footprint sits over a strap slit")
    return problems


def plate(p: P = P()) -> cq.Workplane:
    lay = layout(p)
    w = cq.Workplane("XY").rect(p.width, p.height).extrude(p.thickness).edges("|Z").fillet(p.corner_r)
    w = w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["tft_holes"]).hole(p.tft_hole_d)
    w = w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["uno_holes"]).hole(p.uno_hole_d)
    w = w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["mpu_holes"]).hole(p.mpu_hole_d)
    w = (w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["slits"])
         .slot2D(p.slit_len, p.slit_w, angle=90).cutThruAll())
    return w


def main() -> int:
    p = P()
    lay = layout(p)
    problems = check(p, lay)
    for line in problems:
        print("PROBLEM:", line)
    OUT.mkdir(parents=True, exist_ok=True)
    body = plate(p)
    cq.exporters.export(body.section(p.thickness / 2), str(OUT / "plate.dxf"))
    cq.exporters.export(body, str(OUT / "plate.svg"),
                        opt={"projectionDir": (0, 0, 1), "width": 800, "height": 1000, "showAxes": False})
    print(f"plate {p.width} x {p.height} x {p.thickness} mm; TFT centre {lay['tft']}, UNO Q centre "
          f"({lay['uno'][0]:.1f}, {lay['uno'][1]:.1f}), MPU centre ({lay['mpu'][0]:.1f}, {lay['mpu'][1]:.1f})")
    print("wrote", OUT / "plate.dxf", "and", OUT / "plate.svg")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
