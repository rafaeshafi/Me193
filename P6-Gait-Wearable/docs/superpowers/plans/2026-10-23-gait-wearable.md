# P6 Gait Wearable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thigh-worn UNO Q + MPU6050 + 3.2" TFT wearable that shows, live, whether the wearer is standing, walking, running or biking, from two side-by-side decision trees: Model 1 on the thigh-axis peak acceleration only, Model 2 on all six IMU channels.

**Architecture:** The MCU sketch reads the MPU6050's hardware FIFO at 100 Hz, ships 10-sample batches to the Linux side with `Bridge.notify("imu", bytes)`, and draws the TFT from a short key=value payload plus a byte stream for the waveform sweep. Python on the board (stdlib only, no numpy) cuts 2 s windows every 0.5 s, computes one feature dict, evaluates two JSON decision trees, majority-votes, and serves a phone page over Server-Sent Events with the guided recorder. The Mac trains both trees with scikit-learn from the recorded CSVs, using the very same `features.py`, and exports them as JSON.

**Tech Stack:** Arduino App Lab app (Zephyr sketch + Python 3.13 in the python-apps-base container), Arduino_RouterBridge + MsgPack, own ILI9341 and MPU6050 drivers (no external libs on the board), Python stdlib on the board, scikit-learn + numpy + matplotlib on the Mac, CadQuery 2.8 for the plate DXF, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-gait-wearable-design.md`

---

## Conventions every task follows

- Project root: `~/ME193/P6-Gait-Wearable`. All paths below are relative to it. The git repo root is `~/ME193` (remote `https://github.com/rafaeshafi/Me193.git`); commit messages start with `P6-Gait-Wearable:`. Commit as you go; do **not** push unless the user asks.
- No `Co-Authored-By` trailer on commits (the user's CLAUDE.md forbids it).
- Mac test venv: `.venv/` at the project root (gitignored). Create it in Task 1. Run tests with `.venv/bin/python -m pytest tests -q`.
- Sketch compiles locally in ~20 s with App Lab's arduino-cli (Task 1 wraps it in `scripts/compile.sh`). Board compiles take ~2 min, so always compile locally first.
- Board access is over the USB-C cable with App Lab's adb: `ADB="$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb"`. The app lives at `~/ArduinoApps/gait-wearable` on the board and is mounted at `/app` inside its container, so `/app/data` on the board's Linux host is `/home/arduino/ArduinoApps/gait-wearable/data`.
- Bridge gotchas (from whoop-hr): messages over ~270 bytes are rejected; `RpcCall::result()` on the MCU blocks forever, so the MCU only ever uses `Bridge.notify(...)` towards Python and plain providers for calls from Python.
- Code must run on Python 3.9 (Mac tests) and 3.13 (board): no `X | None` annotations at runtime, no `match`.
- The whoop-hr files reused here are copied, not symlinked, with the namespace renamed `whoophr` -> `gait`.

## File structure

```
app/app.yaml                       App Lab manifest (port 8080)
app/sketch/sketch.ino              providers tft/tfth, IMU FIFO drain, imu batches via Bridge.notify
app/sketch/sketch.yaml             profile copied from whoop-hr
app/sketch/src/ili9341.{h,cpp}     copied from whoop-hr (namespace gait)
app/sketch/src/tft_gfx.{h,cpp}     copied from whoop-hr (namespace gait); only the palette is used
app/sketch/src/mpu6050.{h,cpp}     MPU6050 FIFO driver
app/sketch/src/tft_ui.{h,cpp}      the gait screen: verdicts, cadence, peak g, prompt, waveform sweep
app/python/main.py                 wiring: Bridge, web, loop
app/python/imu.py                  batch decode + ring buffer
app/python/features.py             window features (pure Python; shared with training)
app/python/models.py               JSON decision tree evaluator + majority vote
app/python/recorder.py             guided / single-activity CSV recorder + rolling log
app/python/display.py              TFT payload and waveform bytes
app/python/pipeline.py             Gait: ties ring, features, models, vote, recorder together
app/python/web.py                  phone page, SSE stream, recorder controls (stdlib http.server)
app/python/page.html               the phone page (served by web.py)
models/model1.json, model2.json    seed trees (hand thresholds) replaced by training
train/requirements.txt             numpy scikit-learn matplotlib pytest
train/train.py                     CSV -> windows -> two trees -> JSON + confusion matrices
train/plots.py                     the three Notion figures -> docs/figures/
scripts/compile.sh                 local sketch compile
scripts/deploy.sh                  adb deploy (app + models), keeps data/, sets default app
scripts/pull_data.sh               adb pull of the recordings into data/
plate/plate.py                     CadQuery plate -> plate/out/plate.dxf + plate.svg
tests/conftest.py, tests/test_*.py
docs/notion.md                     Notion page text drafts
README.md                          short project readme for the GitHub link
```

---

### Task 1: Project skeleton, test venv, local compile script

**Files:**
- Create: `app/app.yaml`, `app/sketch/sketch.yaml`, `train/requirements.txt`, `tests/conftest.py`, `tests/test_smoke.py`, `scripts/compile.sh`
- Modify: `.gitignore` (already has data/, my_env/, .venv/, __pycache__/, .pytest_cache/, plate/out/)

- [ ] **Step 1: Create the App Lab manifest**

`app/app.yaml`:
```yaml
name: Gait Wearable
icon: 🦵
description: Thigh-worn gait identifier (stand / walk / run / bike) from an MPU6050, shown live on a 3.2" TFT. Two decision trees side by side, phone page on port 8080 for live view and labelled recording.

# A list of network ports that the application exposes.
ports: [8080]

# A list of bricks used by this application.
bricks: []
```

- [ ] **Step 2: Copy the sketch profile from whoop-hr**

```bash
mkdir -p ~/ME193/P6-Gait-Wearable/app/sketch/src
cp ~/whoop-hr/app/sketch/sketch.yaml ~/ME193/P6-Gait-Wearable/app/sketch/sketch.yaml
```

Expected content (verify with `cat`):
```yaml
profiles:
  default:
    fqbn: arduino:zephyr:unoq
    platforms:
      - platform: arduino:zephyr
    libraries:
      - MsgPack (0.4.2)
      - DebugLog (0.8.4)
      - ArxContainer (0.7.0)
      - ArxTypeTraits (0.3.1)

default_profile: default
```

- [ ] **Step 3: Mac requirements and venv**

`train/requirements.txt`:
```
numpy
scikit-learn
matplotlib
pytest
```

```bash
cd ~/ME193/P6-Gait-Wearable
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
"$PY" -m venv .venv && .venv/bin/pip install -q -r train/requirements.txt && .venv/bin/python -c "import sklearn, numpy, matplotlib; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: conftest and a smoke test**

`tests/conftest.py`:
```python
"""Make the board-side package and the training scripts importable from the tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("app/python", "train"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
```

`tests/test_smoke.py`:
```python
from pathlib import Path


def test_layout_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app" / "app.yaml").exists()
    assert (root / "app" / "sketch" / "sketch.yaml").exists()
```

Run: `.venv/bin/python -m pytest tests -q`
Expected: `1 passed`

- [ ] **Step 5: Local compile script**

`scripts/compile.sh`:
```bash
#!/usr/bin/env bash
# Compile the sketch on the Mac with App Lab's own arduino-cli (~20 s) instead of on the board (~2 min).
set -euo pipefail
cd "$(dirname "$0")/.."
WS="$HOME/Library/Application Support/arduino_applab_workspace"
CLI="$WS/resources/arduino/arduino-cli/arduino-cli"
"$CLI" --config-file "$WS/arduino15/arduino-cli.yaml" compile --fqbn arduino:zephyr:unoq \
  --build-path /tmp/claude-501/gait-build app/sketch "$@"
```
`chmod +x scripts/compile.sh`. It cannot run yet (no sketch.ino); Task 3 uses it.

- [ ] **Step 6: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable && git commit -m "P6-Gait-Wearable: project skeleton (App Lab manifest, sketch profile, Mac venv requirements, pytest conftest, local compile script)"
```

---

### Task 2: Laser-cut plate (CadQuery -> DXF)

**Files:**
- Create: `plate/plate.py`
- Output (gitignored): `plate/out/plate.dxf`, `plate/out/plate.svg`

Uses the existing CAD venv `~/whoop-hr/.venv-cad/bin/python` (cadquery 2.8, ezdxf). Layout, looking at the plate from the front with the hip at the top: TFT (93 x 60, landscape) across the top on 25 mm standoffs; UNO Q below it turned 90 degrees so its USB-C faces the knee edge and its hole-free middle carries the lower strap (10 mm standoffs); MPU6050 right of the UNO Q near the bottom, long side along the thigh so its X axis runs down the thigh (5 mm standoffs, pins up). The two elastic straps run on the FRONT face under the boards: in through a left slit, across, out through the right slit, buckle behind. `check()` keeps every standoff out of both 25.4 mm strap bands and the DXF export is verified to contain the geometry.

- [ ] **Step 1: Write the plate script**

`plate/plate.py`:
```python
"""Laser-cut 3 mm acrylic mounting plate for the P6 gait wearable (UNO Q + 3.2" ILI9341 TFT + MPU6050) on the thigh.

Front view, hip at the top (+Y), origin at the plate centre. The TFT spans the top. The UNO Q sits below it, turned
90 degrees so its USB-C faces the bottom (knee) edge and its hole-free middle carries the lower strap. The MPU6050
sits right of the UNO Q near the bottom, long (header) side along the thigh so its X axis runs down the thigh. Boards
stand on M3 standoffs (TFT 25 mm so its rear header housings clear the strap, UNO Q 10 mm, MPU 5 mm with its pins
pointing up). Two 25.4 mm elastic straps run across the FRONT face under the boards: in through a left slit, out
through the right slit, buckle behind the thigh. check() keeps every standoff out of both strap bands. Values marked
VERIFY want a caliper check before cutting.

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
    height: float = 144.0
    thickness: float = 3.0
    corner_r: float = 6.0
    margin: float = 3.5            # board edge to plate edge
    min_web: float = 2.0           # acrylic left between a hole and the plate edge
    standoff_r: float = 2.75       # M3 standoff, 5.5 mm across flats: straps stay this far from every hole centre
    # 3.2" TFT module (measured for whoop-hr); header column on its right edge, pins out the back (~20 mm, VERIFY)
    tft_w: float = 93.0
    tft_h: float = 60.0
    tft_hole_dx: float = 87.4
    tft_hole_dy: float = 54.4
    tft_hole_d: float = 3.2
    tft_standoff: float = 25.0     # mm; must exceed the header housings so they clear the strap underneath
    # UNO Q (UNO R3 outline) turned 90 degrees: USB-C on the bottom edge. Holes from the board's lower-left corner.
    uno_w: float = 53.34           # across the thigh
    uno_h: float = 68.58           # along the thigh
    uno_holes: tuple = ((50.80, 13.97), (2.54, 15.24), (45.72, 66.04), (17.78, 66.04))
    uno_hole_d: float = 3.2
    uno_strap_y: float = 40.6      # lower strap band centre above the UNO Q's bottom edge: between its hole rows
    gap: float = 8.0               # TFT bottom edge to UNO Q top edge
    # MPU6050 GY-521 breakout, long (header) side along the thigh so the printed X arrow runs down the thigh (VERIFY the
    # arrow on your board); its two holes sit at the corners of the long edge opposite the header (VERIFY both numbers)
    mpu_w: float = 16.0            # across the thigh
    mpu_h: float = 21.0            # along the thigh
    mpu_hole_dx: float = 15.5      # hole spacing along the long side
    mpu_hole_off: float = 6.0      # holes this far left of the board centreline (header on the right, pins up)
    mpu_hole_d: float = 3.2
    mpu_y: float = 14.0            # MPU centre above the UNO Q's bottom edge: below the lower strap band
    # straps: VELCRO 90441, 25.4 mm wide, ~1.5 mm thick
    strap_w: float = 25.4
    slit_len: float = 27.0
    slit_w: float = 4.0
    slit_inset: float = 3.0        # plate edge to slit


def layout(p: P = P()) -> dict:
    """Centres of everything, in plate coordinates (mm). Also used by the checks."""
    tft_cy = p.height / 2 - p.margin - p.tft_h / 2
    uno_cx = -p.width / 2 + p.margin + p.uno_w / 2
    uno_top = tft_cy - p.tft_h / 2 - p.gap
    uno_ll = (uno_cx - p.uno_w / 2, uno_top - p.uno_h)
    uno_cy = uno_top - p.uno_h / 2
    mpu_cx = uno_ll[0] + p.uno_w + 2.0 + p.mpu_w / 2   # 2 mm right of the UNO Q
    mpu_cy = uno_ll[1] + p.mpu_y
    strap_lo_y = uno_ll[1] + p.uno_strap_y
    slit_x = p.width / 2 - p.slit_inset - p.slit_w / 2
    return {
        "tft": (0.0, tft_cy),
        "tft_holes": [(sx * p.tft_hole_dx / 2, tft_cy + sy * p.tft_hole_dy / 2) for sx in (-1, 1) for sy in (-1, 1)],
        "uno": (uno_cx, uno_cy),
        "uno_holes": [(uno_ll[0] + hx, uno_ll[1] + hy) for hx, hy in p.uno_holes],
        "mpu": (mpu_cx, mpu_cy),
        "mpu_holes": [(mpu_cx - p.mpu_hole_off, mpu_cy - p.mpu_hole_dx / 2),
                      (mpu_cx - p.mpu_hole_off, mpu_cy + p.mpu_hole_dx / 2)],
        "straps": [tft_cy, strap_lo_y],
        "slits": [(sx * slit_x, y) for sx in (-1, 1) for y in (tft_cy, strap_lo_y)],
    }


def check(p: P, lay: dict) -> list[str]:
    """Plain-English clearance report; a non-empty list means do not cut."""
    problems = []
    half_w, half_h = p.width / 2, p.height / 2
    holes = [(name, x, y, d) for name, d in (("tft", p.tft_hole_d), ("uno", p.uno_hole_d), ("mpu", p.mpu_hole_d))
             for x, y in lay[name + "_holes"]]
    # boards inside the plate
    for name, (cx, cy), (w, h) in (("TFT", lay["tft"], (p.tft_w, p.tft_h)), ("UNO Q", lay["uno"], (p.uno_w, p.uno_h)),
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
    w = w.faces(">Z").workplane(centerOption="CenterOfBoundBox").pushPoints(lay["tft_holes"]).hole(p.tft_hole_d)
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
    print(f"plate {p.width} x {p.height} x {p.thickness} mm; TFT centre {lay['tft']} on {p.tft_standoff} mm standoffs, "
          f"UNO Q centre ({lay['uno'][0]:.1f}, {lay['uno'][1]:.1f}) USB-C down, MPU centre ({lay['mpu'][0]:.1f}, "
          f"{lay['mpu'][1]:.1f}), strap bands at y = {lay['straps'][0]:.1f} and {lay['straps'][1]:.1f}")
    print("wrote", dxf, "and", svg)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

```bash
cd ~/ME193/P6-Gait-Wearable && ~/whoop-hr/.venv-cad/bin/python plate/plate.py
```
Expected: no `PROBLEM:` lines, a summary line with `strap bands at y = 38.5 and -27.5`, and `wrote .../plate/out/plate.dxf and .../plate.svg`. Exit code 0.

- [ ] **Step 3: Eyeball the preview**

```bash
qlmanage -t -s 1000 -o /tmp/claude-501 ~/ME193/P6-Gait-Wearable/plate/out/plate.svg && open /tmp/claude-501/plate.svg.png
```
Check: 4 TFT holes in a rectangle across the top; below them 4 UNO holes (two near the top of the UNO area, two near the bottom) with a wide hole-free band between; 2 MPU holes stacked vertically right of the UNO area near the bottom; 4 vertical slits near the left and right edges, the lower pair level with the UNO's hole-free band.

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/plate/plate.py && git commit -m "P6-Gait-Wearable: parametric laser-cut mounting plate (TFT on top, UNO Q + MPU6050 below, two strap slits per side) with a clearance report and DXF/SVG export"
```

Tell the user: the DXF is at `plate/out/plate.dxf`; before cutting, verify with calipers the MPU6050 hole spacing (15.5 mm) and their offset from the board centreline (6 mm), and that the TFT's header housings are under 25 mm (its standoff height).

---

### Task 3: Sketch, part 1: TFT driver copies, MPU6050 FIFO driver, IMU batches over the Bridge

**Files:**
- Create: `app/sketch/src/ili9341.{h,cpp}`, `app/sketch/src/tft_gfx.{h,cpp}` (copies), `app/sketch/src/mpu6050.{h,cpp}`, `app/sketch/sketch.ino`

- [ ] **Step 1: Copy the TFT driver and palette from whoop-hr, renaming the namespace**

```bash
cd ~/ME193/P6-Gait-Wearable/app/sketch/src
for f in ili9341.h ili9341.cpp tft_gfx.h tft_gfx.cpp; do sed 's/whoophr/gait/g' ~/whoop-hr/app/sketch/src/$f > $f; done
grep -c "namespace gait" ili9341.h tft_gfx.h
```
Expected: `1` for each.

- [ ] **Step 2: MPU6050 FIFO driver header**

`app/sketch/src/mpu6050.h`:
```cpp
#pragma once
#include <stdint.h>

namespace gait {

// MPU6050 at 0x68 (AD0 low) on Wire, 100 Hz into the chip's own FIFO: accel +-8 g, gyro +-1000 deg/s, DLPF 44 Hz.
// loop() drains the FIFO, so the sampling clock is the sensor's and the sketch may be late by up to ~800 ms (1024 B).
// A sample is 12 bytes straight from the FIFO: ax ay az gx gy gz as big-endian int16.
class Mpu6050 {
 public:
  static const int kBytesPerSample = 12;
  bool begin();                      // true when WHO_AM_I answered and the FIFO runs; false leaves present() false
  bool present() const { return present_; }
  // Drains up to maxSamples whole samples into out (kBytesPerSample each) in one burst. Returns the count, or -1 on a
  // bus error, after which present() is false and begin() must be called again. Restarts the FIFO after an overflow
  // (returns 0). Every restart, including begin(), bumps resets(): samples before and after it are not contiguous.
  int read(uint8_t* out, int maxSamples);
  uint8_t resets() const { return resets_; }

 private:
  bool write(uint8_t reg, uint8_t value);
  bool readRegs(uint8_t reg, uint8_t* buf, int len);
  bool present_ = false;
  uint8_t resets_ = 0;
};

}  // namespace gait
```

- [ ] **Step 3: MPU6050 driver implementation**

`app/sketch/src/mpu6050.cpp`:
```cpp
#include "mpu6050.h"

#include <Arduino.h>
#include <Wire.h>

namespace gait {

namespace {
const uint8_t kAddr = 0x68;
const uint8_t kSmplrtDiv = 0x19, kConfig = 0x1A, kGyroConfig = 0x1B, kAccelConfig = 0x1C, kFifoEn = 0x23,
              kIntStatus = 0x3A, kUserCtrl = 0x6A, kPwrMgmt1 = 0x6B, kFifoCountH = 0x72, kFifoRW = 0x74,
              kWhoAmI = 0x75;
const int kFifoBytes = 1024;
}  // namespace

bool Mpu6050::write(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(kAddr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool Mpu6050::readRegs(uint8_t reg, uint8_t* buf, int len) {
  Wire.beginTransmission(kAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)kAddr, len) != len) return false;
  for (int i = 0; i < len; ++i) buf[i] = Wire.read();
  return true;
}

bool Mpu6050::begin() {
  present_ = false;
  Wire.begin();
  Wire.setClock(400000);
  uint8_t who = 0;
  // genuine parts answer 0x68; common clones answer 0x70/0x72/0x98, so only an absent device (0x00/0xFF) is rejected
  if (!readRegs(kWhoAmI, &who, 1) || who == 0x00 || who == 0xFF) return false;
  bool ok = write(kPwrMgmt1, 0x80);  // reset
  delay(100);
  ok = write(kPwrMgmt1, 0x01) && ok;    // wake, clock from the gyro X PLL
  ok = write(kSmplrtDiv, 9) && ok;      // 1 kHz / (1 + 9) = 100 Hz
  ok = write(kConfig, 0x03) && ok;      // DLPF 44 Hz accel / 42 Hz gyro
  ok = write(kGyroConfig, 0x10) && ok;  // +-1000 deg/s
  ok = write(kAccelConfig, 0x10) && ok; // +-8 g
  ok = write(kUserCtrl, 0x04) && ok;    // FIFO reset
  ok = write(kFifoEn, 0x78) && ok;      // accel x y z then gyro x y z into the FIFO (12 bytes per sample)
  ok = write(kUserCtrl, 0x40) && ok;    // FIFO enable
  present_ = ok;
  if (ok) ++resets_;
  return ok;
}

int Mpu6050::read(uint8_t* out, int maxSamples) {
  if (!present_) return -1;
  uint8_t status = 0, cnt[2];
  if (!readRegs(kIntStatus, &status, 1) || !readRegs(kFifoCountH, cnt, 2)) {
    present_ = false;
    return -1;
  }
  int bytes = (cnt[0] << 8) | cnt[1];
  if ((status & 0x10) || bytes >= kFifoBytes - kBytesPerSample) {  // overflow: the stream is misaligned, start again
    present_ = write(kUserCtrl, 0x04) && write(kUserCtrl, 0x40);
    ++resets_;
    return present_ ? 0 : -1;
  }
  int n = bytes / kBytesPerSample;
  if (n > maxSamples) n = maxSamples;
  if (n > 0 && !readRegs(kFifoRW, out, n * kBytesPerSample)) {  // one burst; the FIFO register auto-advances
    present_ = false;
    return -1;
  }
  return n;
}

}  // namespace gait
```

- [ ] **Step 4: A temporary tft_ui stub so the sketch compiles before Task 4**

`app/sketch/src/tft_ui.h` (Task 4 replaces this file entirely):
```cpp
#pragma once
#include <stdint.h>

namespace gait {
class TftUi {
 public:
  void begin(int, int, int) {}
  bool setPayload(const char*) { return true; }
  void pushWave(const uint8_t*, int) {}
  void render() {}
};
}  // namespace gait
```

- [ ] **Step 5: The sketch**

`app/sketch/sketch.ino`:
```cpp
// Gait wearable, the MCU side. Dumb and fast; all classification lives in the Python app over the Bridge.
//   IMU:  the MPU6050's FIFO is drained in loop(); every 10 samples (100 ms) a batch goes to Python with
//         Bridge.notify("imu", bytes): byte 0 = sequence number, byte 1 = sample count, byte 2 = FIFO reset counter
//         (changes when samples stop being contiguous), then 12 bytes per sample (ax ay az gx gy gz, int16 big-endian,
//         raw). An empty batch every 500 ms says "alive, no IMU".
//   TFT:  provider "tft" takes one short key=value payload (see tft_ui.h); "tfth" takes waveform bytes to sweep.
// Wiring: TFT SCK D13, MOSI D11, CS D10, DC D9, RST D8, VCC 5V, LED 3V3; MPU6050 on the SDA/SCL header at 3V3, AD0 to GND.

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <vector>

#include "src/mpu6050.h"
#include "src/tft_ui.h"

const int kTftCs = 10, kTftDc = 9, kTftRst = 8;
const int kBatch = 10;  // 10 x 12 B + 3 B header = 123 B, well under the Bridge's message size limit
const int kHeader = 3;
const uint32_t kProbeMs = 2000, kHeartbeatMs = 500;

gait::Mpu6050 imu;
gait::TftUi tft;
// The providers run on the Bridge's own thread; uiLock guards the hand-over of these buffers to loop().
static char tftPayload[256];
static bool tftDirty = false;
static uint8_t wave[64];
static int waveCount = -1;  // -1 = nothing new
static uint8_t batch[kHeader + kBatch * gait::Mpu6050::kBytesPerSample];
static int batched = 0;
static uint8_t seq = 0, sentResets = 0;
static uint32_t nextProbeMs = 0, nextHeartbeatMs = 0;
K_MUTEX_DEFINE(uiLock);

bool showTft(MsgPack::str_t payload) {
  k_mutex_lock(&uiLock, K_FOREVER);
  strncpy(tftPayload, payload.c_str(), sizeof tftPayload - 1);
  tftPayload[sizeof tftPayload - 1] = '\0';
  tftDirty = true;
  k_mutex_unlock(&uiLock);
  return true;
}

bool pushWave(std::vector<uint8_t> samples) {
  k_mutex_lock(&uiLock, K_FOREVER);
  int have = waveCount < 0 ? 0 : waveCount;
  int room = (int)sizeof wave - have;
  int n = (int)samples.size() < room ? (int)samples.size() : room;
  memcpy(wave + have, samples.data() + (samples.size() - n), n);
  waveCount = have + n;
  k_mutex_unlock(&uiLock);
  return true;
}

void flush() {
  batch[0] = seq++;
  batch[1] = (uint8_t)batched;
  batch[2] = sentResets;
  std::vector<uint8_t> msg(batch, batch + kHeader + batched * gait::Mpu6050::kBytesPerSample);
  Bridge.notify("imu", msg);
  batched = 0;
  nextHeartbeatMs = millis() + kHeartbeatMs;
}

void setup() {
  tft.begin(kTftCs, kTftDc, kTftRst);
  Bridge.begin();
  Bridge.provide("tft", showTft);
  Bridge.provide("tfth", pushWave);
}

void loop() {
  uint32_t now = millis();
  if (!imu.present()) {
    if ((int32_t)(now - nextProbeMs) >= 0) {
      nextProbeMs = now + kProbeMs;
      imu.begin();
    }
  } else {
    int n;  // drain everything that is waiting, however long the last render took
    while ((n = imu.read(batch + kHeader + batched * gait::Mpu6050::kBytesPerSample, kBatch - batched)) > 0) {
      batched += n;
      if (batched >= kBatch) flush();
    }
  }
  if (imu.resets() != sentResets) {  // the stream restarted: what is batched belongs to the old stream
    if (batched > 0) flush();
    sentResets = imu.resets();
  }
  if ((int32_t)(now - nextHeartbeatMs) >= 0) flush();  // partial batch, or empty = alive but no IMU
  static char payloadCopy[sizeof tftPayload];
  static uint8_t waveCopy[sizeof wave];
  bool haveText = false;
  int haveWave = -1;
  k_mutex_lock(&uiLock, K_FOREVER);
  if (tftDirty) {
    tftDirty = false;
    memcpy(payloadCopy, tftPayload, sizeof payloadCopy);
    haveText = true;
  }
  if (waveCount >= 0) {
    haveWave = waveCount;
    waveCount = -1;
    memcpy(waveCopy, wave, haveWave);
  }
  k_mutex_unlock(&uiLock);
  if (haveText) tft.setPayload(payloadCopy);
  if (haveWave >= 0) tft.pushWave(waveCopy, haveWave);
  tft.render();
  delay(5);
}
```

- [ ] **Step 6: Compile locally**

```bash
cd ~/ME193/P6-Gait-Wearable && scripts/compile.sh 2>&1 | tail -4
```
Expected: `Sketch uses ... bytes` lines and no `error:`. If `Wire.setClock` does not exist on this core, delete that line (the default 100 kHz still moves 1.2 kB/s with room to spare).

- [ ] **Step 7: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/sketch && git commit -m "P6-Gait-Wearable: sketch drains the MPU6050 FIFO at 100 Hz and ships 10-sample batches (seq, count, reset counter) to Python with Bridge.notify, TFT driver and palette copied from whoop-hr, tft providers via provide_safe"
```

---

### Task 4: Sketch, part 2: the gait screen

**Files:**
- Replace: `app/sketch/src/tft_ui.h`
- Create: `app/sketch/src/tft_ui.cpp`

Screen (320 x 240, black): big Model 2 word top-left (scale 6, coloured per class); small grey captions; Model 1 word at scale 2 in green when the models agree, amber when they differ; a prompt line (recording script) at scale 2; a 240 x 120 waveform sweep bottom-left with a dim 1 g line; cadence and peak g stacked on the right of the sweep; a short status word under them.

- [ ] **Step 1: Header**

`app/sketch/src/tft_ui.h`:
```cpp
#pragma once
#include <stdint.h>

#include "ili9341.h"

namespace gait {

// The gait screen. Python sends one payload per change as key=value pairs joined by ';':
//   act=STAND|WALK|RUN|BIKE|NO IMU|---   (Model 2, all IMU channels: the big word)
//   m1=STAND|WALK|RUN|BIKE|---           (Model 1, peak g only)
//   agree=1|0  cad=123|--  pk=2.3|--  prompt=WALK 18s (empty hides it)  st=REC|--- (short status)
// The waveform arrives separately as bytes (pushWave): thigh-axis accel, 0..255 = -4..+8 g, newest last, 50 Hz.
// It is drawn as an oscilloscope sweep: one column per sample, cursor wraps at the right edge.
class TftUi {
 public:
  static const int kWaveX = 8, kWaveY = 112, kWaveW = 240, kWaveH = 120;
  void begin(int cs, int dc, int rst);
  bool setPayload(const char* payload);  // false if malformed (previous payload kept)
  void pushWave(const uint8_t* samples, int count);
  void render();  // call from loop(); redraws only fields that changed

 private:
  struct Fields {
    char act[8], m1[8], cad[6], pk[6], prompt[24], st[8];
    int agree;  // 1 agree, 0 differ, -1 neutral
  };
  static bool parse(const char* payload, Fields& out);
  static uint16_t colorFor(const char* act);
  static int yOf(int sample);
  void field(int x, int y, int scale, const char* text, int maxChars, uint16_t color);
  void drawStatic();
  Ili9341 lcd_;
  Fields want_{}, shown_{};
  bool dirty_ = true, first_ = true;
  int waveCol_ = 0, waveLast_ = -1;
};

}  // namespace gait
```

- [ ] **Step 2: Implementation**

`app/sketch/src/tft_ui.cpp`:
```cpp
#include "tft_ui.h"

#include <stdlib.h>
#include <string.h>

#include "tft_gfx.h"

namespace gait {

namespace {
using brand::kBlack; using brand::kWhite; using brand::kGrey;
const uint16_t kGreen = brand::kRecGreen;
const uint16_t kAmber = brand::kRecYellow;
const uint16_t kRed = brand::kRecRed;
const uint16_t kBlue = brand::kStrain;
const uint16_t kOrange = Ili9341::rgb(255, 120, 0);
const uint16_t kTrace = Ili9341::rgb(0, 230, 230);
const int kOneG = 106;  // (1 g + 4) / 12 * 255
const int kRightX = 256;

bool take(const char*& p, const char* key, char* out, int cap) {
  // copies the value of "key=" if the pair at p has that key; always advances p past the pair, even a malformed one
  const char* end = strchr(p, ';');
  if (!end) end = p + strlen(p);
  const char* eq = (const char*)memchr(p, '=', end - p);
  bool match = eq && (int)strlen(key) == eq - p && strncmp(p, key, eq - p) == 0;
  if (match && out) {
    int n = (int)(end - (eq + 1));
    if (n > cap - 1) n = cap - 1;
    memcpy(out, eq + 1, n);
    out[n] = '\0';
  }
  p = *end ? end + 1 : end;
  return match;
}
}  // namespace

bool TftUi::parse(const char* payload, Fields& f) {
  memset(&f, 0, sizeof f);
  strcpy(f.act, "---"); strcpy(f.m1, "---"); strcpy(f.cad, "--"); strcpy(f.pk, "--");
  f.agree = -1;
  const char* p = payload;
  bool any = false;
  char num[8];
  while (*p) {
    const char* start = p;
    if (take(p, "act", f.act, sizeof f.act) || take((p = start), "m1", f.m1, sizeof f.m1) ||
        take((p = start), "cad", f.cad, sizeof f.cad) || take((p = start), "pk", f.pk, sizeof f.pk) ||
        take((p = start), "prompt", f.prompt, sizeof f.prompt) || take((p = start), "st", f.st, sizeof f.st)) {
      any = true;
    } else if (take((p = start), "agree", num, sizeof num)) {
      f.agree = (num[0] == '-') ? -1 : (atoi(num) != 0 ? 1 : 0);
      any = true;
    } else {
      p = start;
      take(p, "", nullptr, 0);  // unknown or malformed pair: skip it
    }
  }
  return any;
}

uint16_t TftUi::colorFor(const char* act) {
  if (strcmp(act, "STAND") == 0) return kWhite;
  if (strcmp(act, "WALK") == 0) return kGreen;
  if (strcmp(act, "RUN") == 0) return kOrange;
  if (strcmp(act, "BIKE") == 0) return kBlue;
  if (strcmp(act, "NO IMU") == 0) return kRed;
  return kGrey;
}

int TftUi::yOf(int sample) { return kWaveY + kWaveH - 2 - sample * (kWaveH - 4) / 255; }

void TftUi::begin(int cs, int dc, int rst) {
  lcd_.begin(cs, dc, rst);
  parse("act=---;m1=---", want_);
  first_ = true;
  dirty_ = true;
}

bool TftUi::setPayload(const char* payload) {
  Fields f;
  if (!parse(payload, f)) return false;
  want_ = f;
  dirty_ = true;
  return true;
}

void TftUi::field(int x, int y, int scale, const char* text, int maxChars, uint16_t color) {
  char buf[32];
  int n = 0;
  while (n < maxChars && n < (int)sizeof buf - 1 && text[n]) { buf[n] = text[n]; ++n; }
  buf[n] = '\0';
  lcd_.drawText(x, y, buf, scale, color, kBlack);
  int used = n * (Ili9341::kGlyphW + 1) * scale;  // drawText paints the gap after the last glyph too
  int box = Ili9341::textWidth(maxChars, scale) + scale;
  if (box > used) lcd_.fillRect(x + used, y, box - used, Ili9341::textHeight(scale), kBlack);
}

void TftUi::drawStatic() {
  lcd_.fillScreen(kBlack);
  lcd_.drawText(8, 50, "MODEL 2: ALL SIX IMU CHANNELS", 1, kGrey, kBlack);
  lcd_.drawText(8, 62, "MODEL 1: PEAK G ONLY", 1, kGrey, kBlack);
  lcd_.drawText(kRightX, kWaveY, "CADENCE", 1, kGrey, kBlack);
  lcd_.drawText(kRightX, kWaveY + 48, "PEAK G", 1, kGrey, kBlack);
  lcd_.fillRect(kWaveX, kWaveY, kWaveW, 1, kGrey);
  lcd_.fillRect(kWaveX, kWaveY + kWaveH - 1, kWaveW, 1, kGrey);
  lcd_.fillRect(kWaveX, yOf(kOneG), kWaveW, 1, brand::kTrack);
  waveCol_ = 0;
  waveLast_ = -1;
}

void TftUi::pushWave(const uint8_t* samples, int count) {
  if (first_) return;  // the static frame is not up yet
  for (int i = 0; i < count; ++i) {
    int x = kWaveX + waveCol_;
    int y = yOf(samples[i]);
    int y0 = waveLast_ < 0 ? y : yOf(waveLast_);
    lcd_.fillRect(x, yOf(kOneG), 1, 1, brand::kTrack);
    int top = y < y0 ? y : y0;
    int h = (y > y0 ? y - y0 : y0 - y) + 1;
    lcd_.fillRect(x, top, 1, h, kTrace);
    waveLast_ = samples[i];
    waveCol_ = (waveCol_ + 1) % kWaveW;
    lcd_.fillRect(kWaveX + waveCol_, kWaveY + 1, 1, kWaveH - 2, kBlack);  // gap ahead of the cursor
  }
}

void TftUi::render() {
  if (!dirty_) return;
  dirty_ = false;
  bool full = first_;
  if (first_) {
    drawStatic();
    first_ = false;
  }
  const Fields& w = want_;
  Fields& s = shown_;
  if (full || strcmp(w.act, s.act) != 0) field(8, 4, 6, w.act, 6, colorFor(w.act));
  if (full || strcmp(w.m1, s.m1) != 0 || w.agree != s.agree) {
    uint16_t c = w.agree < 0 ? kGrey : (w.agree ? kGreen : kAmber);
    field(8, 72, 2, w.m1, 6, c);
    field(90, 72, 2, w.agree < 0 ? "" : (w.agree ? "AGREES" : "DIFFERS"), 7, c);
  }
  if (full || strcmp(w.prompt, s.prompt) != 0) field(8, 92, 2, w.prompt, 20, kWhite);
  if (full || strcmp(w.cad, s.cad) != 0) field(kRightX, kWaveY + 12, 3, w.cad, 3, kWhite);
  if (full || strcmp(w.pk, s.pk) != 0) field(kRightX, kWaveY + 60, 3, w.pk, 3, kWhite);
  if (full || strcmp(w.st, s.st) != 0) field(kRightX, kWaveY + 100, 1, w.st, 9, kAmber);
  s = w;
}

}  // namespace gait
```

- [ ] **Step 3: Compile**

```bash
cd ~/ME193/P6-Gait-Wearable && scripts/compile.sh 2>&1 | tail -4
```
Expected: compiles with no `error:`.

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/sketch && git commit -m "P6-Gait-Wearable: TFT gait screen (big Model 2 word coloured per class, Model 1 word green/amber for agree/differ, recording prompt, 240-column accel sweep with a 1 g line, cadence and peak g, status)"
```

---

### Task 5: `imu.py`: batch decode and ring buffer

**Files:**
- Create: `app/python/imu.py`, `tests/test_imu.py`

- [ ] **Step 1: Failing tests**

`tests/test_imu.py`:
```python
import struct

import pytest

from imu import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB, Ring, decode


def batch(seq, samples, resets=0):
    """samples: list of 6 raw int16 tuples -> the sketch's wire format."""
    body = b"".join(struct.pack(">6h", *s) for s in samples)
    return bytes([seq, len(samples), resets]) + body


def test_decode_scales_to_g_and_dps():
    seq, resets, samples = decode(batch(7, [(4096, 0, -4096, 0, 16384, 0)], resets=3))
    assert seq == 7 and resets == 3 and len(samples) == 1
    ax, ay, az, gx, gy, gz = samples[0]
    assert ax == pytest.approx(4096 * ACCEL_G_PER_LSB) == pytest.approx(1.0)
    assert az == pytest.approx(-1.0)
    assert gy == pytest.approx(16384 * GYRO_DPS_PER_LSB) == pytest.approx(500.0)


def test_decode_accepts_list_of_ints_and_empty_batch():
    seq, resets, samples = decode(list(batch(3, [])))
    assert seq == 3 and resets == 0 and samples == []


def test_decode_rejects_malformed():
    with pytest.raises(ValueError):
        decode(b"\x01\x00")
    with pytest.raises(ValueError):
        decode(bytes([0, 2, 0]) + b"\x00" * 12)  # says 2 samples, carries 1


def test_ring_window_and_dropped_batches():
    ring = Ring(seconds=1.0, rate=10)
    assert ring.window(0.5) is None
    for seq in range(3):
        ring.push(seq, [(seq, 0, 0, 0, 0, 0)] * 4)  # 12 samples into a 10-sample ring
    assert ring.total == 12 and ring.dropped == 0
    win = ring.window(0.5)  # 5 newest samples
    assert len(win) == 5 and win[-1][0] == 2
    ring.push(5, [(5, 0, 0, 0, 0, 0)])  # seq 3 and 4 never came
    assert ring.dropped == 1
    assert ring.window(0.5)[-1][0] == 5
    ring.clear()                        # the sketch said the stream restarted
    assert ring.window(0.5) is None and ring.total == 13


def test_ring_sequence_wrap_and_clear_keep_counters():
    ring = Ring(seconds=1.0, rate=10)
    ring.push(255, [(0,) * 6])
    ring.push(0, [(0,) * 6])
    assert ring.dropped == 0
    ring.push(2, [(0,) * 6])
    assert ring.dropped == 1
    ring.clear()
    ring.push(9, [(0,) * 6])
    assert ring.dropped == 1 and ring.total == 4
```

Run: `.venv/bin/python -m pytest tests/test_imu.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'imu'`

- [ ] **Step 2: Implementation**

`app/python/imu.py`:
```python
"""Decode the sketch's IMU batches and keep a short ring buffer of samples.

Wire format (see sketch.ino): byte 0 sequence number, byte 1 sample count, byte 2 FIFO reset counter (a change
means the samples are not contiguous with the previous batch), then 12 bytes per sample (ax ay az gx gy gz as
big-endian int16, raw). Full scale is +-8 g and +-1000 deg/s.
"""
import struct
from collections import deque

RATE = 100
HEADER = 3
BYTES_PER_SAMPLE = 12
ACCEL_G_PER_LSB = 8.0 / 32768
GYRO_DPS_PER_LSB = 1000.0 / 32768


def decode(batch):
    """-> (seq, resets, [(ax, ay, az, gx, gy, gz), ...]) in g and deg/s. Raises ValueError on a malformed batch."""
    if isinstance(batch, list):
        batch = bytes(batch)
    if len(batch) < HEADER:
        raise ValueError("batch shorter than its header")
    seq, n, resets = batch[0], batch[1], batch[2]
    if len(batch) != HEADER + n * BYTES_PER_SAMPLE:
        raise ValueError("batch says %d samples but carries %d bytes" % (n, len(batch) - HEADER))
    raw = struct.unpack(">%dh" % (6 * n), batch[HEADER:])
    samples = [
        (raw[i] * ACCEL_G_PER_LSB, raw[i + 1] * ACCEL_G_PER_LSB, raw[i + 2] * ACCEL_G_PER_LSB,
         raw[i + 3] * GYRO_DPS_PER_LSB, raw[i + 4] * GYRO_DPS_PER_LSB, raw[i + 5] * GYRO_DPS_PER_LSB)
        for i in range(0, 6 * n, 6)
    ]
    return seq, resets, samples


class Ring:
    """The last `seconds` of samples; counts sequence-number gaps (a batch or more went missing)."""

    def __init__(self, seconds=5.0, rate=RATE):
        self.rate = rate
        self._buf = deque(maxlen=int(seconds * rate))
        self._seq = None
        self.total = 0
        self.dropped = 0

    def push(self, seq, samples):
        if self._seq is not None and seq != (self._seq + 1) % 256:
            self.dropped += 1
        self._seq = seq
        self._buf.extend(samples)
        self.total += len(samples)

    def clear(self):
        """Forget the buffered samples and the sequence (the sensor stream restarted); counters are kept."""
        self._buf.clear()
        self._seq = None

    def window(self, seconds=2.0):
        """The newest `seconds` of samples as a list, or None until that many have arrived."""
        n = int(round(seconds * self.rate))
        if len(self._buf) < n:
            return None
        return list(self._buf)[-n:]
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_imu.py -q`
Expected: `5 passed`

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/imu.py P6-Gait-Wearable/tests/test_imu.py && git commit -m "P6-Gait-Wearable: decode the sketch's IMU batches (g, deg/s) and keep a 5 s ring buffer that counts dropped batches"
```

---

### Task 6: `features.py`: the window features both models read

**Files:**
- Create: `app/python/features.py`, `tests/synth.py`, `tests/test_features.py`

Pure Python on purpose: the board has no guaranteed numpy, and training must use this exact file.

- [ ] **Step 1: Synthetic signals for the tests (also reused by later tasks)**

`tests/synth.py`:
```python
"""Idealised thigh IMU signals for tests: (ax, ay, az, gx, gy, gz) in g and deg/s, X along the thigh."""
import math


def synth(kind, seconds=2.0, rate=100, phase=0.0):
    out = []
    for i in range(int(seconds * rate)):
        t = i / rate + phase
        if kind == "STAND":
            out.append((1.0 + 0.005 * math.sin(37 * t), 0.02 * math.sin(7 * t), 0.003 * math.sin(11 * t),
                        1.0 + 0.3 * math.sin(23 * t), -1.0, 0.5))  # ~5 mg accel noise, ~1 deg/s gyro bias plus a little jitter
        elif kind == "WALK":   # 1.8 steps/s, heel strikes ~1.6 g, thigh pitching at half the step rate
            out.append((1.0 + 0.6 * math.sin(2 * math.pi * 1.8 * t), 0.1, 0.05,
                        150 * math.sin(2 * math.pi * 0.9 * t), 10.0, 5.0))
        elif kind == "RUN":    # 2.8 steps/s, sharp 3.5 g peaks
            s = math.sin(2 * math.pi * 2.8 * t)
            out.append((1.0 + 2.5 * (s if s > 0 else 0.0), 0.3 * s, 0.1,
                        350 * math.sin(2 * math.pi * 1.4 * t), 20.0, 10.0))
        elif kind == "BIKE":   # lying back: gravity on Y, smooth 1.2 Hz pedalling, big gyro
            out.append((0.3 * math.sin(2 * math.pi * 1.2 * t), 0.95, 0.1,
                        200 * math.sin(2 * math.pi * 1.2 * t), 15.0, 5.0))
        else:
            raise ValueError(kind)
    return out
```

- [ ] **Step 2: Failing tests**

`tests/test_features.py`:
```python
import pytest

from features import MODEL1_FEATURES, MODEL2_FEATURES, compute, dominant_frequency
from synth import synth


def test_feature_sets_and_keys():
    f = compute(synth("WALK"))
    assert MODEL1_FEATURES == ["peak_x"]
    assert set(MODEL1_FEATURES) <= set(f) and set(MODEL2_FEATURES) <= set(f)
    assert all(isinstance(v, float) for v in f.values())


def test_peak_x_orders_stand_walk_run_and_bike_hides_among_them():
    peaks = {k: compute(synth(k))["peak_x"] for k in ("STAND", "WALK", "RUN", "BIKE")}
    assert peaks["STAND"] < peaks["WALK"] < peaks["RUN"]
    assert peaks["BIKE"] < peaks["WALK"]            # the Model 1 failure mode: biking looks quieter than walking
    assert peaks["STAND"] == pytest.approx(1.0, abs=0.05)


def test_gyro_and_gravity_features_separate_bike():
    f = {k: compute(synth(k)) for k in ("STAND", "WALK", "RUN", "BIKE")}
    assert f["BIKE"]["gyro_ratio"] > 2 * max(f["STAND"]["gyro_ratio"], f["WALK"]["gyro_ratio"], f["RUN"]["gyro_ratio"])
    assert f["STAND"]["gyro_ratio"] < 0.05
    assert f["BIKE"]["mean_y"] > 0.8 and f["WALK"]["mean_y"] < 0.3
    assert f["STAND"]["gyro_rms"] < 5 < f["WALK"]["gyro_rms"] < f["RUN"]["gyro_rms"]


def test_dominant_frequency_is_the_stride_or_pedal_rate():
    assert compute(synth("WALK"))["dom_freq"] == pytest.approx(0.9, abs=0.15)   # thigh swings once per stride
    assert compute(synth("RUN"))["dom_freq"] == pytest.approx(1.4, abs=0.15)
    assert compute(synth("BIKE"))["dom_freq"] == pytest.approx(1.2, abs=0.15)  # once per pedal revolution, not doubled
    assert dominant_frequency([1.0] * 200) == 0.5  # flat signal: the lowest bin, no crash
    assert dominant_frequency([1.0]) == 0.5        # degenerate window, no crash
```

Run: `.venv/bin/python -m pytest tests/test_features.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'features'`

- [ ] **Step 3: Implementation**

`app/python/features.py`:
```python
"""Window features for both models, from a list of (ax, ay, az, gx, gy, gz) samples in g and deg/s.

Pure Python so the same file runs on the board (no numpy there) and in training on the Mac. X is the thigh axis.
Model 1 is allowed exactly one feature, peak_x. Model 2 reads MODEL2_FEATURES.
dom_freq is the thigh swing rate (stride or pedal cycle), not the step rate; cadence in steps/min is 2 x 60 x dom_freq for walk and run.
"""
import math

RATE = 100
X, Y, Z, GX, GY, GZ = range(6)
MODEL1_FEATURES = ["peak_x"]
MODEL2_FEATURES = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mag_std", "mag_peak",
                   "gyro_rms", "gyro_peak", "dom_freq", "gyro_ratio"]
FREQ_LO, FREQ_HI, FREQ_STEP = 0.5, 4.0, 0.1


def _mean(v):
    return sum(v) / len(v)


def _std(v):
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v))


def _percentile(v, q):
    s = sorted(v)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _swing_axis(window):
    """The signed gyro axis with the most movement: the thigh's flexion axis, whichever way the breakout is mounted."""
    best, best_std = GX, -1.0
    for axis in (GX, GY, GZ):
        s = _std([x[axis] for x in window])
        if s > best_std:
            best, best_std = axis, s
    return best


def dominant_frequency(signal, rate=RATE):
    """Frequency in [FREQ_LO, FREQ_HI] Hz with the most power in a SIGNED signal: a direct DFT of the mean-removed,
    Hann-windowed signal at FREQ_STEP spacing. Use a signed axis, not a magnitude: |x| of an oscillation doubles the
    frequency. About 36 x 200 multiply-adds for a 2 s window, a few ms in CPython."""
    n = len(signal)
    if n < 2:
        return FREQ_LO
    m = _mean(signal)
    w = [(signal[i] - m) * (0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]
    best_f, best_p = FREQ_LO, -1.0
    for k in range(int(round((FREQ_HI - FREQ_LO) / FREQ_STEP)) + 1):
        f = FREQ_LO + k * FREQ_STEP
        re = im = 0.0
        step = 2 * math.pi * f / rate
        for i, x in enumerate(w):
            re += x * math.cos(step * i)
            im -= x * math.sin(step * i)
        p = re * re + im * im
        if p > best_p + 1e-12:
            best_f, best_p = f, p
    return round(best_f, 1)


def compute(window, rate=RATE):
    """-> dict of every feature for one window (floats)."""
    ax = [s[X] for s in window]
    ay = [s[Y] for s in window]
    az = [s[Z] for s in window]
    gx = [s[GX] for s in window]
    gy = [s[GY] for s in window]
    gz = [s[GZ] for s in window]
    mag = [math.sqrt(s[X] ** 2 + s[Y] ** 2 + s[Z] ** 2) for s in window]
    gyro = [math.sqrt(s[GX] ** 2 + s[GY] ** 2 + s[GZ] ** 2) for s in window]
    mag_std = _std(mag)
    gyro_rms = _rms(gyro)
    gyro_dyn = math.sqrt(_std(gx) ** 2 + _std(gy) ** 2 + _std(gz) ** 2)  # rotation without the gyro's static bias
    swing = [s[_swing_axis(window)] for s in window]
    return {
        "peak_x": _percentile([abs(a) for a in ax], 0.95),
        "mean_x": _mean(ax), "mean_y": _mean(ay), "mean_z": _mean(az),
        "std_x": _std(ax), "std_y": _std(ay), "std_z": _std(az),
        "mag_std": mag_std,
        "mag_peak": max(mag),
        "gyro_rms": gyro_rms,
        "gyro_peak": max(gyro),
        "dom_freq": dominant_frequency(swing, rate),  # thigh swing rate: one cycle per stride or pedal revolution
        # rotation per unit of impact: large when the thigh swings without hitting anything, i.e. pedalling.
        # 50 mg of regularisation keeps standing (gyro noise over accel noise) near zero on real sensors.
        "gyro_ratio": gyro_dyn / (1000.0 * mag_std + 50.0),
    }
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_features.py -q`
Expected: `4 passed`. If `dom_freq` is off, remember it is measured on the signed gyro swing axis (0.9 Hz walk, 1.4 run, 1.2 bike in the synth). If BIKE's `gyro_ratio` is not twice the others, check the synth gyro amplitudes before touching the formula.

- [ ] **Step 5: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/features.py P6-Gait-Wearable/tests/synth.py P6-Gait-Wearable/tests/test_features.py && git commit -m "P6-Gait-Wearable: pure-Python window features (peak_x for Model 1; gravity means, spreads, gyro energy, dominant frequency and gyro/impact ratio for Model 2) with synthetic stand/walk/run/bike signals for tests"
```

---

### Task 7: `models.py`: JSON decision trees, majority vote, seed models

**Files:**
- Create: `app/python/models.py`, `models/model1.json`, `models/model2.json`, `tests/test_models.py`

JSON format: `{"features": [...], "classes": [...], "root": node}`; an inner node is `{"f": name, "t": threshold, "l": node, "r": node}` and goes left when `features[f] <= t` (scikit-learn's convention); a leaf is `{"label": "WALK"}`.

- [ ] **Step 1: Failing tests**

`tests/test_models.py`:
```python
import json
from pathlib import Path

import pytest

from features import compute
from models import Tree, Vote
from synth import synth

ROOT = Path(__file__).resolve().parents[1]

SPEC = {
    "features": ["peak_x"],
    "classes": ["RUN", "STAND", "WALK"],
    "root": {"f": "peak_x", "t": 1.15, "l": {"label": "STAND"},
             "r": {"f": "peak_x", "t": 2.2, "l": {"label": "WALK"}, "r": {"label": "RUN"}}},
}


def test_tree_predict_follows_thresholds():
    t = Tree(SPEC)
    assert t.predict({"peak_x": 1.0}) == "STAND"
    assert t.predict({"peak_x": 1.15}) == "STAND"   # <= goes left, like scikit-learn
    assert t.predict({"peak_x": 1.5}) == "WALK"
    assert t.predict({"peak_x": 3.0}) == "RUN"
    assert t.depth() == 2


def test_tree_load_from_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(SPEC))
    assert Tree.load(p).predict({"peak_x": 0.9}) == "STAND"


def test_seed_models_classify_the_synthetic_signals():
    m1 = Tree.load(ROOT / "models" / "model1.json")
    m2 = Tree.load(ROOT / "models" / "model2.json")
    for kind in ("STAND", "WALK", "RUN", "BIKE"):
        f = compute(synth(kind))
        assert m2.predict(f) == kind
        if kind != "BIKE":
            assert m1.predict(f) == kind
    assert m1.predict(compute(synth("BIKE"))) in ("STAND", "WALK")  # the honest limitation of one feature


def test_tree_rejects_a_split_on_an_unlisted_feature():
    bad = {"features": ["peak_x"], "classes": ["STAND", "WALK"],
           "root": {"f": "gyro_rms", "t": 1.0, "l": {"label": "STAND"}, "r": {"label": "WALK"}}}
    with pytest.raises(ValueError):
        Tree(bad)


def test_vote_majority_and_recent_tiebreak():
    v = Vote(3)
    assert v.push("WALK") == "WALK"
    assert v.push("RUN") == "RUN"            # 1-1 tie: the newest wins
    assert v.push("WALK") == "WALK"          # 2-1
    assert v.push("RUN") == "RUN"            # window is RUN WALK RUN
    assert v.push("STAND") == "STAND"        # WALK RUN STAND: three-way tie, newest wins
```

Run: `.venv/bin/python -m pytest tests/test_models.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 2: Implementation**

`app/python/models.py`:
```python
"""Decision trees exported from scikit-learn as JSON, evaluated here without scikit-learn, plus a majority vote."""
import json
from collections import Counter, deque


class Tree:
    """JSON: {"features": [...], "classes": [...], "root": node}; node = {"f", "t", "l", "r"} or {"label"}.
    Goes left when features[f] <= t, exactly as scikit-learn does."""

    def __init__(self, spec):
        self.features = list(spec["features"])
        self.classes = list(spec["classes"])
        self.root = spec["root"]
        self._check(self.root)

    def _check(self, node):
        """A bad export fails at load time, not on the first window."""
        if "label" in node:
            return
        if node["f"] not in self.features:
            raise ValueError("tree splits on %r, which is not in its feature list" % node["f"])
        self._check(node["l"])
        self._check(node["r"])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def predict(self, features):
        node = self.root
        while "label" not in node:
            node = node["l"] if features[node["f"]] <= node["t"] else node["r"]
        return node["label"]

    def depth(self):
        def d(node):
            return 0 if "label" in node else 1 + max(d(node["l"]), d(node["r"]))
        return d(self.root)


class Vote:
    """Majority of the last n labels; ties go to the most recent label among the tied."""

    def __init__(self, n=3):
        self._h = deque(maxlen=n)

    def push(self, label):
        self._h.append(label)
        counts = Counter(self._h)
        top = max(counts.values())
        for lab in reversed(self._h):
            if counts[lab] == top:
                return lab
```

- [ ] **Step 3: Seed models (hand thresholds from the spec table; training replaces these files)**

`models/model1.json`:
```json
{
  "features": ["peak_x"],
  "classes": ["RUN", "STAND", "WALK"],
  "note": "seed thresholds from the spec, replaced by train/train.py",
  "root": {
    "f": "peak_x", "t": 1.15,
    "l": {"label": "STAND"},
    "r": {"f": "peak_x", "t": 2.2, "l": {"label": "WALK"}, "r": {"label": "RUN"}}
  }
}
```

`models/model2.json`:
```json
{
  "features": ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "mag_std", "mag_peak", "gyro_rms", "gyro_peak", "dom_freq", "gyro_ratio"],
  "classes": ["BIKE", "RUN", "STAND", "WALK"],
  "note": "seed thresholds from the spec, replaced by train/train.py",
  "root": {
    "f": "gyro_rms", "t": 40.0,
    "l": {"label": "STAND"},
    "r": {
      "f": "gyro_ratio", "t": 0.6,
      "l": {"f": "mag_peak", "t": 2.2, "l": {"label": "WALK"}, "r": {"label": "RUN"}},
      "r": {"label": "BIKE"}
    }
  }
}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_models.py -q`
Expected: `5 passed`. If the seed Model 2 misfires on a synthetic signal, print `compute(synth(kind))` and move the seed threshold; do not change the features.

- [ ] **Step 5: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/models.py P6-Gait-Wearable/models P6-Gait-Wearable/tests/test_models.py && git commit -m "P6-Gait-Wearable: JSON decision-tree evaluator and 3-window majority vote, with hand-set seed models for the first hardware test"
```

---

### Task 8: `recorder.py`: guided and single-activity labelled CSVs, rolling log

**Files:**
- Create: `app/python/recorder.py`, `tests/test_recorder.py`

- [ ] **Step 1: Failing tests**

`tests/test_recorder.py`:
```python
import csv

from recorder import GAP_S, HEADER, SCRIPT, Recorder


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def read(path):
    with open(path) as f:
        return list(csv.reader(f))


SAMPLE = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_guided_script_prompts_and_finishes_on_the_last_batch(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    assert not r.active and r.prompt() == ""
    r.start_guided(who="rafae")
    assert r.active and r.path.name.endswith("-guided-rafae.csv")
    assert r.prompt() == "GET READY %ds" % GAP_S
    clk.t += GAP_S
    assert r.prompt() == "STAND 20s"
    clk.t += 19.5
    assert r.prompt() == "STAND 1s"
    clk.t += 0.5 + GAP_S                      # into WALK
    assert r.prompt().startswith("WALK")
    total = sum(GAP_S + s for _, s in SCRIPT)
    clk.t = 1000.0 + total + 0.1
    assert r.prompt() == "" and r.active       # script over, but the file waits for the last batch
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)
    assert not r.active


def test_samples_get_the_label_of_their_own_time(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    r.start_single("WALK", seconds=10)
    # a 10-sample batch ending 45 ms after the gap ends: samples at -45..-5 ms are in the gap, 5..45 ms are WALK
    clk.t += GAP_S + 0.045
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)
    rows = read(r.path)
    assert rows[0] == HEADER
    labels = [row[1] for row in rows[1:]]
    assert labels == [""] * 5 + ["WALK"] * 5
    assert float(rows[-1][0]) > float(rows[1][0])
    assert r.rows == 10


def test_status_ahead_of_the_samples_does_not_move_the_labels(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=False)
    r.start_single("WALK", seconds=10)
    clk.t += GAP_S + 0.05                          # the clock is 50 ms into WALK
    assert r.status()["label"] == "WALK"           # the phone page asks first...
    r.on_samples(clk.t - 0.1, [SAMPLE] * 10, rate=100)   # ...then the batch that ended inside the gap arrives
    assert [row[1] for row in read(r.path)[1:]] == [""] * 10
    clk.t = 1000.0 + GAP_S + 10 + 1.0              # a second past the end
    assert r.prompt() == "" and r.status()["active"] and r.active
    r.on_samples(clk.t, [SAMPLE] * 10, rate=100)   # nothing to write, closes the file
    assert not r.active and r.rows == 10 and r.sessions() == [r.path.name]


def test_single_rejects_unknown_label_and_stop_closes(tmp_path):
    r = Recorder(tmp_path, clock=Clock(), rolling=False)
    try:
        r.start_single("FLY")
        assert False, "expected ValueError"
    except ValueError:
        pass
    r.start_single("RUN")
    assert r.active and r.sessions() == []           # the file being written is not offered
    r.stop()
    assert not r.active and r.sessions() == [r.path.name]


def test_rolling_log_runs_only_while_idle(tmp_path):
    clk = Clock()
    r = Recorder(tmp_path, clock=clk, rolling=True)
    r.on_samples(clk.t, [SAMPLE] * 3, rate=100)
    files = list(tmp_path.glob("rolling-*.csv"))
    assert len(files) == 1
    r.start_single("STAND")
    clk.t += 1.0
    r.on_samples(clk.t, [SAMPLE] * 3, rate=100)      # goes to the session, not the rolling log
    r.stop()
    r.close()
    rows = read(files[0])
    assert len(rows) == 4 and rows[1][1] == "-"
    assert r.sessions() == [r.path.name]             # rolling logs are not sessions
```

Run: `.venv/bin/python -m pytest tests/test_recorder.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'recorder'`

- [ ] **Step 2: Implementation**

`app/python/recorder.py`:
```python
"""Labelled CSV recording on the board: the guided script, single-activity takes, and an unlabelled rolling log.

Rows are raw 100 Hz samples, not features, so features can change later without re-recording.
CSV columns: t (seconds since the session started), label ("" during a get-ready gap, "-" in the rolling log),
ax ay az (g), gx gy gz (deg/s). Files: <YYYYmmdd-HHMMSS>-guided[-who].csv, <stamp>-walk[-who].csv, rolling-<stamp>.csv.

Threads: on_samples() runs on the Bridge thread at 10 Hz; start/stop/status come from the web and main threads, so
every public method takes the lock. The script's position is a pure function of time (bisect over the step ends):
asking for the prompt never moves it, and only on_samples() closes a finished session, after writing its last rows.
`t` values are seconds on the same monotonic clock the samples are stamped with. The rolling log runs only while no
session is active, starts a new file every hour and keeps the last two.
"""
import bisect
import csv
import math
import re
import threading
import time
from pathlib import Path

LABELS = ("STAND", "WALK", "RUN", "BIKE")
SCRIPT = [("STAND", 20), ("WALK", 30), ("RUN", 30), ("BIKE", 30)]
GAP_S = 5
HEADER = ["t", "label", "ax", "ay", "az", "gx", "gy", "gz"]
ROLLING_ROTATE_S = 3600
ROLLING_KEEP = 2


def clean(who):
    return re.sub(r"[^A-Za-z0-9_-]", "", who or "")[:16]


def _row(t, label, sample):
    return ["%.3f" % t, label] + ["%.4f" % v for v in sample]


class Recorder:
    def __init__(self, data_dir, clock=time.monotonic, rolling=True):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions = None  # cached listing, cleared when a session starts or ends
        self._labels = []     # one per step, "" = unlabelled gap
        self._ends = []       # cumulative end time of each step (clock seconds)
        self._t0 = 0.0
        self._file = None
        self._writer = None
        self.path = None
        self.rows = 0
        self.who = ""
        self._rolling_on = rolling
        self._rolling = None
        self._rolling_writer = None
        self._rolling_t0 = 0.0

    # --- starting and stopping ---------------------------------------------------------------------------
    def start_guided(self, who=""):
        steps = []
        for label, seconds in SCRIPT:
            steps += [("", GAP_S), (label, seconds)]
        self._start(steps, who, "guided")

    def start_single(self, label, seconds=30, who=""):
        label = label.upper()
        if label not in LABELS:
            raise ValueError("unknown label %r" % label)
        self._start([("", GAP_S), (label, seconds)], who, label.lower())

    def _start(self, steps, who, kind):
        with self._lock:
            self._close_session()
            self.who = clean(who)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = "%s-%s%s" % (stamp, kind, "-" + self.who if self.who else "")
            path = self.dir / (base + ".csv")
            try:
                f = open(path, "x", newline="")
            except FileExistsError:  # two takes within one second
                path = self.dir / (base + "-2.csv")
                f = open(path, "x", newline="")
            self._file, self.path = f, path
            self._writer = csv.writer(f)
            self._writer.writerow(HEADER)
            self._t0 = self._clock()
            self._labels = [label for label, _ in steps]
            self._ends = []
            end = self._t0
            for _, seconds in steps:
                end += seconds
                self._ends.append(end)
            self.rows = 0
            self._sessions = None

    def _close_session(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
        self._labels, self._ends = [], []
        self._sessions = None

    def stop(self):
        with self._lock:
            self._close_session()

    def close(self):
        """Stop everything, including the rolling log (shutdown)."""
        with self._lock:
            self._close_session()
            if self._rolling:
                self._rolling.close()
                self._rolling = None

    @property
    def active(self):
        return self._file is not None

    # --- the script, read-only -----------------------------------------------------------------------------
    def _step(self, now):
        """-> (label, remaining_s) of the step active at `now`, or None when idle or past the end. Never mutates."""
        if not self._ends:
            return None
        i = bisect.bisect_right(self._ends, now)
        if i >= len(self._ends):
            return None
        return self._labels[i], self._ends[i] - now

    def current(self, now=None):
        with self._lock:
            return self._step(self._clock() if now is None else now)

    def prompt(self, now=None):
        return self._prompt(self.current(now))

    @staticmethod
    def _prompt(cur):
        if cur is None:
            return ""
        label, remaining = cur
        return "%s %ds" % (label or "GET READY", int(math.ceil(remaining)))

    # --- data ------------------------------------------------------------------------------------------------
    def on_samples(self, t, samples, rate=100):
        """Write a batch; `t` is the clock time of the LAST sample, earlier ones are spaced 1/rate before it.
        Closes the session once the batch reaches the end of the script; feeds the rolling log while idle."""
        n = len(samples)
        with self._lock:
            if self._file is None:
                self._rolling_write(t, samples, rate)
                return
            rows = []
            for i, s in enumerate(samples):
                ts = t - (n - 1 - i) / rate
                if ts < self._t0:
                    continue  # stamped before the session started
                cur = self._step(ts)
                if cur is None:
                    break  # past the end of the script
                rows.append(_row(ts - self._t0, cur[0], s))
            self._writer.writerows(rows)
            self.rows += len(rows)
            self._file.flush()
            if t >= self._ends[-1]:
                self._close_session()

    def _rolling_write(self, t, samples, rate):
        if not self._rolling_on:
            return
        if self._rolling is None or t - self._rolling_t0 > ROLLING_ROTATE_S:
            self._rotate_rolling(t)
        n = len(samples)
        self._rolling_writer.writerows(_row(t - (n - 1 - i) / rate - self._rolling_t0, "-", s)
                                       for i, s in enumerate(samples))
        self._rolling.flush()

    def _rotate_rolling(self, t):
        if self._rolling:
            self._rolling.close()
        current = self.dir / ("rolling-%s.csv" % time.strftime("%Y%m%d-%H%M%S"))
        self._rolling = open(current, "w", newline="")
        self._rolling_writer = csv.writer(self._rolling)
        self._rolling_writer.writerow(HEADER)
        self._rolling_t0 = t
        # prune by modification time, never the file just opened: the board's clock can come up earlier after a reboot
        others = sorted((p for p in self.dir.glob("rolling-*.csv") if p != current), key=lambda p: p.stat().st_mtime)
        for old in others[:-(ROLLING_KEEP - 1)] if ROLLING_KEEP > 1 else others:
            old.unlink()

    def sessions(self):
        """Finished session files (not the rolling logs, not the one being written), oldest first."""
        with self._lock:
            if self._sessions is None:
                busy = self.path.name if self._file else None
                self._sessions = sorted(p.name for p in self.dir.glob("*.csv")
                                        if not p.name.startswith("rolling-") and p.name != busy)
            return list(self._sessions)

    def status(self):
        with self._lock:
            cur = self._step(self._clock())
            active = self._file is not None
            name = self.path.name if self.path else None
            rows, who = self.rows, self.who
        return {"active": active, "file": name, "rows": rows,
                "label": cur[0] if cur else None, "remaining_s": round(cur[1], 1) if cur else None,
                "prompt": self._prompt(cur), "who": who}
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_recorder.py -q`
Expected: `5 passed`

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/recorder.py P6-Gait-Wearable/tests/test_recorder.py && git commit -m "P6-Gait-Wearable: recorder writes raw 100 Hz samples with labels for the guided STAND/WALK/RUN/BIKE script and single takes, plus an hour-capped unlabelled rolling log"
```

---

### Task 9: `display.py`: TFT payload and waveform bytes

**Files:**
- Create: `app/python/display.py`, `tests/test_display.py`

- [ ] **Step 1: Failing tests**

`tests/test_display.py`:
```python
from display import payload, wave_byte, wave_bytes


def test_payload_keys_and_agreement():
    p = payload("BIKE", "STAND", cad=72, pk=1.13, prompt="BIKE 12s", st="REC")
    assert p == "act=BIKE;m1=STAND;agree=0;cad=72;pk=1.1;prompt=BIKE 12s;st=REC"
    assert "agree=1" in payload("WALK", "WALK", cad=110, pk=1.6)
    assert payload("---", "---", cad=None, pk=None) == "act=---;m1=---;agree=-;cad=--;pk=--;prompt=;st="
    assert "agree=-" in payload("NO IMU", "---", None, None)


def test_payload_is_short_and_sanitised():
    p = payload("NO IMU", "---", None, None, prompt="x;y=z" * 10, st="a=b;c")
    assert ";" not in p.split("prompt=")[1].split(";st=")[0]
    assert len(p) < 120


def test_wave_bytes_map_minus4_to_plus8_g():
    assert wave_byte(-4.0) == 0 and wave_byte(8.0) == 255
    assert wave_byte(1.0) == 106
    assert wave_byte(-9.0) == 0 and wave_byte(20.0) == 255
    assert wave_bytes([-4.0, 1.0, 8.0]) == bytes([0, 106, 255])
```

Run: `.venv/bin/python -m pytest tests/test_display.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'display'`

- [ ] **Step 2: Implementation**

`app/python/display.py`:
```python
"""What the TFT gets: the short key=value payload (provider "tft") and waveform bytes (provider "tfth")."""

WAVE_LO_G, WAVE_HI_G = -4.0, 8.0
ACTS = ("STAND", "WALK", "RUN", "BIKE")


def _safe(text, limit):
    return str(text).replace(";", " ").replace("=", " ")[:limit]


def payload(act, m1, cad, pk, prompt="", st=""):
    """Keys: act (Model 2), m1 (Model 1), agree, cad (per minute), pk (peak g), prompt (recording), st (status)."""
    real = act in ACTS and m1 in ACTS
    pairs = [
        ("act", _safe(act, 7)),
        ("m1", _safe(m1, 7)),
        ("agree", (1 if act == m1 else 0) if real else "-"),
        ("cad", "--" if cad is None else int(cad)),
        ("pk", "--" if pk is None else "%.1f" % pk),
        ("prompt", _safe(prompt, 20)),
        ("st", _safe(st, 7)),
    ]
    return ";".join("%s=%s" % kv for kv in pairs)


def wave_byte(ax):
    """Thigh-axis acceleration in g -> one sweep sample byte (0 = -4 g, 255 = +8 g)."""
    v = int(round((ax - WAVE_LO_G) * 255 / (WAVE_HI_G - WAVE_LO_G)))
    return max(0, min(255, v))


def wave_bytes(ax_values):
    return bytes(wave_byte(a) for a in ax_values)
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_display.py -q`
Expected: `3 passed`

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/display.py P6-Gait-Wearable/tests/test_display.py && git commit -m "P6-Gait-Wearable: TFT payload builder (both verdicts, agreement, cadence, peak g, prompt, status) and -4..+8 g waveform bytes"
```

---

### Task 10: `pipeline.py`: the Gait object that ties it together

**Files:**
- Create: `app/python/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Failing tests**

`tests/test_pipeline.py`:
```python
import struct
from pathlib import Path

from imu import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB
from models import Tree
from pipeline import Gait
from recorder import Recorder
from synth import synth

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def to_batch(seq, samples, resets=0):
    raw = []
    for ax, ay, az, gx, gy, gz in samples:
        raw.append(struct.pack(">6h", int(ax / ACCEL_G_PER_LSB), int(ay / ACCEL_G_PER_LSB), int(az / ACCEL_G_PER_LSB),
                               int(gx / GYRO_DPS_PER_LSB), int(gy / GYRO_DPS_PER_LSB), int(gz / GYRO_DPS_PER_LSB)))
    return bytes([seq & 0xFF, len(samples), resets]) + b"".join(raw)


def make(tmp_path, clk):
    return Gait(Tree.load(ROOT / "models" / "model1.json"), Tree.load(ROOT / "models" / "model2.json"),
                Recorder(tmp_path, clock=clk, rolling=False), clock=clk)


def feed(g, clk, kind, seconds):
    sig = synth(kind, seconds=seconds)
    for i in range(0, len(sig), 10):
        clk.t += 0.1
        g.on_batch(to_batch(i // 10, sig[i:i + 10]))
        g.tick()


def test_no_imu_until_samples_arrive(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    assert g.tick() is True and g.state["act"] == "---"          # nothing heard yet
    clk.t += 0.6
    g.on_batch(to_batch(0, []))                                  # heartbeat, empty: the sketch has no IMU
    g.tick()
    assert g.state["act"] == "NO IMU"
    assert "act=NO IMU" in g.payload()


def test_classifies_walk_then_bike_with_seed_models(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 3.0)
    assert g.state["act"] == "WALK" and g.state["m1"] == "WALK"
    assert 90 < g.state["cad"] < 130                              # 1.8 Hz steps -> 108 per minute
    feed(g, clk, "BIKE", 3.5)
    assert g.state["act"] == "BIKE" and g.state["m1"] in ("STAND", "WALK")
    assert "agree=0" in g.payload()


def test_fifo_reset_clears_the_window(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 2.5)
    assert g.ring.window(2.0) is not None
    clk.t += 0.1
    g.on_batch(to_batch(99, synth("WALK", seconds=0.1), resets=1))   # the sketch restarted the FIFO
    assert g.ring.window(2.0) is None and g.ring.window(0.1) is not None


def test_wave_bytes_are_downsampled_to_50_hz(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    g.on_batch(to_batch(0, synth("STAND", seconds=0.1)))          # 10 samples
    w = g.take_wave()
    assert len(w) == 5 and g.take_wave() == b""


def test_recorder_gets_every_sample_and_prompt_reaches_payload(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    g.recorder.start_single("STAND", seconds=5)
    clk.t += 6.0
    g.on_batch(to_batch(0, synth("STAND", seconds=0.1)))
    assert g.recorder.rows == 10
    assert "prompt=STAND" in g.payload()


def test_snapshot_is_json_friendly(tmp_path):
    import json
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "RUN", 2.5)
    snap = json.loads(json.dumps(g.snapshot()))
    assert snap["act"] == "RUN" and snap["recorder"]["active"] is False
    assert len(snap["wave"]) <= 150 and isinstance(snap["features"]["peak_x"], float)


def test_votes_clear_after_an_imu_gap(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 3.0)
    assert g.state["act"] == "WALK"
    clk.t += 2.0                                   # silence: NO IMU, votes dropped
    g.tick()
    assert g.state["act"] == "NO IMU"
    sig = synth("BIKE", seconds=2.5)
    for i in range(0, len(sig), 10):
        clk.t += 0.1
        g.on_batch(to_batch(100 + i // 10, sig[i:i + 10], resets=1))
        g.tick()
    assert g.state["act"] == "BIKE"               # the first classified window after the gap wins outright


def test_no_imu_after_samples_then_silence(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "STAND", 2.5)
    assert g.state["act"] == "STAND"
    clk.t += 1.6
    g.tick()
    assert g.state["act"] == "NO IMU" and g.state["features"] == {}


def test_on_batch_from_a_thread_while_ticking(tmp_path):
    import threading
    clk = Clock()
    g = make(tmp_path, clk)
    sig = synth("RUN", seconds=4.0)
    errors = []

    def feeder():
        try:
            for i in range(0, len(sig), 10):
                clk.t += 0.1
                g.on_batch(to_batch(i // 10, sig[i:i + 10]))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    th = threading.Thread(target=feeder)
    th.start()
    while th.is_alive():
        g.tick()
        snap = g.snapshot()
        assert snap["act"] in ("---", "NO IMU", "STAND", "WALK", "RUN", "BIKE")
    th.join()
    assert not errors
    g.tick()
    assert g.state["act"] == "RUN"
```

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 2: Implementation**

`app/python/pipeline.py`:
```python
"""Gait: IMU batches in, both verdicts out. Owns the ring buffer, the two trees, the votes and the recorder."""
import threading
import time

from display import payload as tft_payload, wave_byte
from features import X, compute
from imu import RATE, Ring, decode
from models import Vote

WINDOW_S = 2.0
HOP_S = 0.5
IMU_TIMEOUT_S = 1.5
WAVE_MAX = 64  # the sketch's buffer
WAVE_DECIMATE = 2  # 100 Hz samples -> 50 Hz sweep columns
# dom_freq is the thigh swing rate: a stride has two steps, a pedal cycle is one revolution
CADENCE_MULT = {"WALK": 2.0, "RUN": 2.0, "BIKE": 1.0}


class Gait:
    def __init__(self, model1, model2, recorder, rate=RATE, clock=time.monotonic):
        self.m1, self.m2, self.recorder, self.rate, self._clock = model1, model2, recorder, rate, clock
        self.ring = Ring(5.0, rate)
        self.vote1, self.vote2 = Vote(3), Vote(3)
        self._lock = threading.Lock()
        self._wave = bytearray()
        self._last_sample_t = None
        self._resets = None
        self._next_tick = 0.0
        self.state = {"act": "---", "m1": "---", "cad": None, "pk": None, "features": {}, "dropped": 0, "imu": False}

    # --- from the Bridge thread -------------------------------------------------------------------------------
    def on_batch(self, data):
        try:
            seq, resets, samples = decode(data)
        except ValueError:
            return
        now = self._clock()
        with self._lock:
            if resets != self._resets:  # the sensor stream restarted: older samples are not contiguous with these
                self.ring.clear()
                self._resets = resets
            if samples:
                self._last_sample_t = now
                self.ring.push(seq, samples)
                self._wave.extend(wave_byte(s[X]) for s in samples[::WAVE_DECIMATE])
                del self._wave[:-WAVE_MAX]
            elif self._last_sample_t is None:
                self._last_sample_t = -1.0  # heard the sketch, it has no IMU
        if samples:
            self.recorder.on_samples(now, samples, self.rate)  # the recorder has its own lock

    def _set(self, **kw):
        new = dict(self.state)
        new.update(kw)
        self.state = new  # readers see the old frame or the new one, never half of each

    def take_wave(self):
        with self._lock:
            out = bytes(self._wave)
            self._wave.clear()
        return out

    # --- from the main loop -------------------------------------------------------------------------------------
    def tick(self, now=None):
        """Every HOP_S: classify the newest window. Returns True when it ran (the payload may have changed)."""
        now = self._clock() if now is None else now
        if now < self._next_tick:
            return False
        self._next_tick = now + HOP_S
        with self._lock:
            heard = self._last_sample_t is not None
            imu_ok = heard and self._last_sample_t >= 0 and now - self._last_sample_t < IMU_TIMEOUT_S
            win = self.ring.window(WINDOW_S) if imu_ok else None
            dropped = self.ring.dropped
        if not imu_ok:
            self.vote1, self.vote2 = Vote(3), Vote(3)
            self._set(act="NO IMU" if heard else "---", m1="---", cad=None, pk=None, features={}, imu=False)
            return True
        if win is None:
            self.vote1, self.vote2 = Vote(3), Vote(3)
            self._set(act="---", m1="---", cad=None, pk=None, features={}, imu=True)
            return True
        f = compute(win, self.rate)
        a1 = self.vote1.push(self.m1.predict(f))
        a2 = self.vote2.push(self.m2.predict(f))
        cad = None if a2 == "STAND" else int(round(f["dom_freq"] * 60 * CADENCE_MULT.get(a2, 1.0)))
        self._set(act=a2, m1=a1, cad=cad, pk=f["peak_x"], features=f, dropped=dropped, imu=True)
        return True

    def payload(self):
        s = self.state
        return tft_payload(s["act"], s["m1"], s["cad"], s["pk"], prompt=self.recorder.prompt(),
                       st="REC" if self.recorder.active else "")

    def snapshot(self):
        """Everything the phone page shows, as plain JSON-able types."""
        with self._lock:
            recent = self.ring.window(1.5) or []
        s = self.state
        return {"act": s["act"], "m1": s["m1"], "cad": s["cad"], "pk": s["pk"], "imu": s["imu"], "dropped": s["dropped"],
                "features": {k: round(v, 4) for k, v in s["features"].items()},
                "wave": [round(r[X], 3) for r in recent],
                "recorder": self.recorder.status(), "sessions": self.recorder.sessions()}
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: `9 passed`. If `test_classifies_walk_then_bike` fails on cadence, print `g.state["features"]["dom_freq"]`; the WALK synth swings the thigh at 0.9 Hz, so cad should be 108.

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/pipeline.py P6-Gait-Wearable/tests/test_pipeline.py && git commit -m "P6-Gait-Wearable: Gait pipeline: batches into the ring and recorder, 2 s window every 0.5 s through both trees and the vote, NO IMU detection, cadence, waveform bytes, JSON snapshot"
```

---

### Task 11: `web.py` + `page.html`: the phone page

**Files:**
- Create: `app/python/web.py`, `app/python/page.html`, `tests/test_web.py`

Standard library only (`http.server`); live data by Server-Sent Events at 10 Hz, which Safari supports natively.

- [ ] **Step 1: Failing tests**

`tests/test_web.py`:
```python
import http.client
import json
from pathlib import Path

import pytest

from models import Tree
from pipeline import Gait
from recorder import Recorder
from web import serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(tmp_path):
    g = Gait(Tree.load(ROOT / "models" / "model1.json"), Tree.load(ROOT / "models" / "model2.json"),
             Recorder(tmp_path, rolling=False))
    (tmp_path / "20261023-120000-walk.csv").write_text("t,label\n0.0,WALK\n")
    s = serve(g, port=0)
    yield s, g
    s.shutdown()
    s.server_close()


def req(server, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    c.request(method, path, body=body)
    r = c.getresponse()
    data = r.read()
    c.close()
    req.last_headers = dict(r.getheaders())
    return r.status, r.getheader("Content-Type", ""), data


def test_page_and_state(server):
    s, g = server
    status, ctype, body = req(s, "GET", "/")
    assert status == 200 and "text/html" in ctype and b"EventSource" in body
    status, ctype, body = req(s, "GET", "/state")
    assert status == 200 and json.loads(body)["act"] == "---"


def test_recorder_controls(server):
    s, g = server
    status, _, body = req(s, "POST", "/record?label=walk&who=ann")
    assert status == 200 and json.loads(body)["recorder"]["active"] is True
    assert g.recorder.who == "ann" and g.recorder.path.name.endswith("-walk-ann.csv")
    status, _, _ = req(s, "POST", "/record?label=fly")
    assert status == 400
    status, _, body = req(s, "POST", "/stop")
    assert status == 200 and json.loads(body)["recorder"]["active"] is False
    status, _, body = req(s, "POST", "/guided?who=bob")
    assert status == 200 and g.recorder.active and g.recorder.path.name.endswith("-guided-bob.csv")
    g.recorder.stop()


def test_session_download_and_path_safety(server):
    s, g = server
    status, ctype, body = req(s, "GET", "/data/20261023-120000-walk.csv")
    assert status == 200 and "text/csv" in ctype and body.startswith(b"t,label")
    assert "attachment" in req.last_headers["Content-Disposition"]
    status, _, _ = req(s, "GET", "/data/../app.yaml")
    assert status == 404
    status, _, _ = req(s, "GET", "/nope")
    assert status == 404


def test_events_stream_sends_json_frames(server):
    s, g = server
    c = http.client.HTTPConnection("127.0.0.1", s.server_address[1], timeout=3)
    c.request("GET", "/events")
    r = c.getresponse()
    assert r.status == 200 and "text/event-stream" in r.getheader("Content-Type")
    line = r.fp.readline()
    while not line.startswith(b"data: "):
        assert line.strip() in (b"", b"retry: 2000")
        line = r.fp.readline()
    assert json.loads(line[6:])["act"] == "---"
    c.close()
```

Run: `.venv/bin/python -m pytest tests/test_web.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'web'`

- [ ] **Step 2: The page**

`app/python/page.html`:
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gait Wearable</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 16px; font-family: -apple-system, system-ui, sans-serif; background: #000; color: #eee; }
  h1 { font-size: 18px; margin: 0 0 12px; color: #999; font-weight: 500; }
  .verdict { font-size: 56px; font-weight: 800; letter-spacing: 2px; line-height: 1; }
  .STAND { color: #fff; } .WALK { color: #16ec06; } .RUN { color: #ff7800; } .BIKE { color: #0093e7; } .NOIMU { color: #ff0026; }
  .row { display: flex; gap: 16px; margin: 10px 0; flex-wrap: wrap; }
  .card { background: #111; border-radius: 10px; padding: 10px 14px; flex: 1; min-width: 120px; }
  .card small { display: block; color: #888; font-size: 12px; text-transform: uppercase; }
  .card b { font-size: 24px; }
  .agree { color: #16ec06; } .differ { color: #ffde00; }
  canvas { width: 100%; height: 160px; background: #0a0a0a; border-radius: 10px; display: block; }
  button { font-size: 16px; padding: 12px 14px; border: 0; border-radius: 10px; background: #0093e7; color: #fff; margin: 4px 4px 4px 0; }
  button.stop { background: #ff0026; } button.sec { background: #333; }
  input { font-size: 16px; padding: 10px; border-radius: 8px; border: 1px solid #333; background: #111; color: #eee; width: 120px; }
  #prompt { font-size: 28px; color: #fff; min-height: 34px; }
  ul { padding-left: 18px; } a { color: #0093e7; }
  .muted { color: #777; font-size: 13px; }
</style>
</head>
<body>
<h1>Gait Wearable</h1>
<div id="act" class="verdict">---</div>
<div class="muted">Model 2: all six IMU channels</div>
<div class="row">
  <div class="card"><small>Model 1 (peak g only)</small><b id="m1">---</b> <span id="agree"></span></div>
  <div class="card"><small>Cadence</small><b id="cad">--</b></div>
  <div class="card"><small>Peak g</small><b id="pk">--</b></div>
</div>
<canvas id="wave" width="640" height="160"></canvas>
<div class="muted">thigh-axis acceleration, last 1.5 s; line = 1 g</div>
<div id="prompt"></div>
<div class="row">
  <input id="who" placeholder="who (optional)">
  <button onclick="post('/guided')">Start guided session</button>
</div>
<div class="row">
  <button class="sec" onclick="post('/record?label=STAND')">Stand</button>
  <button class="sec" onclick="post('/record?label=WALK')">Walk</button>
  <button class="sec" onclick="post('/record?label=RUN')">Run</button>
  <button class="sec" onclick="post('/record?label=BIKE')">Bike</button>
  <button class="stop" onclick="post('/stop')">Stop</button>
</div>
<div id="rec" class="muted"></div>
<h1>Sessions</h1>
<ul id="sessions"></ul>
<details><summary class="muted">features</summary><pre id="features" class="muted"></pre></details>
<script>
const $ = id => document.getElementById(id);
const canvas = $("wave"), ctx = canvas.getContext("2d");
function post(path) {
  const who = encodeURIComponent($("who").value.trim());
  fetch(path + (path.includes("?") ? "&" : "?") + "who=" + who, {method: "POST"})
    .then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(render)
    .catch(e => { $("rec").textContent = "error: " + e.message; });
}
function sizeCanvas() {
  canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = 160 * devicePixelRatio;
}
sizeCanvas(); window.addEventListener("resize", sizeCanvas);
function drawWave(w) {
  const W = canvas.width, H = canvas.height, lo = -4, hi = 8;
  ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, W, H);
  const y = g => H - 2 - (g - lo) / (hi - lo) * (H - 4);
  ctx.strokeStyle = "#333"; ctx.beginPath(); ctx.moveTo(0, y(1)); ctx.lineTo(W, y(1)); ctx.stroke();
  if (w.length < 2) return;
  ctx.strokeStyle = "#00e6e6"; ctx.lineWidth = 2 * devicePixelRatio; ctx.beginPath();
  w.forEach((g, i) => { const x = i / (w.length - 1) * W; i ? ctx.lineTo(x, y(g)) : ctx.moveTo(x, y(g)); });
  ctx.stroke();
}
function render(s) {
  const act = $("act"); act.textContent = s.act; act.className = "verdict " + s.act.replace(" ", "");
  $("m1").textContent = s.m1;
  const real = !["---", "NO IMU"].includes(s.act) && !["---", "NO IMU"].includes(s.m1);
  const same = s.act === s.m1 || !real;
  $("agree").textContent = !real ? "" : (same ? "agrees" : "differs"); $("agree").className = same ? "agree" : "differ";
  $("cad").textContent = s.cad == null ? "--" : s.cad;
  $("pk").textContent = s.pk == null ? "--" : s.pk.toFixed(1);
  drawWave(s.wave || []);
  $("prompt").textContent = s.recorder.prompt || "";
  $("rec").textContent = s.recorder.active ? `recording ${s.recorder.file} (${s.recorder.rows} rows)` : (s.imu ? "" : "no IMU data");
  const sessionsHtml = (s.sessions || []).slice().reverse().map(n => `<li><a href="/data/${encodeURIComponent(n)}">${n}</a></li>`).join("");
  if ($("sessions").innerHTML !== sessionsHtml) $("sessions").innerHTML = sessionsHtml;   // keep the links stable so a tap lands
  $("features").textContent = Object.entries(s.features || {}).map(([k, v]) => `${k} = ${v}`).join("\n");
}
let es = null;
function connect() {
  if (es) es.close();
  es = new EventSource("/events");
  es.onmessage = e => render(JSON.parse(e.data));
  es.onerror = () => { if (es.readyState === 2) setTimeout(connect, 2000); };
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && (!es || es.readyState === 2)) connect(); });
connect();
</script>
</body>
</html>
```

- [ ] **Step 3: The server**

`app/python/web.py`:
```python
"""The phone page: live verdicts and waveform over Server-Sent Events, recorder controls, CSV downloads. Stdlib only.

GET  /            the page            GET /events  SSE stream of Gait.snapshot() at 10 Hz
GET  /state       one snapshot (JSON) GET /data/<session.csv>  download a recording
POST /guided?who= start the guided script   POST /record?label=WALK&who=   POST /stop
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from recorder import LABELS, clean

PAGE = (Path(__file__).resolve().parent / "page.html").read_bytes()
EVENT_PERIOD_S = 0.1


class Handler(BaseHTTPRequestHandler):
    gait = None  # set by serve()

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif url.path == "/state":
            self._json(self.gait.snapshot())
        elif url.path == "/events":
            self._events()
        elif url.path.startswith("/data/"):
            self._file(url.path[len("/data/"):])
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            who = clean(q.get("who", [""])[0])
            rec = self.gait.recorder
            if url.path == "/guided":
                rec.start_guided(who=who)
            elif url.path == "/record":
                label = q.get("label", [""])[0].upper()
                if label not in LABELS:
                    return self._json({"error": "label must be one of %s" % ", ".join(LABELS)}, 400)
                rec.start_single(label, who=who)
            elif url.path == "/stop":
                rec.stop()
            else:
                return self._send(404, "text/plain", b"not found")
            self._json(self.gait.snapshot())
        except (ValueError, OSError) as e:
            return self._json({"error": str(e)}, 400)

    def _file(self, name):
        path = self.gait.recorder.dir / Path(name).name
        if Path(name).name != name or not path.is_file() or path.suffix != ".csv":
            return self._send(404, "text/plain", b"not found")
        self._send(200, "text/csv", path.read_bytes(),
                   [("Content-Disposition", 'attachment; filename="%s"' % name)])

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.connection.settimeout(5)
            self.wfile.write(b"retry: 2000\n\n")
            while True:
                self.wfile.write(("data: %s\n\n" % json.dumps(self.gait.snapshot())).encode())
                self.wfile.flush()
                time.sleep(EVENT_PERIOD_S)
        except OSError:  # broken pipe, reset, or a socket timeout
            pass


def serve(gait, port=8080, host="0.0.0.0"):
    """Serve on a background thread; returns the server (server_address[1] is the port, shutdown() stops it)."""
    Handler.gait = gait
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_web.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/web.py P6-Gait-Wearable/app/python/page.html P6-Gait-Wearable/tests/test_web.py && git commit -m "P6-Gait-Wearable: phone page (live verdicts, waveform, guided/single recording controls, session downloads) on a stdlib HTTP server with Server-Sent Events"
```

---

### Task 12: `main.py`, deploy and pull scripts, README

**Files:**
- Create: `app/python/main.py`, `scripts/deploy.sh`, `scripts/pull_data.sh`, `README.md`

- [ ] **Step 1: main.py**

`app/python/main.py`:
```python
"""Gait wearable entry point on the UNO Q; App Lab runs this inside the app container (cwd /app/python)."""
import logging
import time
from pathlib import Path

from arduino.app_utils import App, Bridge

from models import Tree
from pipeline import Gait
from recorder import Recorder
from web import serve

ROOT = Path(__file__).resolve().parent.parent  # /app on the board = ~/ArduinoApps/gait-wearable on its Linux host
MODELS, DATA = ROOT / "models", ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(DATA / "app.log")])
log = logging.getLogger("gait")

gait = Gait(Tree.load(MODELS / "model1.json"), Tree.load(MODELS / "model2.json"), Recorder(DATA))
Bridge.provide("imu", gait.on_batch)
serve(gait, port=8080)
log.info("models loaded (depth %d and %d), web on 8080, data in %s", gait.m1.depth(), gait.m2.depth(), DATA)

_last_payload = None
_last_sent = 0.0
_last_log = 0.0


def loop():
    global _last_payload, _last_sent, _last_log
    gait.tick()
    p = gait.payload()
    if p != _last_payload or time.monotonic() - _last_sent > 5:   # resend so a reset MCU gets the screen back
        try:
            Bridge.call("tft", p, timeout=2)
            _last_payload = p
            _last_sent = time.monotonic()
        except Exception as e:  # the sketch may still be compiling or resetting; keep going
            log.warning("tft: %s", e)
    wave = gait.take_wave()
    if wave:
        try:
            Bridge.notify("tfth", wave)
        except Exception as e:
            log.warning("tfth: %s", e)
    now = time.monotonic()
    if now - _last_log > 10:
        _last_log = now
        s = gait.state
        log.info("act=%s m1=%s cad=%s pk=%s dropped=%d imu=%s", s["act"], s["m1"], s["cad"],
                 None if s["pk"] is None else round(s["pk"], 2), s["dropped"], s["imu"])
    time.sleep(0.05)


App.run(user_loop=loop)
```

- [ ] **Step 2: deploy.sh**

`scripts/deploy.sh`:
```bash
#!/usr/bin/env bash
# Deploy the gait wearable to the UNO Q over the USB cable (App Lab's adb) and (re)start it.
#   app/ + models/  ->  ~/ArduinoApps/gait-wearable   (mounted at /app in the app container; first start compiles ~2 min)
# Recordings in ~/ArduinoApps/gait-wearable/data are kept. Any other running user app is stopped (App Lab runs one at a
# time) and this app is made the default so it comes up on power alone.
# Logs:  adb shell tail -f '~/ArduinoApps/gait-wearable/data/app.log'
set -euo pipefail
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
APP_DIR='~/ArduinoApps/gait-wearable'
cd "$(dirname "$0")/.."

remote() { "$ADB" shell "$1"; }

STAGE="$(mktemp -d -t gait-stage)"
TGZ="$(mktemp -t gait-app).tgz"
trap 'rm -rf "$STAGE" "$TGZ"' EXIT
cp -R app/. "$STAGE"/
mkdir -p "$STAGE/models" && cp models/model1.json models/model2.json "$STAGE/models/"
COPYFILE_DISABLE=1 tar -C "$STAGE" --exclude .cache --exclude __pycache__ -czf "$TGZ" .
# adb shell does not carry stdin reliably: push the tarball, then unpack it.
"$ADB" push "$TGZ" /tmp/gait-app.tgz
remote "mkdir -p $APP_DIR && cd $APP_DIR &&
    find . -mindepth 1 -maxdepth 1 ! -name .cache ! -name data -exec rm -rf {} + &&
    tar --warning=no-timestamp -xzf /tmp/gait-app.tgz && rm /tmp/gait-app.tgz"

remote "arduino-app-cli app list | awk '\$1 ~ /^user:/ && \$0 ~ / running / && \$1 != \"user:gait-wearable\" {sub(/^user:/, \"\", \$1); print \$1}' |
  while read -r other; do echo \"stopping \$other\"; arduino-app-cli app stop ~/ArduinoApps/\$other; done"
remote "arduino-app-cli app restart $APP_DIR"
remote "arduino-app-cli properties set default $APP_DIR" || echo "could not set the default app (older arduino-app-cli?)"
echo "deployed; first start compiles the sketch (~2 min). Logs: $ADB shell tail -f '$APP_DIR/data/app.log'"
```

- [ ] **Step 3: pull_data.sh**

`scripts/pull_data.sh`:
```bash
#!/usr/bin/env bash
# Copy the board's recordings into ./data (gitignored) over the USB cable.
set -euo pipefail
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
cd "$(dirname "$0")/.."
mkdir -p data
"$ADB" pull /home/arduino/ArduinoApps/gait-wearable/data/. data/
ls -la data/*.csv | tail -20
```

```bash
chmod +x ~/ME193/P6-Gait-Wearable/scripts/*.sh
```

- [ ] **Step 4: README**

`README.md`:
```markdown
# P6: Gait Wearable (ME 193)

A thigh-worn wearable built from an Arduino UNO Q, an MPU6050 IMU and a 3.2" ILI9341 TFT. It tells standing, walking,
running and biking (including air-biking on your back) apart, live on the screen, with two models side by side:

- **Model 1** sees one number: the peak acceleration along the thigh axis (95th percentile of |ax| in a 2 s window).
- **Model 2** sees all six IMU channels through twelve window features (gravity direction, spreads, gyro energy,
  dominant frequency, rotation-per-impact).

Both are depth-limited decision trees trained with scikit-learn and exported as JSON; the board evaluates them in
pure Python. No neural nets.

## Layout

- `app/` the Arduino App Lab app: `sketch/` (MPU6050 FIFO at 100 Hz, TFT) and `python/` (features, models, recorder,
  phone page on port 8080).
- `models/` the two exported trees. `train/` the scikit-learn training and the figures. `plate/` the laser-cut plate.
- `scripts/` deploy over USB (`deploy.sh`), pull recordings (`pull_data.sh`), local compile (`compile.sh`).
- `docs/superpowers/specs/` the design; `docs/figures/` the Notion figures.

## Workflow

1. `scripts/deploy.sh` with the board on the cable.
2. Open the Personal Hotspot screen on the phone for ~30 s so the board joins, then browse to the board's address on
   port 8080. Tap **Start guided session**: the screen walks you through STAND, WALK, RUN, BIKE.
3. `scripts/pull_data.sh`, then `.venv/bin/python train/train.py` and `.venv/bin/python train/plots.py`.
4. `scripts/deploy.sh` again to ship the trained models.

Tests: `.venv/bin/python -m pytest tests -q`.
```

- [ ] **Step 5: Full test run, then commit**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all green (36 tests so far).

```bash
cd ~/ME193 && git add P6-Gait-Wearable && git commit -m "P6-Gait-Wearable: main.py wires Bridge, pipeline and web; adb deploy (keeps data/, sets default app) and pull_data scripts; README"
```

---

### Task 13: `train/train.py`: windows, two trees, JSON export, confusion matrices

**Files:**
- Create: `train/train.py`, `tests/test_train.py`

- [ ] **Step 1: Failing tests**

`tests/test_train.py`:
```python
import csv
import json
import math
import random

import numpy as np

from features import MODEL1_FEATURES, MODEL2_FEATURES, compute
from models import Tree
from synth import synth
from train import LABELS, dataset, export_tree, fit, matrix, person_of, windows


def write_session(path, kinds, seconds=6.0, who=""):
    random.seed(0)
    rows = []
    t = 0.0
    for kind in kinds:
        for s in synth(kind, seconds=seconds, phase=random.random()):
            rows.append(["%.3f" % t, kind] + ["%.4f" % v for v in s])
            t += 0.01
        for s in synth("STAND", seconds=1.0):          # unlabelled gap between activities
            rows.append(["%.3f" % t, ""] + ["%.4f" % v for v in s])
            t += 0.01
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "label", "ax", "ay", "az", "gx", "gy", "gz"])
        w.writerows(rows)


def test_windows_skip_mixed_and_unlabelled_chunks(tmp_path):
    p = tmp_path / "20261023-120000-guided.csv"
    write_session(p, ["WALK", "BIKE"], seconds=4.0)
    labels = [lab for lab, _ in windows(p)]
    assert set(labels) == {"WALK", "BIKE"}
    # 400 labelled rows at a 50-row hop give (400 - 200) / 50 + 1 = 5 windows; the BIKE block starts at row 500, a hop multiple
    assert labels.count("WALK") == 5 and labels.count("BIKE") == 5   # 4 s at 2 s windows / 0.5 s hop = 5 whole windows


def test_person_of_filename():
    assert person_of("20261023-120000-guided-ann.csv") == "ann"
    assert person_of("20261023-120000-walk.csv") == "me"
    assert person_of("20261023-120000-guided.csv") == "me"
    assert person_of("20261023-120000-walk-ann-2.csv") == "ann"
    assert person_of("20261023-120000-walk-2.csv") == "me"


def test_export_matches_sklearn_and_round_trips(tmp_path):
    for kind in LABELS:
        write_session(tmp_path / ("20261023-1200%02d-%s.csv" % (LABELS.index(kind), kind.lower())), [kind], seconds=8.0)
    feats, y, groups, persons = dataset(sorted(tmp_path.glob("*.csv")))
    assert set(y) == set(LABELS) and len(feats) == len(y) == len(groups) == len(persons)
    jittered = [{k: v * (1 + 1e-3 * ((i % 7) - 3)) for k, v in f.items()} for i, f in enumerate(feats)]
    for names, depth, leaf in ((MODEL1_FEATURES, 3, 1), (MODEL2_FEATURES, 5, 1), (MODEL1_FEATURES, 3, 5), (MODEL2_FEATURES, 5, 5)):
        clf, X = fit(feats, y, names, depth=depth, leaf=leaf)
        tree = Tree(export_tree(clf, names))
        for rows in (feats, jittered):
            ours = [tree.predict(f) for f in rows]
            assert ours == list(clf.predict(matrix(rows, names)))
        spec = json.loads(json.dumps(export_tree(clf, names)))
        assert spec["features"] == names and spec["classes"] == list(clf.classes_)
    # Model 2 must separate all four synthetic activities perfectly; Model 1 is allowed to miss BIKE
    clf2, X2 = fit(feats, y, MODEL2_FEATURES, depth=5, leaf=1)
    assert (clf2.predict(X2) == np.array(y)).all()


def test_windows_do_not_straddle_a_time_gap(tmp_path):
    p = tmp_path / "20261023-120000-walk.csv"
    write_session(p, ["WALK"], seconds=6.0)
    with open(p) as f:
        lines = list(csv.reader(f))
    for row in lines[1 + 300:]:
        row[0] = "%.3f" % (float(row[0]) + 1.0)          # FIFO reset: t jumps 1.0 s after 3 s
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(lines)
    labels = [lab for lab, _ in windows(p)]
    assert len(labels) == 6                                # 2 x ((300 - 200) / 50 + 1)


def test_windows_survive_arrival_jitter(tmp_path):
    p = tmp_path / "20261023-120000-walk.csv"
    write_session(p, ["WALK"], seconds=30.0)
    clean = len(list(windows(p)))
    with open(p) as f:
        lines = list(csv.reader(f))
    for i, row in enumerate(lines[1:]):
        row[0] = "%.3f" % (float(row[0]) + 0.03 * math.sin(i // 10))   # one shift per 10-row batch
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(lines)
    assert clean == 57                                     # (3000 - 200) / 50 + 1
    assert len(list(windows(p))) == clean
```

Run: `.venv/bin/python -m pytest tests/test_train.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 2: Implementation**

`train/train.py`:
```python
"""Train both decision trees from the recorded CSVs and export them as JSON for the board.

  .venv/bin/python train/train.py            # data/*.csv -> models/model1.json, models/model2.json
  .venv/bin/python train/train.py --peer     # also report leave-one-person-out accuracy (who = filename suffix)

Windows are cut exactly as on the board (2 s, 0.5 s hop) with app/python/features.py, so there is no train/live
mismatch. Model 1: depth-3 tree on peak_x only. Model 2: depth-5 tree on MODEL2_FEATURES. Hold-out is by session so a
tree never sees windows from the session it is scored on; the exported trees are refit on everything.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "python"))
from features import MODEL1_FEATURES, MODEL2_FEATURES, RATE, compute  # noqa: E402

LABELS = ["STAND", "WALK", "RUN", "BIKE"]
WINDOW_S, HOP_S = 2.0, 0.5
MODELS = {"model1": (MODEL1_FEATURES, 3, 5), "model2": (MODEL2_FEATURES, 5, 5)}  # name: (features, depth, min leaf)


def load_session(path):
    """-> [(t, label, (ax, ay, az, gx, gy, gz)), ...] in file order."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]), r["label"], tuple(float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz"))))
    return rows


def runs(rows):
    """Split rows at a real break in t (FIFO overflow or re-probe, 0.8 s or more). Batches are stamped on arrival, so ordinary jitter makes t step back a few ms; that must not split."""
    run = []
    for row in rows:
        if run and not (-0.3 <= row[0] - run[-1][0] <= 0.3):
            yield run
            run = []
        run.append(row)
    if run:
        yield run


def windows(path, rate=RATE):
    """Yield (label, features) for every 2 s window (0.5 s hop) whose rows all carry one real label."""
    n, hop = int(WINDOW_S * rate), int(HOP_S * rate)
    for rows in runs(load_session(path)):
        for start in range(0, len(rows) - n + 1, hop):
            chunk = rows[start:start + n]
            labels = {lab for _, lab, _ in chunk}
            if len(labels) != 1:
                continue
            label = labels.pop()
            if label not in LABELS:
                continue
            yield label, compute([s for _, _, s in chunk], rate)


def person_of(name):
    """<stamp>-<kind>[-<who>].csv -> who, or 'me' when there is no suffix."""
    stem = re.sub(r"-2$", "", Path(name).stem)      # a second take in the same second gets a -2 suffix
    parts = stem.split("-", 3)
    return parts[3] if len(parts) >= 4 else "me"


def dataset(paths):
    feats, y, groups, persons = [], [], [], []
    for p in paths:
        for label, f in windows(p):
            feats.append(f)
            y.append(label)
            groups.append(Path(p).name)
            persons.append(person_of(p))
    return feats, y, groups, persons


def matrix(feats, names):
    return np.array([[f[n] for n in names] for f in feats], dtype=float)


def fit(feats, y, names, depth, leaf):
    X = matrix(feats, names)
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X, y)
    return clf, X


def export_tree(clf, names):
    t = clf.tree_

    def node(i):
        if t.children_left[i] == -1:
            return {"label": str(clf.classes_[int(np.argmax(t.value[i]))])}
        return {"f": names[t.feature[i]], "t": float(t.threshold[i]),
                "l": node(t.children_left[i]), "r": node(t.children_right[i])}

    return {"features": list(names), "classes": [str(c) for c in clf.classes_], "root": node(0)}


def heldout_predictions(X, y, groups, depth, leaf):
    """Every window predicted by a tree that never saw its own session (leave-one-session-out), pooled."""
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0)
    return cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut())


def report(title, y_true, y_pred):
    print("\n%s: accuracy %.1f%% on %d windows" % (title, 100 * accuracy_score(y_true, y_pred), len(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("true \\ pred  " + "".join("%7s" % l for l in LABELS))
    for lab, row in zip(LABELS, cm):
        print("%-12s" % lab + "".join("%7d" % v for v in row))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    print("recall  " + "  ".join("%s %.2f" % (l, r) for l, r in zip(LABELS, recall)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--peer", action="store_true", help="also score leave-one-person-out")
    ap.add_argument("--force", action="store_true", help="export even when some activities are missing")
    args = ap.parse_args()
    paths = sorted(p for p in Path(args.data).glob("*.csv") if not p.name.startswith("rolling-"))
    if not paths:
        print("no recordings in", args.data, "(run scripts/pull_data.sh first)")
        return 1
    feats, y, groups, persons = dataset(paths)
    if len(y) == 0:
        print("no labelled windows")
        return 1
    print("%d windows from %d sessions: %s" % (len(y), len(paths), dict(Counter(y))))
    if len(set(y)) < 4:
        print("WARNING: not all four activities are recorded yet")
        if not args.force:
            print("refusing to overwrite the deployed models; pass --force")
            return 1
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for name, (names, depth, leaf) in MODELS.items():
        X = matrix(feats, names)
        y_arr = np.array(y)
        if len(set(groups)) >= 2:
            pred = heldout_predictions(X, y_arr, groups, depth, leaf)
            report("%s leave-one-session-out" % name, y_arr, pred)
            heldout = accuracy_score(y_arr, pred)
        else:
            print("only one session: no held-out estimate")
            heldout = None
        if args.peer and len(set(persons)) >= 2:
            ppl = np.array(persons)
            for who in sorted(set(persons)):
                tr, te = ppl != who, ppl == who
                clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X[tr], y_arr[tr])
                report("%s tested on %s (trained on everyone else)" % (name, who), y_arr[te], clf.predict(X[te]))
        clf, X = fit(feats, y, names, depth, leaf)
        spec = export_tree(clf, names)
        spec["trained_on_windows"] = len(y)
        spec["train_accuracy"] = round(float(accuracy_score(y, clf.predict(X))), 4)
        spec["heldout_accuracy"] = None if heldout is None else round(float(heldout), 4)
        out = Path(args.out) / ("%s.json" % name)
        out.write_text(json.dumps(spec, indent=1))
        print("wrote", out, "(depth %d, %d leaves)" % (clf.get_depth(), clf.get_n_leaves()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_train.py -q`
Expected: `5 passed`

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/train/train.py P6-Gait-Wearable/tests/test_train.py && git commit -m "P6-Gait-Wearable: training: board-identical windows, depth-3 peak-g tree and depth-5 all-features tree, session and per-person hold-outs with confusion matrices, JSON export verified against scikit-learn"
```

---

### Task 14: `train/plots.py`: the three Notion figures

**Files:**
- Create: `train/plots.py`, `tests/test_plots.py`
- Output: `docs/figures/tree_model1.png`, `tree_model2.png`, `confusion.png`, `features.png`

- [ ] **Step 1: Failing test**

`tests/test_plots.py`:
```python
from pathlib import Path

from plots import make_figures
from test_train import write_session
from train import LABELS


def test_make_figures_writes_four_pngs(tmp_path):
    data, out = tmp_path / "data", tmp_path / "fig"
    data.mkdir()
    for i, kind in enumerate(LABELS):
        write_session(data / ("20261023-1200%02d-%s.csv" % (i, kind.lower())), [kind], seconds=6.0)
    written = make_figures(sorted(data.glob("*.csv")), out)
    assert sorted(p.name for p in written) == ["confusion.png", "features.png", "tree_model1.png", "tree_model2.png"]
    assert all(p.stat().st_size > 5000 for p in written)
```

Run: `.venv/bin/python -m pytest tests/test_plots.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'plots'`

- [ ] **Step 2: Implementation**

`train/plots.py`:
```python
"""The Notion figures: both tree diagrams, the two confusion matrices side by side, and the feature-space picture
that shows why Model 1 cannot see biking.   .venv/bin/python train/plots.py   -> docs/figures/*.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "app" / "python"))
from features import MODEL2_FEATURES  # noqa: E402
from train import LABELS, MODELS, dataset, fit, heldout_predictions, matrix  # noqa: E402

COLORS = {"STAND": "#888888", "WALK": "#16a806", "RUN": "#ff7800", "BIKE": "#0093e7"}


def make_figures(paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats, y, groups, _ = dataset(paths)
    y_arr = np.array(y)
    written = []

    # 1 + 2: the trees, fit on everything (what the board runs)
    for name, (names, depth, leaf) in MODELS.items():
        clf, _ = fit(feats, y, names, depth, leaf)
        if name == "model2":
            clf2 = clf
        fig, ax = plt.subplots(figsize=(max(14, 1.3 * clf.get_n_leaves()), 2 + 1.8 * clf.get_depth()))
        plot_tree(clf, feature_names=names, class_names=list(clf.classes_), filled=True, rounded=True,
                  impurity=False, proportion=False, fontsize=7, ax=ax)
        ax.set_title("%s (tree deployed: fit on all sessions)" % name)
        fig.text(0.01, 0.01, "class order in the boxes: " + ", ".join(clf.classes_), fontsize=7)
        p = out_dir / ("tree_%s.png" % name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 3: confusion matrices on held-out sessions (or on everything when there is only one session)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, (names, depth, leaf)) in zip(axes, MODELS.items()):
        X = matrix(feats, names)
        if len(set(groups)) >= 2:
            pred = heldout_predictions(X, y_arr, groups, depth, leaf)
            suffix = "leave-one-session-out"
        else:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X, y_arr)
            pred = clf.predict(X)
            suffix = "(training data, one session only)"
        cm = confusion_matrix(y_arr, pred, labels=LABELS)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, "%d\n%.0f%%" % (cm[i, j], 100 * norm[i, j]), ha="center", va="center",
                        color="black" if norm[i, j] < 0.5 else "white")
        ax.set_xticks(range(4), LABELS)
        ax.set_yticks(range(4), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        acc = np.trace(cm) / max(1, cm.sum())
        ax.set_title("%s: %.0f%% %s" % (name, 100 * acc, suffix))
    fig.tight_layout()
    p = out_dir / "confusion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 4: feature space. Left: peak_x alone (all Model 1 ever sees). Right: peak_x against gyro RMS.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rng = np.random.default_rng(0)
    root = MODEL2_FEATURES[clf2.tree_.feature[0]]
    for lab in LABELS:
        m = y_arr == lab
        px = np.array([f["peak_x"] for f, keep in zip(feats, m) if keep])
        gr = np.array([f[root] for f, keep in zip(feats, m) if keep])
        if len(px) == 0:
            continue
        axes[0].scatter(px, rng.uniform(-0.3, 0.3, len(px)) + LABELS.index(lab), s=12, c=COLORS[lab], label=lab)
        axes[1].scatter(px, gr, s=12, c=COLORS[lab], label=lab)
    axes[0].set_yticks(range(4), LABELS)
    axes[0].set_xlabel("peak thigh-axis acceleration (g)")
    axes[0].set_title("Model 1 sees only peak thigh-axis g")
    axes[1].set_xlabel("peak thigh-axis acceleration (g)")
    axes[1].set_ylabel(root)
    axes[1].set_title("Model 2 also sees %s" % root)
    axes[1].legend()
    fig.tight_layout()
    p = out_dir / "features.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


def main():
    paths = sorted(p for p in (ROOT / "data").glob("*.csv") if not p.name.startswith("rolling-"))
    if not paths:
        print("no recordings in data/ (run scripts/pull_data.sh first)")
        return 1
    for p in make_figures(paths, ROOT / "docs" / "figures"):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_plots.py -q`
Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/train/plots.py P6-Gait-Wearable/tests/test_plots.py && git commit -m "P6-Gait-Wearable: Notion figures: both tree diagrams, held-out confusion matrices, and the peak-g-only vs peak-g-and-gyro feature-space picture"
```

---

### Task 15: Hardware bring-up on the board (user at the bench)

**Files:** none new. This task is run with the user; the board is on the USB-C cable and App Lab is open on the Mac
(it owns the adb server).

- [ ] **Step 1: Wire it**

Ask the user to wire, with the board unpowered:
- TFT: VCC -> 5V, LED -> 3V3, GND -> GND, CS -> D10, DC -> D9, RST -> D8, SDI/MOSI -> D11, SCK -> D13, SDO unconnected.
- MPU6050: VCC -> 3V3, GND -> GND, SDA -> SDA header pin, SCL -> SCL header pin, AD0 -> GND. The breakout's X arrow
  points along the thigh (towards the knee when worn).
Then plug the cable in. Reminder from whoop-hr: a dim power LED is the over-current latch; unplug for a minute.

- [ ] **Step 2: Deploy**

```bash
cd ~/ME193/P6-Gait-Wearable && scripts/deploy.sh
```
Expected: `deployed; first start compiles the sketch (~2 min)`. Then:
```bash
ADB="$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb"; "$ADB" shell "tail -f ~/ArduinoApps/gait-wearable/data/app.log"
```
Expected within ~3 min: `models loaded (depth 2 and 3), web on 8080` then every 10 s `act=... m1=...`. `act=NO IMU`
means the sketch is up but the MPU6050 did not answer: check SDA/SCL and AD0. `act=---` for longer than a minute means
no batches at all: check the sketch compiled (`arduino-app-cli app list` over adb shell; App Lab shows compile errors).

- [ ] **Step 3: Screen check**

The TFT shows the big word (STAND when the plate lies still, its X axis horizontal reads ~0 g so expect `pk` near 0;
hold it X-down to read 1.0), "MODEL 1: PEAK G ONLY" caption, the sweep drawing a flat line at the 1 g mark when X points
down, cadence `--`, peak g. Swing the plate like a thigh (rotate it back and forth about an axis across the plate, a full stride per second) while
bobbing it along X: the sweep shows the bobs, Model 1 flips to WALK or RUN from the peaks, and Model 2 flips too because it
needs the rotation; a straight shake with no rotation moves only Model 1. Pedal it in a smooth circle lying flat and
the big word should go BIKE while Model 1 stays STAND or WALK.

- [ ] **Step 4: Phone page over the cable first (no hotspot needed)**

```bash
"$ADB" forward tcp:8080 tcp:8080 && open http://localhost:8080
```
Expected: the page with the live verdict and waveform; press **Stand**, see `recording ...-stand.csv (n rows)` count up,
press **Stop**, see the session appear in the list and download it.

- [ ] **Step 5: Phone page over the hotspot**

User opens the Personal Hotspot screen on the iPhone and waits ~30 s; `"$ADB" shell "ip -4 addr show wlan0 | grep inet"`
gives the board's address (172.20.10.x on earlier projects). On the phone: `http://<that address>:8080`, add to the home
screen. If the board does not join: `"$ADB" shell "nmcli connection up 'Rafae’s Iphone'"` (curly apostrophe).

- [ ] **Step 6: Fix anything found, then commit**

Typical fixes: `Wire.setClock` unsupported (delete the line), FIFO byte order (if `pk` reads ~0 while X points down,
swap the accel/gyro halves in `decode`), waveform direction. Commit each with a message saying what the hardware showed.

---

### Task 16: Record, train, deploy, tune (user wearing it)

**Files:**
- Modify: `models/model1.json`, `models/model2.json` (regenerated), `app/python/pipeline.py:CADENCE_MULT` if needed
- Output: `docs/figures/*.png`

- [ ] **Step 1: Mount on the thigh**

Plate cut (Task 2), boards on standoffs, two straps threaded in through the left slits, across under the boards, out the
right slits, buckled behind. Snug. Pocket bank, USB-C up the leg, tape loop on the strap.

- [ ] **Step 2: Three guided sessions**

From the phone page, **Start guided session** three times, with a minute between: the screen says GET READY, then STAND,
WALK, RUN, BIKE with the countdown. Bike = lying on the back, air-pedalling; in later sessions also a real bike if
available. Vary pace within reason (slow and brisk walk, jog and faster run). Then one 30 s **Stand** single take while
fidgeting a little and one while standing on one leg, so STAND is not only perfect stillness.

- [ ] **Step 3: Pull and train**

```bash
cd ~/ME193/P6-Gait-Wearable && scripts/pull_data.sh && .venv/bin/python train/train.py && .venv/bin/python train/plots.py
```
Expected: `N windows from M sessions: {...}`, two confusion blocks, `wrote models/model1.json`, `wrote models/model2.json`,
four figures. Acceptance: Model 2 held-out accuracy >= 95 % on all four; Model 1 >= 90 % on STAND/WALK/RUN with BIKE
landing mostly in STAND or WALK (that is the story, not a bug). If Model 2 confuses WALK and RUN, record more at the
boundary pace. If STAND leaks into BIKE, the fidget takes are missing.

- [ ] **Step 4: Cadence sanity**

Open `data/<guided>.csv` for a WALK stretch and count steps in the video or by ear for 10 s; compare with the page's
cadence. If the page reads half, set `CADENCE_MULT["WALK"] = 2.0` (and RUN likewise); same check for BIKE revolutions.
Change the test expectation in `tests/test_pipeline.py` to match (the WALK synth has |a| at 1.8 Hz).

- [ ] **Step 5: Deploy the trained models and verify live**

```bash
scripts/deploy.sh
```
Wear it again: all four activities, watch the big word and the Model 1 line. BIKE should show the amber DIFFERS.

- [ ] **Step 6: Peer rehearsal**

Have one friend wear it (guided session with their name in **who**), then
`.venv/bin/python train/train.py --peer` to see how the models trained on you score on them. If Model 2 drops below
~90 % on the friend, retrain including their session and deploy.

- [ ] **Step 7: Commit models and figures**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/models P6-Gait-Wearable/docs/figures P6-Gait-Wearable/app/python/pipeline.py P6-Gait-Wearable/tests && git commit -m "P6-Gait-Wearable: trained models from N guided sessions (Model 2 X % held out, Model 1 Y %), Notion figures, cadence multiplier from the recordings"
```
(fill N, X and Y from train.py's output).

---

### Task 17: Notion page drafts

**Files:**
- Create: `docs/notion.md`

- [ ] **Step 1: Ask the user for the four reflection questions** (they are in the assignment, not in this plan).

- [ ] **Step 2: Write the drafts**

`docs/notion.md` with these sections, each ready to paste (the Notion markdown-paste trick works in the user's Chrome):

1. **Intro paragraph** (~120 words): what it is, where it is worn, the two models, no neural nets. Pull the actual
   held-out numbers from `train/train.py`'s output.
2. **How it works**: 5 bullets following the data: MPU6050 FIFO at 100 Hz -> Bridge -> 2 s windows -> one feature dict
   -> two JSON trees -> TFT and phone page. One sentence per bullet.
3. **Which model is better and why** (~150 words): Model 1 separates stand/walk/run by impact height but a pedalling
   thigh never hits anything, so its peak g sits between standing and walking and it guesses; Model 2 sees the gyro
   (big, periodic) and the gravity direction (lying down), which is exactly where biking lives. Reference
   `features.png` and the confusion matrices. Note the honest cost: Model 2 needs 12 numbers and a deeper tree, Model 1
   is one threshold pair a person could set by hand.
4. **Figures**: file list with one-line captions: `tree_model1.png`, `tree_model2.png`, `confusion.png`, `features.png`.
5. **Demo video**: a shot list for the user to film: strap on, stand, walk, jog, lie down and air-bike, with the screen
   and the phone page both in frame; end on the amber DIFFERS during biking.
6. **GitHub**: `https://github.com/rafaeshafi/Me193/tree/main/P6-Gait-Wearable` (push when the user asks).
7. **Reflection answers**: the four questions as headings with drafted answers grounded in what happened in Tasks 15
   and 16 (what broke, what surprised, what the peers said).

- [ ] **Step 3: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/docs/notion.md && git commit -m "P6-Gait-Wearable: Notion page drafts (intro, how it works, which model is better and why, figure captions, video shot list, reflection answers)"
```

Then offer to push `main` and to paste the page into Notion through the user's Chrome.

---

## Revision 2 (2026-10-23): the TFT is dropped, the web page is the display

The user scrapped the TFT after Task 14 (it took both power-header pins and most of the plate). Tasks 4 and 9 are
superseded, Tasks 2, 3, 10, 12 and 15 change as below; the code blocks earlier in this file are re-synced to the
files after each change, so later tasks read the file tree, not history.

### Task 18: Sketch without the TFT
- Delete `app/sketch/src/{ili9341,tft_gfx,tft_ui}.{h,cpp}`. `sketch.ino` keeps the MPU6050 FIFO drain, the 3-byte
  batch header, the reset-counter flush and the 500 ms heartbeat; remove the `tft`/`tfth` providers, the buffers,
  the mutex, `Bridge.provide` calls and the copy-out in `loop()`. `delay(5)` stays. Compile with `scripts/compile.sh`.

### Task 19: Python without the TFT, longer trace on the page
- Delete `app/python/display.py` and `tests/test_display.py`.
- `pipeline.py`: remove `payload()`, `take_wave()`, `_wave`, `WAVE_DECIMATE`, `WAVE_MAX` and the `display` import;
  `snapshot()["wave"]` becomes the last 4 s of thigh-axis accel (`ring.window(4.0)`), rounded to 3 dp; add
  `snapshot()["agree"]`: `1`/`0` when both labels are real activities, else `None`. Tests: drop the wave-bytes test,
  add `snapshot()["agree"]` assertions, keep the rest.
- `main.py`: remove the Bridge `tft`/`tfth` calls and `_last_payload`/`_last_sent`; the loop ticks, logs every 10 s,
  sleeps 50 ms.
- `page.html`: use `s.agree` (null hides the word), label the trace "last 4 s", make the verdict block fill the
  width on a phone and the page readable on a laptop (max-width 720 px, centred). Nothing else.
- `web.py` unchanged.

### Task 20: Plate without the TFT
- `plate/plate.py`: remove every TFT parameter and hole; `height = 109`; UNO Q (turned, USB-C down) at the top under
  the top margin; MPU6050 right of it at `mpu_y = 14`; straps at the UNO Q's hole-free middle (`uno_strap_y = 40.6`)
  and in the band below the UNO Q (centre 4 + 12.7 mm below its bottom edge). `check()` unchanged in spirit (edge,
  slits, strap bands). Re-run, confirm no PROBLEM lines, DXF entities, preview.

### Task 15 (revised): bench
- Wire only the MPU6050: VCC 3V3, GND, SDA, SCL header pins, AD0 to GND. Deploy; the log must show `act=STAND`
  with the board still (after ~2.5 s), WALK/RUN when swung like a thigh and bobbed, BIKE when pedalled flat; the
  page over `adb forward tcp:8080 tcp:8080` shows the same with the trace.
