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
from seed_models import SEED1, SEED2
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
    return Gait(Tree(SEED1), Tree(SEED2),
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
    assert g.snapshot()["agree"] is None


def test_classifies_walk_then_bike_with_seed_models(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 3.0)
    assert g.state["act"] == "WALK" and g.state["m1"] == "WALK"
    assert 90 < g.state["cad"] < 130                              # 1.8 Hz steps -> 108 per minute
    feed(g, clk, "BIKE", 3.5)
    assert g.state["act"] == "BIKE" and g.state["m1"] in ("STAND", "WALK")
    assert g.snapshot()["agree"] == 0


def test_fifo_reset_clears_the_window(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "WALK", 2.5)
    assert g.ring.window(2.0) is not None
    clk.t += 0.1
    g.on_batch(to_batch(99, synth("WALK", seconds=0.1), resets=1))   # the sketch restarted the FIFO
    assert g.ring.window(2.0) is None and g.ring.window(0.1) is not None


def test_recorder_gets_every_sample_and_prompt_reaches_snapshot(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    g.recorder.start_single("STAND", seconds=5)
    clk.t += 6.0
    g.on_batch(to_batch(0, synth("STAND", seconds=0.1)))
    assert g.recorder.rows == 10
    assert g.snapshot()["recorder"]["prompt"].startswith("STAND")


def test_snapshot_is_json_friendly(tmp_path):
    import json
    clk = Clock()
    g = make(tmp_path, clk)
    feed(g, clk, "RUN", 2.5)
    snap = json.loads(json.dumps(g.snapshot()))
    assert snap["act"] == "RUN" and snap["recorder"]["active"] is False
    assert len(snap["wave"]) <= 400 and snap["agree"] in (0, 1) and isinstance(snap["features"]["peak_x"], float)


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


def test_race_runs_on_the_gait(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    assert g.snapshot()["race"]["phase"] == "idle" and g.snapshot()["speed_kmh"] == 0.0
    g.race.start(100, "ann")
    feed(g, clk, "RUN", 6.0)                       # 2.8 Hz synth run: ~168 spm x stride(3.5 g) 1.85 m = ~5 m/s
    snap = g.snapshot()
    assert snap["speed_kmh"] > 10
    assert snap["race"]["phase"] in ("running", "finished") and snap["race"]["dist"] > 0
    assert len(snap["race"]["ghosts"]) == 3
    g.race.abort()
    assert g.snapshot()["race"]["phase"] == "idle"


def test_race_distance_does_not_grow_while_standing(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    g.race.start(100, "ann")
    feed(g, clk, "STAND", 6.0)
    snap = g.snapshot()
    assert snap["race"]["phase"] == "running" and snap["race"]["dist"] == 0.0 and snap["speed_kmh"] == 0.0
```

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 2: Implementation**

`app/python/pipeline.py`:
```python
"""Gait: IMU batches in, both verdicts out. Owns the ring buffer, the two trees, the votes and the recorder."""
import threading
import time
from pathlib import Path

from features import X, compute
from imu import RATE, Ring, decode
from models import Vote
from race import Race, ResultStore, SpeedModel

WINDOW_S = 2.0
HOP_S = 0.5
IMU_TIMEOUT_S = 1.5
# dom_freq is the thigh swing rate: a stride has two steps, a pedal cycle is one revolution
CADENCE_MULT = {"WALK": 2.0, "RUN": 2.0, "BIKE": 1.0}


class Gait:
    def __init__(self, model1, model2, recorder, rate=RATE, clock=time.monotonic, races_path=None):
        self.m1, self.m2, self.recorder, self.rate, self._clock = model1, model2, recorder, rate, clock
        self.speed = SpeedModel()
        self.race = Race(ResultStore(races_path or (Path(recorder.dir) / "races.json")), clock=clock)
        self.ring = Ring(5.0, rate)
        self.vote1, self.vote2 = Vote(3), Vote(3)
        self._lock = threading.Lock()
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
            elif self._last_sample_t is None:
                self._last_sample_t = -1.0  # heard the sketch, it has no IMU
        if samples:
            self.recorder.on_samples(now, samples, self.rate)  # the recorder has its own lock

    def _set(self, **kw):
        new = dict(self.state)
        new.update(kw)
        self.state = new  # readers see the old frame or the new one, never half of each

    # --- from the main loop -------------------------------------------------------------------------------------
    def tick(self, now=None):
        """Advance the race every call; every HOP_S also classify the newest window. Returns True when it classified."""
        now = self._clock() if now is None else now
        self.race.tick(self.speed.speed, now)
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
            self.speed.update(self.state["act"], self.state["cad"], self.state["pk"], now)
            return True
        if win is None:
            self.vote1, self.vote2 = Vote(3), Vote(3)
            self._set(act="---", m1="---", cad=None, pk=None, features={}, imu=True)
            self.speed.update(self.state["act"], self.state["cad"], self.state["pk"], now)
            return True
        f = compute(win, self.rate)
        a1 = self.vote1.push(self.m1.predict(f))
        a2 = self.vote2.push(self.m2.predict(f))
        cad = None if a2 == "STAND" else int(round(f["dom_freq"] * 60 * CADENCE_MULT.get(a2, 1.0)))
        self._set(act=a2, m1=a1, cad=cad, pk=f["peak_x"], features=f, dropped=dropped, imu=True)
        self.speed.update(self.state["act"], self.state["cad"], self.state["pk"], now)
        return True

    def snapshot(self):
        """Everything the phone page shows, as plain JSON-able types."""
        with self._lock:
            recent = self.ring.recent(4.0)
        s = self.state
        real = ("---", "NO IMU")
        agree = None if s["act"] in real or s["m1"] in real else int(s["act"] == s["m1"])
        return {"act": s["act"], "m1": s["m1"], "cad": s["cad"], "pk": s["pk"], "imu": s["imu"], "agree": agree, "dropped": s["dropped"],
                "speed_kmh": round(self.speed.speed * 3.6, 1), "race": self.race.snapshot(),
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
from seed_models import SEED1, SEED2
from pipeline import Gait
from recorder import Recorder
from web import serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(tmp_path):
    g = Gait(Tree(SEED1), Tree(SEED2),
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


def test_race_routes(server):
    s, g = server
    status, _, body = req(s, "POST", "/race/start?distance=400&who=ann")
    snap = json.loads(body)
    assert status == 200 and snap["race"]["phase"] == "countdown" and snap["race"]["distance"] == 400
    assert g.race.who == "ann"
    status, _, body = req(s, "POST", "/race/start?distance=123")
    assert status == 400
    status, _, body = req(s, "POST", "/race/abort")
    assert status == 200 and json.loads(body)["race"]["phase"] == "idle"
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
  main { max-width: 720px; margin: 0 auto; }
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
  .race { background: #0d0d0d; border: 1px solid #222; border-radius: 12px; padding: 12px 14px; margin-bottom: 18px; }
  .track { height: 360px; width: 100%; display: block; background: #1f2430; border-radius: 8px; margin: 10px 0 4px; touch-action: manipulation; }
  .countdown { font-size: 64px; font-weight: 800; text-align: center; color: #fff; min-height: 0; }
  .result { display: none; margin-top: 8px; }
  .board { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }
  .board td, .board th { padding: 4px 6px; text-align: left; border-bottom: 1px solid #222; color: #ccc; }
  .board th { color: #888; font-weight: 500; text-transform: uppercase; font-size: 11px; }
  .dist.on { background: #0093e7; }
</style>
</head>
<body>
<main>
<h1>Gait Wearable</h1>
<section id="race" class="race">
  <div class="row" style="align-items:center">
    <h1 style="margin:0">Race</h1>
    <span class="muted">run on the spot</span>
    <span style="flex:1"></span>
    <button class="sec dist" data-d="100">100 m</button>
    <button class="sec dist" data-d="400">400 m</button>
    <button class="sec dist" data-d="1000">1 km</button>
    <button id="raceStart" onclick="raceStart()">Start</button>
    <button id="raceAbort" class="stop" onclick="post('/race/abort')">Abort</button>
  </div>
  <canvas id="raceCanvas" class="track"></canvas>
  <div class="muted">speed = cadence x stride, stride = 0.6 m + 0.5 m per g of impact above 1 g; WALK capped at 7 km/h, STAND and BIKE do not move</div>
  <div class="row">
    <div class="card"><small>Speed</small><b id="rSpeed">0.0</b> km/h</div>
    <div class="card"><small>Distance</small><b id="rDist">0</b> m</div>
    <div class="card"><small>Time</small><b id="rTime">0.0</b> s</div>
  </div>
  <div id="rResult" class="card result"></div>
  <table id="rBoard" class="board"></table>
</section>
<div id="act" class="verdict">---</div>
<div class="muted">Model 2: all six IMU channels</div>
<div class="row">
  <div class="card"><small>Model 1 (peak g only)</small><b id="m1">---</b> <span id="agree"></span></div>
  <div class="card"><small>Cadence</small><b id="cad">--</b></div>
  <div class="card"><small>Peak g</small><b id="pk">--</b></div>
</div>
<canvas id="wave" width="640" height="160"></canvas>
<div class="muted">thigh-axis acceleration, last 4 s; line = 1 g</div>
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
let raceDistance = 400;
document.querySelectorAll(".dist").forEach(b => b.onclick = () => { raceDistance = +b.dataset.d; document.querySelectorAll(".dist").forEach(x => x.classList.toggle("on", x === b)); });
document.querySelector('.dist[data-d="400"]').classList.add("on");
function raceStart() { post("/race/start?distance=" + raceDistance); }
const ghostColor = {Jogger: "#16ec06", Runner: "#ffde00", Sprinter: "#ff0026", "Your best": "#0093e7"};
function renderRace(r, speedKmh) {
  $("rSpeed").textContent = speedKmh.toFixed(1);
  $("rDist").textContent = Math.round(r.dist);
  $("rTime").textContent = r.elapsed.toFixed(1);
  $("raceStart").style.display = r.phase === "idle" || r.phase === "finished" ? "" : "none";
  $("raceAbort").style.display = r.phase === "countdown" || r.phase === "running" ? "" : "none";
  raceSnap = r; raceSnapAt = performance.now();
  if (r.result && r.phase === "finished") {
    $("rResult").style.display = "block";
    $("rResult").innerHTML = `<small>${r.who}, ${r.distance} m</small><b>${ordinal(r.result.place || 0)} of ${r.field || 0}</b> &nbsp; ${r.result.time.toFixed(2)} s &nbsp; ${r.result.avg_kmh.toFixed(1)} km/h` + (r.result.record ? `<div style="color:#ffde00;font-weight:700">NEW RECORD: you are the Champion ghost now</div>` : (r.champion ? `<div class="muted">Champion ${r.champion.who || ""} holds ${r.champion.time.toFixed(2)} s</div>` : ""));
  } else { $("rResult").style.display = "none"; }
  const rows = r.leaderboard.map((e, i) => `<tr><td>${i + 1}</td><td>${e.who}</td><td>${e.time.toFixed(2)} s</td><td>${e.avg_kmh.toFixed(1)} km/h</td><td class="muted">${e.when}</td></tr>`).join("");
  const board = rows ? `<tr><th>#</th><th>who</th><th>${r.distance} m</th><th>avg</th><th>when</th></tr>` + rows : "";
  if ($("rBoard").dataset.html !== board) { $("rBoard").innerHTML = board; $("rBoard").dataset.html = board; }
}
const ordinal = n => n + ({1: "st", 2: "nd", 3: "rd"}[n] || "th");
// ---- London 2012 (SEGA) look: chase camera behind the runners on the stadium track ----
// World in metres: x east, y up, z north. The oval is run anticlockwise, as on a real track: the bends turn left, lane 1 is on the inside (left).
// u = metres along the inside kerb (bend radius R), w = metres outward from it. A pinhole chase camera sits 6 m behind and
// 2.2 m above you. Textures (crowd, wraps, boards, roof, flags, grain), the sky and the HUD plates are pre-rendered; per frame
// the stands are texture-mapped segment by segment far to near, the ground is batched into a few fills, runners are 3D skeletons.
const rc = $("raceCanvas"), rctx = rc.getContext("2d");
let raceSnap = null, raceSnapAt = 0;
const SCENE_H = 360, STR = 85.3319, R = 36.5, BND = Math.PI * R, LOOP = 2 * STR + 2 * BND, LW = 1.22, U0 = STR - 50, NEAR = 0.3;
const SEG = 5, NSEG = Math.ceil(LOOP / SEG), SCR0 = 26, SUN = [-0.42, 0.53, 0.74], TAU = Math.PI * 2;
const LANE = {YOU: 3, Jogger: 1, Runner: 2, Sprinter: 5, Champion: 4}, BIB = {YOU: 1, Jogger: 2, Runner: 3, Sprinter: 4, Champion: 5};
const LOOK = {   // singlet, trim, shorts, skin, hair, shoes, hair style (0 short, 1 bun, 2 ponytail)
  YOU: ["#f2c230", "#0b7a55", "#123a8a", "#e3b48d", "#7a4a22", "#1c1c1c", 2], Jogger: ["#1b8f3a", "#ffffff", "#f0f0f0", "#6f4428", "#151515", "#f05a28", 0],
  Runner: ["#f4f4f4", "#c8102e", "#16224f", "#f3cfb0", "#b5651d", "#2b59c3", 1], Sprinter: ["#1f3f8f", "#d22630", "#c81f2b", "#8d5a3b", "#121212", "#c6ff00", 0], Champion: ["#c3338b", "#ffffff", "#ffffff", "#c99268", "#f1d38a", "#ff3d9a", 2]};
const FONT = "-apple-system, system-ui, Helvetica, Arial, sans-serif", COND = '"Avenir Next Condensed", "Arial Narrow", "Roboto Condensed", "Helvetica Neue", Arial, sans-serif';
// track geometry: trackPt(u, w) -> TX, TZ (position) and HX, HZ (unit heading)
let TX = 0, TZ = 0, HX = 0, HZ = 1;
function trackPt(u, w) { u -= Math.floor(u / LOOP) * LOOP; let a;
  if (u < STR) { HX = 0; HZ = 1; TX = 0; TZ = u; } else if (u < STR + BND) { a = (u - STR) / R; HX = Math.sin(a); HZ = Math.cos(a); TX = R - R * HZ; TZ = STR + R * HX; } else if (u < 2 * STR + BND) { HX = 0; HZ = -1; TX = 2 * R; TZ = 2 * STR + BND - u; }
  else { a = (u - 2 * STR - BND) / R; HX = -Math.sin(a); HZ = -Math.cos(a); TX = R - R * HZ; TZ = R * HX; } TX = w * HZ - TX; TZ += w * HX; HX = -HX;   // mirror x so the oval runs anticlockwise; outward is to the right of the heading
}
function arcS(u, w) {   // arc length along the line w (texture coordinate along the stands)
  const f = (R + w) / R; u -= Math.floor(u / LOOP) * LOOP; return u < STR ? u : u < STR + BND ? STR + (u - STR) * f : u < 2 * STR + BND ? u - BND + BND * f : 2 * STR + BND * f + (u - 2 * STR - BND) * f;
}
// camera and projection
let CX = 0, CY = 2.2, CZ = 0, RX = 1, RZ = 0, FX = 0, FY = 0, FZ = 1, UX = 0, UY = 1, UZ = 0, CHX = 0, CHZ = 1, FOC = 400, W2 = 0, HC = 0, HOR = 0, YAW = 0, VW = 0, VH = SCENE_H, CAMU = 0, DPR = 1;
let PX = 0, PY = 0, PZ = 0;
function proj(x, y, z) { const dx = x - CX, dy = y - CY, dz = z - CZ; PZ = dx * FX + dy * FY + dz * FZ; if (PZ < NEAR) return false; const k = FOC / PZ; PX = W2 + (dx * RX + dz * RZ) * k; PY = HC - (dx * UX + dy * UY + dz * UZ) * k; return true; }
function setCam(u, w, bob, sway) { trackPt(u - 6, w + sway); CX = TX; CZ = TZ; CY = 2.2 + bob; CAMU = u - 6;
  trackPt(u + 12, w); let dx = TX - CX, dz = TZ - CZ; const d = Math.hypot(dx, dz); dx /= d; dz /= d; const p = Math.atan2(CY - 0.3, d), cp = Math.cos(p), sp = Math.sin(p);
  RX = dz; RZ = -dx; FX = dx * cp; FY = -sp; FZ = dz * cp; UX = dx * sp; UY = cp; UZ = dz * sp; CHX = dx; CHZ = dz; YAW = Math.atan2(dx, dz); HOR = HC - FOC * sp / cp;
}
const CV = new Float64Array(30), CW = new Float64Array(60), QP = new Float64Array(30), PQ = new Float64Array(12);
function polyW(n) {   // QP holds n world points: clip against the near plane, project, append to the current path
  let m = 0, x0 = 1e9, x1 = -1e9, y0 = 1e9; for (let i = 0; i < n; i++) { const dx = QP[3 * i] - CX, dy = QP[3 * i + 1] - CY, dz = QP[3 * i + 2] - CZ; CV[3 * i] = dx * RX + dz * RZ; CV[3 * i + 1] = dx * UX + dy * UY + dz * UZ; CV[3 * i + 2] = dx * FX + dy * FY + dz * FZ; }
  for (let i = 0; i < n; i++) { const j = i + 1 === n ? 0 : i + 1, za = CV[3 * i + 2], zb = CV[3 * j + 2]; if (za >= NEAR) { CW[2 * m] = W2 + CV[3 * i] * FOC / za; CW[2 * m + 1] = HC - CV[3 * i + 1] * FOC / za; m++; }
    if ((za >= NEAR) !== (zb >= NEAR)) { const t = (NEAR - za) / (zb - za); CW[2 * m] = W2 + (CV[3 * i] + t * (CV[3 * j] - CV[3 * i])) * FOC / NEAR; CW[2 * m + 1] = HC - (CV[3 * i + 1] + t * (CV[3 * j + 1] - CV[3 * i + 1])) * FOC / NEAR; m++; } } if (m < 3) return;
  for (let i = 0; i < m; i++) { const x = CW[2 * i], y = CW[2 * i + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; } if (x1 < 0 || x0 > VW || y0 > VH) return;
  rctx.moveTo(CW[0], CW[1]); for (let i = 1; i < m; i++) rctx.lineTo(CW[2 * i], CW[2 * i + 1]); rctx.closePath();
}
function setQ(i, u, w, y) { trackPt(u, w); QP[3 * i] = TX; QP[3 * i + 1] = y; QP[3 * i + 2] = TZ; }
function quadUW(ua, ub, wa0, wb0, wa1, wb1, y) { setQ(0, ua, wa0, y); setQ(1, ub, wa1, y); setQ(2, ub, wb1, y); setQ(3, ua, wb0, y); polyW(4); }
function quadV(ua, ub, w, y0, y1) { setQ(0, ua, w, y0); setQ(1, ub, w, y0); setQ(2, ub, w, y1); setQ(3, ua, w, y1); polyW(4); }
const stepAt = d => d < 24 ? 1.5 : d < 90 ? 3 : 6;
function strip(wa, wb, y) { for (let d = -3; d < LOOP - 8;) { const s = stepAt(d); quadUW(CAMU + d, CAMU + d + s, wa, wb, wa, wb, y); d += s; } }
function halfW(u, w) { trackPt(u, w); return Math.max(0.03, 0.55 * Math.max(1, (TX - CX) * FX - CY * FY + (TZ - CZ) * FZ) / FOC); }
function lineStrip(w) { for (let d = -3; d < LOOP - 8;) { const s = stepAt(d), u = CAMU + d, h0 = halfW(u, w), h1 = halfW(u + s, w); quadUW(u, u + s, w - h0, w + h0, w - h1, w + h1, 0); d += s; } }
function pq(i, u, w, y) { trackPt(u, w); if (!proj(TX, y, TZ)) return false; PQ[3 * i] = PX; PQ[3 * i + 1] = PY; PQ[3 * i + 2] = PZ; return true; }
function affine(ctx, i, j, k, tw, th) {   // map text space [0,tw]x[0,th] onto the screen points PQ[i] (top left), PQ[j] (top right), PQ[k] (bottom left)
  const x0 = PQ[3 * i], y0 = PQ[3 * i + 1]; ctx.setTransform(DPR * (PQ[3 * j] - x0) / tw, DPR * (PQ[3 * j + 1] - y0) / tw, DPR * (PQ[3 * k] - x0) / th, DPR * (PQ[3 * k + 1] - y0) / th, DPR * x0, DPR * y0);
}
// pre-rendered textures (seeded so the crowd is the same every load)
function cnv(w, h) { const c = document.createElement("canvas"); c.width = Math.max(1, Math.ceil(w)); c.height = Math.max(1, Math.ceil(h)); return c; }
let seed = 11; const rand = () => (seed = seed * 16807 % 2147483647) / 2147483647, pick = a => a[(rand() * a.length) | 0];
function texSet(w, h, period, draw) {   // a texture and two mip levels as repeating patterns
  const cs = [cnv(w, h), cnv(w / 2, h / 2), cnv(w / 4, h / 4)]; draw(cs[0].getContext("2d"), w, h); for (let i = 1; i < 3; i++) cs[i].getContext("2d").drawImage(cs[i - 1], 0, 0, cs[i].width, cs[i].height);
  return {pat: cs.map(c => rctx.createPattern(c, "repeat")), w: cs.map(c => c.width), h: cs.map(c => c.height), period, cv: cs, org: 0};
}
function rings(g, x, y, r, cols, lw) {   // the five rings, interlaced: top row blue, black, red; bottom row yellow, green
  const C = [[x - 2.2 * r, y], [x - 1.1 * r, y + r], [x, y], [x + 1.1 * r, y + r], [x + 2.2 * r, y]];
  const ring = (i, a0, a1) => { g.beginPath(); g.arc(C[i][0], C[i][1], r, a0, a1); if (cols[5]) { g.lineWidth = lw + 1.8; g.strokeStyle = cols[5]; g.stroke(); } g.lineWidth = lw; g.strokeStyle = cols[i]; g.stroke(); }; for (const i of [1, 3, 0, 2, 4]) ring(i, 0, TAU);
  for (const i of [1, 3]) ring(i, -1.95, -1.2);   // each lower ring crosses over its neighbours at the top crossings, under them at the sides
}
const SHIRT = ["#f4f4f4", "#e6e6e6", "#cfcfd4", "#c8102e", "#1f3f8f", "#1c1c1c", "#ffd100", "#7a2a8c", "#e46aa6", "#5aa0d8", "#2e8b57", "#ff7f00", "#8a8a8a", "#b22234", "#f4f4f4"];
const SKIN = ["#f1c7a5", "#e0ac83", "#c68642", "#8d5524", "#5c3a1e", "#f3d2b8", "#d9a37a"];
const crowd = upper => (g, w, h) => { const rows = 24, rh = h / rows;
  g.fillStyle = upper ? "#3a3446" : "#4a4456"; g.fillRect(0, 0, w, h); for (let r = 0; r < rows; r++) { const y = r * rh; g.fillStyle = "rgba(160,160,180,0.22)"; g.fillRect(0, y + rh - 1, w, 1); for (let x = 15 + rand() * 3; x < w - 4; x += 4.4 + rand() * 1.8) {
      if ((!upper && r > 9 && r < 17 && x > 236 && x < 292) || rand() < 0.05) continue; const sw = 3.4 + rand() * 1.2; g.fillStyle = pick(SHIRT); g.fillRect(x, y + rh * 0.44, sw, rh * 0.56);
      g.fillStyle = pick(SKIN); g.beginPath(); g.arc(x + sw / 2, y + rh * 0.3, 1.55, 0, TAU); g.fill(); if (rand() < 0.035) { g.fillStyle = rand() < 0.5 ? "#c8102e" : "#ffffff"; g.fillRect(x - 1, y - rh * 0.35, 5.5, 3.5); } }
    g.fillStyle = `rgba(14,8,26,${((upper ? 0.45 : 0.3) * (1 - r / rows)).toFixed(3)})`; g.fillRect(0, y, w, rh);   // roof shade on the back rows
  } g.fillStyle = "#6f6b78"; g.fillRect(0, 0, 13, h); g.fillStyle = "#4f4b58"; for (let y = 0; y < h; y += rh / 2) g.fillRect(0, y, 13, 1.2);   // aisle steps
  if (!upper) { g.fillStyle = "#120e18"; g.fillRect(240, 10 * rh, 48, 7 * rh); g.fillStyle = "#9a96a2"; g.fillRect(236, 10 * rh - 2.5, 56, 2.5); }   // vomitory
};
function wrapDraw(g, w, h) { const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, "#5a2386"); gr.addColorStop(0.55, "#71309c"); gr.addColorStop(1, "#4a1a70"); g.fillStyle = gr; g.fillRect(0, 0, w, h);
  g.fillStyle = "#e4007c"; g.fillRect(0, h * 0.86, w, h * 0.07); g.fillStyle = "rgba(255,255,255,0.25)"; g.fillRect(0, h * 0.06, w, 1.5);
  g.fillStyle = "#fff"; g.font = `600 ${h * 0.4}px ${FONT}`; g.textBaseline = "middle"; g.fillText("London 2012", 26, h * 0.47); g.fillText("London 2012", 286, h * 0.47);
  rings(g, 225, h * 0.38, h * 0.11, ["#fff", "#fff", "#fff", "#fff", "#fff"], 1.6); rings(g, 485, h * 0.38, h * 0.11, ["#fff", "#fff", "#fff", "#fff", "#fff"], 1.6);
}
function boardsDraw(g, w, h) { const B = [["#e4007c", "#fff", "london 2012"], ["#5a2386", "#fff", ""], ["#2d1e6b", "#ff9f1c", "London 2012"], ["#f2f2f2", "#5a2386", "london"]];
  for (let i = 0; i < 4; i++) { const x = i * w / 4; g.fillStyle = B[i][0]; g.fillRect(x, 0, w / 4, h); g.fillStyle = "rgba(0,0,0,0.45)"; g.fillRect(x, 0, 2, h); g.fillStyle = B[i][1]; g.font = `bold ${h * 0.5}px ${FONT}`; g.textBaseline = "middle"; g.textAlign = "center";
    if (B[i][2]) g.fillText(B[i][2], x + w / 8, h * 0.53); else rings(g, x + w / 8, h * 0.42, h * 0.16, ["#fff", "#fff", "#fff", "#fff", "#fff"], 2); }
}
function roofDraw(g, w, h) { g.fillStyle = "#e9ecf0"; g.fillRect(0, 0, w, h);
  for (let x = 0; x < w; x += 64) { const gr = g.createLinearGradient(x, 0, x + 64, 0); gr.addColorStop(0, "#f6f7f9"); gr.addColorStop(0.55, "#d9dde3"); gr.addColorStop(1, "#c3c8d0"); g.fillStyle = gr; g.fillRect(x, 0, 64, h); g.fillStyle = "#8b919b"; g.fillRect(x, 0, 2, h); }
  g.fillStyle = "rgba(120,126,136,0.7)"; g.fillRect(0, 18, w, 1); g.fillRect(0, 42, w, 1); const sh = g.createLinearGradient(0, 0, 0, h); sh.addColorStop(0, "rgba(40,40,60,0)"); sh.addColorStop(1, "rgba(40,40,60,0.22)"); g.fillStyle = sh; g.fillRect(0, 0, w, h);
}
const FLAGS = [["v", "#0055a4", "#fff", "#ef4135"], ["v", "#009246", "#fff", "#ce2b37"], ["h", "#000", "#dd0000", "#ffce00"], ["h", "#ae1c28", "#fff", "#21468b"], ["v", "#169b62", "#fff", "#ff883e"],
  ["jp"], ["h", "#fff", "#0039a6", "#d52b1e"], ["v", "#002b7f", "#fcd116", "#ce1126"], ["n", "#006aa7", "#fecc00"], ["n", "#c8102e", "#fff"], ["h", "#00966e", "#fff", "#d62612"], ["uk"], ["v", "#000", "#ffd520", "#ef3340"], ["h", "#009c3b", "#ffdf00", "#009c3b"]];
function flagsDraw(g, w, h) { g.fillStyle = "#c9ced6"; g.fillRect(0, 2, w, 1.5);
  for (let i = 0; i < 8; i++) { const f = FLAGS[(i * 5 + 3) % FLAGS.length], x = i * 64 + 10, fw = 43, fh = 28, y = 4; g.fillStyle = "#fff"; g.fillRect(x, y, fw, fh);
    if (f[0] === "v") for (let k = 0; k < 3; k++) { g.fillStyle = f[k + 1]; g.fillRect(x + k * fw / 3, y, fw / 3 + 0.5, fh); } else if (f[0] === "h") for (let k = 0; k < 3; k++) { g.fillStyle = f[k + 1]; g.fillRect(x, y + k * fh / 3, fw, fh / 3 + 0.5); }
    else if (f[0] === "jp") { g.fillStyle = "#bc002d"; g.beginPath(); g.arc(x + fw / 2, y + fh / 2, 7, 0, TAU); g.fill(); } else if (f[0] === "n") { g.fillStyle = f[1]; g.fillRect(x, y, fw, fh); g.fillStyle = f[2]; g.fillRect(x + 12, y, 6, fh); g.fillRect(x, y + 11, fw, 6); }
    else { g.fillStyle = "#012169"; g.fillRect(x, y, fw, fh); g.strokeStyle = "#fff"; g.lineWidth = 5; g.beginPath(); g.moveTo(x, y); g.lineTo(x + fw, y + fh); g.moveTo(x + fw, y); g.lineTo(x, y + fh); g.stroke();
      g.strokeStyle = "#c8102e"; g.lineWidth = 2; g.stroke(); g.fillStyle = "#fff"; g.fillRect(x + fw / 2 - 5, y, 10, fh); g.fillRect(x, y + fh / 2 - 5, fw, 10); g.fillStyle = "#c8102e"; g.fillRect(x + fw / 2 - 3, y, 6, fh); g.fillRect(x, y + fh / 2 - 3, fw, 6); }
    for (let k = 0; k < 4; k++) { g.fillStyle = k % 2 ? "rgba(0,0,0,0.13)" : "rgba(255,255,255,0.12)"; g.fillRect(x + k * fw / 4, y, fw / 4, fh); }   // the cloth ripples
  }
}
function grainDraw(g, w, h) { for (let i = 0; i < 2600; i++) { g.fillStyle = rand() < 0.5 ? `rgba(255,214,190,${(0.08 + rand() * 0.14).toFixed(2)})` : `rgba(60,8,8,${(0.1 + rand() * 0.2).toFixed(2)})`; g.fillRect((rand() * w) | 0, (rand() * h) | 0, 1 + (rand() < 0.2), 1); }
}
const TEX = {lo: texSet(512, 192, 32, crowd(false)), up: texSet(512, 192, 32, crowd(true)), wrap: texSet(512, 64, 32, wrapDraw), boards: texSet(512, 32, 30, boardsDraw),
  roof: texSet(256, 64, 16, roofDraw), flags: texSet(512, 40, 24, flagsDraw)};
const GRAIN = cnv(128, 128); grainDraw(GRAIN.getContext("2d"), 128, 128); const GRAINP = rctx.createPattern(GRAIN, "repeat");
const SCR = cnv(256, 128), SCRG = SCR.getContext("2d");
TEX.scr = {pat: [null, null, null], w: [256, 256, 256], h: [128, 128, 128], period: arcS((SCR0 + 2) * SEG, 32.5) - arcS(SCR0 * SEG, 32.5), org: SCR0 * SEG, end: (SCR0 + 2) * SEG};
const HAZE = [];   // overlay per stand segment: [shade 0..4][haze 0..10], one rgba string each, built once
for (let s = 0; s <= 4; s++) { HAZE.push([]); for (let k = 0; k <= 10; k++) { const h = k * 0.05, as = s * 0.09, a = h + as * (1 - h); HAZE[s].push(a < 0.005 ? "" : `rgba(${Math.round((h * 196 + as * (1 - h) * 22) / a)},${Math.round((h * 208 + as * (1 - h) * 14) / a)},${Math.round((h * 226 + as * (1 - h) * 44) / a)},${a.toFixed(3)})`); } }
// a stand band: the quad between two lines along the ring (edge A = texture top), texture-mapped with an affine pattern transform
const MAT = new DOMMatrix(), GM = new DOMMatrix();
let LODZ = 60;
function band(T, u0, u1, wA, yA, wB, yB, depth) { if (!pq(0, u0, wA, yA) || !pq(1, u1, wA, yA) || !pq(2, u1, wB, yB) || !pq(3, u0, wB, yB)) return;
  if (typeof T !== "string" && (depth || 0) < 3 && Math.max(PQ[2], PQ[5], PQ[8], PQ[11]) > 1.3 * Math.min(PQ[2], PQ[5], PQ[8], PQ[11])) {   // deep band: split so the affine map stays close to perspective
    const um = (u0 + u1) / 2, d = (depth || 0) + 1; band(T, u0, um, wA, yA, wB, yB, d); band(T, um, u1, wA, yA, wB, yB, d); return; } const ax = PQ[0], ay = PQ[1], bx = PQ[3], by = PQ[4], cx = PQ[6], cy = PQ[7], dx = PQ[9], dy = PQ[10];
  if ((ax < 0 && bx < 0 && cx < 0 && dx < 0) || (ax > VW && bx > VW && cx > VW && dx > VW)) return; const ctx = rctx; if (typeof T === "string") { if (!T) return; ctx.fillStyle = T; } else {
    const lod = PQ[2] < LODZ ? 0 : PQ[2] < 2 * LODZ ? 1 : 2, pat = T.pat[lod]; if (!pat) return; const tw = T.w[lod], th = T.h[lod], k = tw / T.period, sa = arcS(u0, wA), sb = u1 >= LOOP ? 2 * STR + 2 * BND * (R + wA) / R : arcS(u1, wA);
    const s0 = T.org ? (arcS(T.end, wA) - sa) * k : ((-sa * k) % tw + tw) % tw, s1 = s0 - (sb - sa) * k;   // u runs right to left on the stands seen from inside
    MAT.a = (bx - ax) / (s1 - s0); MAT.b = (by - ay) / (s1 - s0); MAT.c = (dx - ax) / th; MAT.d = (dy - ay) / th; MAT.e = ax - MAT.a * s0; MAT.f = ay - MAT.b * s0; pat.setTransform(MAT); ctx.fillStyle = pat; }
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(cx, cy); ctx.lineTo(dx, dy); ctx.closePath(); ctx.fill();
}
function truss(u) {   // a triangular floodlight tower on the roof edge, leaning in over the track
  if (!pq(0, u - 1.6, 31, 31) || !pq(1, u + 1.6, 31, 31) || !pq(2, u, 33.5, 31) || !pq(3, u, 26, 45)) return; const ctx = rctx, k = FOC / PQ[11]; ctx.strokeStyle = "#e3e7ee"; ctx.lineWidth = Math.max(0.6, 0.35 * k); ctx.beginPath();
  for (const i of [0, 1, 2]) { ctx.moveTo(PQ[3 * i], PQ[3 * i + 1]); ctx.lineTo(PQ[9], PQ[10]); } for (let s = 0; s < 5; s++) { const t0 = s / 5, t1 = (s + 0.5) / 5, L = (i, t) => PQ[3 * i] + (PQ[9] - PQ[3 * i]) * t, M = (i, t) => PQ[3 * i + 1] + (PQ[10] - PQ[3 * i + 1]) * t;
    ctx.moveTo(L(0, t0), M(0, t0)); ctx.lineTo(L(1, t1), M(1, t1)); ctx.lineTo(L(2, t0), M(2, t0)); ctx.lineTo(L(0, t1), M(0, t1)); } ctx.stroke(); if (!pq(0, u - 2.6, 25.5, 46.2) || !pq(1, u + 2.6, 25.5, 46.2) || !pq(2, u + 2.6, 25.5, 44.2)) return;
  ctx.fillStyle = "#59606c"; ctx.beginPath(); ctx.moveTo(PQ[0], PQ[1]); ctx.lineTo(PQ[3], PQ[4]); ctx.lineTo(PQ[6], PQ[7]); ctx.lineTo(PQ[6] + PQ[0] - PQ[3], PQ[7] + PQ[1] - PQ[4]); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#fffbe8"; for (let i = 1; i < 6; i++) { const t = i / 6; ctx.fillRect(PQ[0] + (PQ[3] - PQ[0]) * t - 0.5, (PQ[1] + PQ[7]) / 2 - 0.5, Math.max(1, 0.5 * k), Math.max(1, 0.5 * k)); }
}
function drawSeg(i, hz) { const u0 = i * SEG, u1 = Math.min(LOOP, u0 + SEG), scr = i === SCR0 || i === SCR0 + 1;
  band(TEX.lo, u0, u1, 36, 13, 12.5, 2.4); band("#1c1724", u0, u1, 33, 14, 36, 13); band(TEX.wrap, u0, u1, 12.5, 2.4, 12.5, 0);
  band(TEX.boards, u0, u1, 10.5, 0.95, 10.5, 0); band(TEX.wrap, u0, u1, 33, 17.2, 33, 14); band("#221d2b", u0, u1, 58, 32.6, 58, 26); band(TEX.up, u0, u1, 58, 26, 33, 17.2);
  if (scr) { band("#0b0b10", u0 - (i === SCR0 ? 0.25 : 0), u1 + (i > SCR0 ? 0.25 : 0), 32.4, 21.1, 32.4, 13.9); band(TEX.scr, u0, u1, 32.3, 20.7, 32.3, 14.3); }
  band(TEX.flags, u0, u1, 31.3, 29.9, 31.3, 27.6); band(TEX.roof, u0, u1, 31, 30, 56, 32.6); band("#f4f6f8", u0, u1, 31, 31.2, 31, 30); band(hz, u0, u1, 31, 31.2, 10.5, 0); if (i % 6 === 2) truss(u0 + SEG / 2);
}
// runners: 3D skeletons. Joints: 0 head, 1 neck, 2/3 shoulders, 4/5 elbows, 6/7 hands, 8/9 hips, 10/11 knees, 12/13 ankles, 14/15 toes, 16 pelvis,
// 17/18 waist, 19/20 armpits, 21 crown, 22/23 hip outline, 24/25 singlet straps, 26/27 racer-back cut, 28/29 crotch, 30-33 bib corners, 34/35 calves
const JN = 36, JL = new Float64Array(JN * 3), TMP = new Float64Array(6), RB = Array.from({length: 6}, () => ({J: new Float64Array(JN * 3), S: new Float64Array(JN * 3)}));
function sj(i, x, y, z) { JL[3 * i] = x; JL[3 * i + 1] = y; JL[3 * i + 2] = z; }
function pose(p, k) {   // running cycle at stride phase p with effort k (0 = standing still, 1 = sprinting)
  const hipA = 0.5 * k, knA = 1.5 * k, armA = 0.08 + 0.55 * k, lean = 0.03 + 0.16 * k, sp = Math.sin(p), flex = 0.3 + 1.05 * Math.min(1, k * 3) + 0.2 * k; let low = 9; for (let s = 0; s < 2; s++) {
    const q = p + s * Math.PI, sd = s ? 1 : -1, h = hipA * Math.sin(q) + 0.05 * k, hz = 0.025 * sd * sp * k; const kn = 0.05 + knA * Math.pow(Math.max(0, Math.cos(q + 0.35)), 1.6) + 0.2 * k * Math.max(0, -Math.cos(q));
    const kz = hz + 0.46 * Math.sin(h), ky = 0.93 - 0.46 * Math.cos(h), a = h - kn, az = kz + 0.44 * Math.sin(a), ay = ky - 0.44 * Math.cos(a), pf = 0.12 + 0.6 * k * Math.max(0, -Math.sin(q + 0.6)); const tz = az + 0.16 * Math.cos(a - pf), ty = ay + 0.16 * Math.sin(a - pf);
    sj(8 + s, sd * 0.1, 0.93, hz); sj(10 + s, sd * 0.097, ky, kz); sj(12 + s, sd * 0.083, ay, az); sj(14 + s, sd * 0.088, ty, tz); sj(34 + s, sd * 0.094, ky + 0.36 * (ay - ky), kz + 0.36 * (az - kz) - 0.025 * Math.cos(a)); low = Math.min(low, ay - 0.075, ty - 0.025); }
  const ny = 0.93 + 0.55 * Math.cos(lean), nz = 0.55 * Math.sin(lean), shy = ny - 0.035, wy = 0.93 + 0.27 * Math.cos(lean), wz = 0.27 * Math.sin(lean) - 0.08; sj(16, 0, 0.93, -0.02); sj(1, 0, ny, nz - 0.02); sj(0, 0, ny + 0.145, nz + 0.03); sj(21, 0, ny + 0.245, nz - 0.03);
  for (let s = 0; s < 2; s++) { const sd = s ? 1 : -1, tw = 0.05 * k * sp * sd, a = armA * sp * sd + 0.12 + lean * 0.5, ez = nz + tw + 0.28 * Math.sin(a), ey = shy - 0.28 * Math.cos(a), fa = a + flex;
    sj(2 + s, sd * 0.2, shy, nz - 0.03 + tw); sj(4 + s, sd * 0.235, ey, ez); sj(6 + s, sd * (0.2 - 0.06 * Math.max(0, Math.sin(a))), ey - 0.26 * Math.cos(fa), ez + 0.26 * Math.sin(fa));
    sj(17 + s, sd * 0.128, wy, wz); sj(19 + s, sd * 0.175, shy - 0.15, nz * 0.75 - 0.09 + tw * 0.5); sj(22 + s, sd * 0.152, 0.98, -0.08); sj(24 + s, sd * 0.07, ny - 0.005, nz - 0.1); sj(26 + s, sd * 0.1, shy - 0.1, nz * 0.85 - 0.1); sj(28 + s, sd * 0.03, 0.83, -0.02); }
  sj(30, -0.085, wy + 0.2, wz - 0.03 + nz * 0.25); sj(31, 0.085, wy + 0.2, wz - 0.03 + nz * 0.25); sj(32, 0.085, wy + 0.03, wz - 0.03); sj(33, -0.085, wy + 0.03, wz - 0.03);
  const dy = -low + 0.05 * k * Math.max(0, -Math.cos(2 * p));   // feet on the ground in stance, airborne in flight
  for (let i = 0; i < JN; i++) JL[3 * i + 1] += dy;
}
function placeRunner(b, u, w) {   // local joints -> screen (J: x, y, px per metre) and their sun shadows on the ground (S)
  trackPt(u, w); const bx = TX, bz = TZ, hx = HX, hz = HZ, kx = SUN[0] / SUN[1], kz = SUN[2] / SUN[1]; for (let i = 0; i < JN; i++) { const lx = JL[3 * i], ly = JL[3 * i + 1], lz = JL[3 * i + 2], x = bx + lx * hz + lz * hx, z = bz - lx * hx + lz * hz;
    if (!proj(x, ly, z)) return false; b.J[3 * i] = PX; b.J[3 * i + 1] = PY; b.J[3 * i + 2] = FOC / PZ; if (!proj(x - ly * kx, 0, z - ly * kz)) return false; b.S[3 * i] = PX; b.S[3 * i + 1] = PY; b.S[3 * i + 2] = FOC / PZ; } return true;
}
function cap(g, B, i, j, ri, rj, o) {   // tapered capsule between joints i and j (radii in metres), shifted left by o radii, appended to the path
  const r1 = ri * B[3 * i + 2], r2 = rj * B[3 * j + 2], x1 = B[3 * i] - o * r1, y1 = B[3 * i + 1], x2 = B[3 * j] - o * r2, y2 = B[3 * j + 1]; const a = Math.atan2(y2 - y1, x2 - x1), c = -Math.sin(a), s = Math.cos(a);
  g.moveTo(x1 + c * r1, y1 + s * r1); g.lineTo(x2 + c * r2, y2 + s * r2); g.arc(x2, y2, r2, a + 1.5708, a - 1.5708, true); g.lineTo(x1 - c * r1, y1 - s * r1); g.arc(x1, y1, r1, a - 1.5708, a + 1.5708, true); g.closePath();
}
function polyL(g, B, ids) { g.moveTo(B[3 * ids[0]], B[3 * ids[0] + 1]); for (let i = 1; i < ids.length; i++) g.lineTo(B[3 * ids[i]], B[3 * ids[i] + 1]); }
function polyJ(g, B, ids) { g.moveTo(B[3 * ids[0]], B[3 * ids[0] + 1]); for (let i = 1; i < ids.length; i++) g.lineTo(B[3 * ids[i]], B[3 * ids[i] + 1]); g.closePath(); }
const LIT = "rgba(255,236,212,0.24)", DARK = "rgba(20,8,16,0.13)";
function limb(B, segs, cols) {   // segs: [i, j, ri, rj] per segment; each gets its colour, a rim light on the sun side and shade on the other
  const g = rctx; for (let n = 0; n < segs.length; n += 4) { g.fillStyle = cols[n >> 2]; g.beginPath(); cap(g, B, segs[n], segs[n + 1], segs[n + 2], segs[n + 3], 0); g.fill(); }
  g.fillStyle = DARK; g.beginPath(); for (let n = 0; n < segs.length; n += 4) cap(g, B, segs[n], segs[n + 1], segs[n + 2] * 0.55, segs[n + 3] * 0.55, -0.5); g.fill();
  g.fillStyle = LIT; g.beginPath(); for (let n = 0; n < segs.length; n += 4) cap(g, B, segs[n], segs[n + 1], segs[n + 2] * 0.45, segs[n + 3] * 0.45, 0.55); g.fill();
}
const LEG = [[8, 10, 0.094, 0.06, 10, 34, 0.052, 0.062, 34, 12, 0.06, 0.036, 12, 14, 0.05, 0.043], [9, 11, 0.094, 0.06, 11, 35, 0.052, 0.062, 35, 13, 0.06, 0.036, 13, 15, 0.05, 0.043]];
const ARM = [[2, 4, 0.06, 0.044, 4, 6, 0.044, 0.033], [3, 5, 0.06, 0.044, 5, 7, 0.044, 0.033]];
const TORSO = [24, 25, 27, 20, 18, 23, 22, 17, 19, 26], BACK = [2, 1, 3, 20, 19], SEAT = [22, 23, 29, 28];
function drawRunner(e) { const B = e.b.J, L = LOOK[e.name] || LOOK.Jogger, g = rctx, sk = L[3];
  const leg = s => { limb(B, LEG[s], [sk, sk, sk, L[5]]); for (let c = 0; c < 3; c++) { TMP[c] = B[3 * (8 + s) + c]; TMP[3 + c] = TMP[c] + (B[3 * (10 + s) + c] - TMP[c]) * 0.24; }
    g.fillStyle = L[2]; g.beginPath(); cap(g, TMP, 0, 1, 0.102, 0.098, 0); g.fill(); g.fillStyle = s ? DARK : LIT; g.fill(); };
  const arm = s => { limb(B, ARM[s], [sk, sk]); g.fillStyle = sk; g.beginPath(); g.arc(B[3 * (6 + s)], B[3 * (6 + s) + 1], 0.045 * B[3 * (6 + s) + 2], 0, TAU); g.fill(); }; const torso = () => {
    g.fillStyle = sk; g.beginPath(); polyJ(g, B, BACK); g.fill(); g.beginPath(); cap(g, B, 2, 19, 0.06, 0.05, 0); cap(g, B, 3, 20, 0.06, 0.05, 0); g.fill();
    g.fillStyle = L[0]; g.beginPath(); polyJ(g, B, TORSO); g.fill(); g.fillStyle = DARK; g.beginPath(); polyJ(g, B, [1, 25, 27, 20, 18, 23, 16]); g.fill(); g.beginPath(); polyJ(g, B, [25, 27, 20, 18, 23]); g.fill();
    g.fillStyle = LIT; g.beginPath(); polyJ(g, B, [24, 26, 19, 17, 22]); g.fill(); g.strokeStyle = L[1]; g.lineWidth = Math.max(0.8, 0.025 * B[50]); g.beginPath(); polyL(g, B, [24, 26, 19]); polyL(g, B, [25, 27, 20]); g.stroke();
    g.fillStyle = L[2]; g.beginPath(); polyJ(g, B, SEAT); g.fill(); g.fillStyle = DARK; g.beginPath(); polyJ(g, B, [16, 23, 29]); g.fill(); g.fillStyle = LIT; g.beginPath(); polyJ(g, B, [22, 16, 28]); g.fill(); const bw = Math.hypot(B[93] - B[90], B[94] - B[91]);
    if (bw > 5) { g.fillStyle = "#fafafa"; g.beginPath(); polyJ(g, B, [30, 31, 32, 33]); g.fill(); if (bw > 11) { PQ[0] = B[90]; PQ[1] = B[91]; PQ[3] = B[93]; PQ[4] = B[94]; PQ[6] = B[99]; PQ[7] = B[100]; affine(g, 0, 1, 2, 100, 60);
        g.fillStyle = "#111"; g.font = `bold 48px ${FONT}`; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(String(BIB[e.name] || 0), 50, 33); g.setTransform(DPR, 0, 0, DPR, 0, 0); } } }; const head = () => {
    const x = B[0], y = B[1], s = B[2]; g.fillStyle = sk; g.beginPath(); cap(g, B, 1, 0, 0.056, 0.05, 0); g.fill();
    g.beginPath(); g.ellipse(x - 0.093 * s, y + 0.01 * s, 0.02 * s, 0.032 * s, 0, 0, TAU); g.ellipse(x + 0.093 * s, y + 0.01 * s, 0.02 * s, 0.032 * s, 0, 0, TAU); g.fill(); g.beginPath(); g.ellipse(x, y, 0.094 * s, 0.116 * s, 0, 0, TAU); g.fill();
    g.fillStyle = L[4]; g.beginPath(); g.ellipse(x, y - 0.018 * s, 0.1 * s, 0.108 * s, 0, 0, TAU); g.fill(); if (L[6] === 1) { g.beginPath(); g.arc(B[63], B[64] + 0.01 * s, 0.05 * s, 0, TAU); g.fill(); }
    if (L[6] === 2) { const sw = Math.sin(e.ph * 2) * 0.05 * s * e.k; g.beginPath(); g.moveTo(x - 0.03 * s, y - 0.06 * s); g.quadraticCurveTo(x + sw, y + 0.12 * s, x + sw * 2, y + 0.2 * s); g.quadraticCurveTo(x + sw + 0.01 * s, y + 0.1 * s, x + 0.03 * s, y - 0.06 * s); g.fill(); }
    g.fillStyle = "rgba(255,240,215,0.22)"; g.beginPath(); g.ellipse(x - 0.04 * s, y - 0.05 * s, 0.045 * s, 0.06 * s, -0.4, 0, TAU); g.fill(); }; // limbs swung forward are farther from the camera: draw them before the torso, the others after
  const kL = e.z3[12], kR = e.z3[13], aL = e.z3[6], aR = e.z3[7]; if (aL > 0.05) arm(0); if (aR > 0.05) arm(1); if (kL > kR) { leg(0); leg(1); } else { leg(1); leg(0); } torso(); if (aL <= 0.05) arm(0); if (aR <= 0.05) arm(1); head();
}
// shadows: one low-resolution offscreen layer (soft for free when scaled up), composited once so overlapping shadows do not darken
let SHC = cnv(1, 1), SHG = SHC.getContext("2d"), SHS = 0.5;
function drawShadow(B) { const g = SHG; g.beginPath();
  for (const s of LEG) for (let n = 0; n < s.length; n += 4) cap(g, B, s[n], s[n + 1], s[n + 2], s[n + 3], 0); for (const s of ARM) for (let n = 0; n < s.length; n += 4) cap(g, B, s[n], s[n + 1], s[n + 2], s[n + 3], 0);
  g.fill(); g.beginPath(); polyJ(g, B, [2, 3, 23, 22]); g.fill(); g.beginPath(); cap(g, B, 1, 21, 0.1, 0.1, 0); g.fill();
}
// HUD helpers
const LED = {"0": [14, 17, 19, 21, 25, 17, 14], "1": [4, 12, 4, 4, 4, 4, 14], "2": [14, 17, 1, 2, 4, 8, 31], "3": [31, 2, 4, 2, 1, 17, 14], "4": [2, 6, 10, 18, 31, 2, 2], "5": [31, 16, 30, 1, 1, 17, 14],
  "6": [6, 8, 16, 30, 17, 17, 14], "7": [31, 1, 2, 4, 8, 8, 8], "8": [14, 17, 17, 14, 17, 17, 14], "9": [14, 17, 17, 15, 1, 2, 12], "M": [17, 27, 21, 21, 17, 17, 17], ".": [0, 0, 0, 0, 0, 3, 3], "-": [0, 0, 0, 31, 0, 0, 0], " ": [0, 0, 0, 0, 0, 0, 0]};
const ledW = ch => ch === "." ? 2 : 5;
function ledText(g, s, col, row, p, right) {   // lit dots of string s on the panel grid (pitch p), into the current path
  let w = 0; for (const ch of s) w += ledW(ch) + 1; if (right) col -= w - 1;
  for (const ch of s) { const m = LED[ch] || LED[" "], cw = ledW(ch); for (let y = 0; y < 7; y++) for (let x = 0; x < cw; x++) if (m[y] >> (cw - 1 - x) & 1) g.rect((col + x) * p, (row + y) * p, p * 0.78, p * 0.78); col += cw + 1; }
}
function rrect(g, x, y, w, h, r) { g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
let UI = 1, SKY = null, SKY0 = 0, LEDP = null, LEDPP = 3, TBAR = null, GR = {};
function setupSize(W, H, dpr) { DPR = dpr; VW = W; VH = H; W2 = W / 2; HC = H / 2; FOC = Math.max(W2 / Math.tan(35 * Math.PI / 180), HC / Math.tan(26 * Math.PI / 180));
  UI = Math.max(0.62, Math.min(1.15, W / 640)); LODZ = 0.09 * FOC * dpr; const hor = HC - FOC * Math.tan(0.105); SKY0 = hor + 24; SKY = cnv(W * 1.6, SKY0 + 30); const g = SKY.getContext("2d"), sg = g.createLinearGradient(0, 0, 0, SKY.height);
  sg.addColorStop(0, "#4f8fd0"); sg.addColorStop(0.6, "#9cc6e8"); sg.addColorStop(1, "#e3edf3"); g.fillStyle = sg; g.fillRect(0, 0, SKY.width, SKY.height);
  seed = 5; for (let i = 0; i < 26; i++) { const cx = rand() * SKY.width, cy = 10 + rand() * SKY0 * 0.5, r = (18 + rand() * 40) * UI;
    for (const o of [-SKY.width, 0, SKY.width]) { const cg = g.createRadialGradient(cx + o, cy, 0, cx + o, cy, r); cg.addColorStop(0, "rgba(255,255,255,0.75)"); cg.addColorStop(1, "rgba(255,255,255,0)"); g.fillStyle = cg; g.fillRect(cx + o - r, cy - r, 2 * r, 2 * r); } }
  const t = rctx.createLinearGradient(0, hor, 0, H); t.addColorStop(0, "#94402f"); t.addColorStop(0.35, "#bb4e3a"); t.addColorStop(1, "#d2603f"); GR.track = t;
  const hz = rctx.createLinearGradient(0, hor - 6, 0, hor + H * 0.22); hz.addColorStop(0, "rgba(206,214,226,0.6)"); hz.addColorStop(0.35, "rgba(206,214,226,0.22)"); hz.addColorStop(1, "rgba(206,214,226,0)"); GR.haze = hz; GR.hor = hor;
  const kl = rctx.createRadialGradient(W * 0.32, H * 1.05, 0, W * 0.32, H * 1.05, W * 0.75); kl.addColorStop(0, "rgba(255,200,140,0.24)"); kl.addColorStop(1, "rgba(255,196,130,0)"); GR.key = kl;
  const vg = rctx.createRadialGradient(W2, H * 0.55, Math.min(W, H) * 0.45, W2, H * 0.55, Math.max(W, H) * 0.78); vg.addColorStop(0, "rgba(0,0,0,0)"); vg.addColorStop(1, "rgba(0,0,0,0.42)"); GR.vig = vg; SHC = cnv(W * SHS, H * SHS); SHG = SHC.getContext("2d");
  LEDPP = Math.max(2, Math.round(2.6 * UI * 2) / 2); LEDP = cnv((34 * LEDPP + 10) * dpr, (19 * LEDPP + 10) * dpr); { const p = LEDPP, lg = LEDP.getContext("2d"); lg.scale(dpr, dpr);
    lg.fillStyle = "#0d0c12"; lg.beginPath(); rrect(lg, 0, 0, 34 * p + 10, 19 * p + 10, 5); lg.fill(); lg.strokeStyle = "#5c5a66"; lg.lineWidth = 2; lg.stroke();
    lg.fillStyle = "#26201a"; lg.translate(5, 5); lg.beginPath(); for (let y = 0; y < 19; y++) if (y < 7 || y > 10) for (let x = 0; x < 34; x++) lg.rect(x * p, y * p, p * 0.78, p * 0.78); lg.fill(); }
  const bw = 236 * UI, bh = 40 * UI; TBAR = cnv((bw + 4) * dpr, (bh + 4) * dpr); { const tg = TBAR.getContext("2d"); tg.scale(dpr, dpr);
    const gr = tg.createLinearGradient(0, 0, bw, 0); gr.addColorStop(0, "rgba(30,24,40,0.15)"); gr.addColorStop(0.35, "rgba(30,24,40,0.78)"); gr.addColorStop(1, "rgba(30,24,40,0.9)");
    tg.fillStyle = gr; tg.beginPath(); tg.moveTo(bh * 0.4, 0); tg.lineTo(bw, 0); tg.lineTo(bw, bh); tg.lineTo(0, bh); tg.closePath(); tg.fill();
    const mg = tg.createLinearGradient(0, 0, bw, 0); mg.addColorStop(0, "rgba(228,0,124,0)"); mg.addColorStop(0.4, "#e4007c"); mg.addColorStop(1, "#ff3d9a"); tg.fillStyle = mg; tg.fillRect(bh * 0.2, bh - 4 * UI, bw - bh * 0.2, 4 * UI);
    tg.fillStyle = "#fff"; tg.beginPath(); rrect(tg, bw - 74 * UI, 6 * UI, 68 * UI, bh - 14 * UI, 4 * UI); tg.fill(); rings(tg, bw - 40 * UI, 14 * UI, 6.2 * UI, ["#0085c7", "#f4c300", "#000000", "#009f3d", "#df0024", "#ffffff"], 1.7 * UI); } sizedW = rc.width;
}
function updateScreen(r, t, d) {   // the big screen on the far stand: live time and distance
  const g = SCRG; g.fillStyle = "#07070c"; g.fillRect(0, 0, 256, 128); g.fillStyle = "#c3338b"; g.fillRect(0, 0, 256, 26);
  g.fillStyle = "#fff"; g.textBaseline = "middle"; g.textAlign = "left"; g.font = `bold 15px ${FONT}`; g.fillText("London 2012", 8, 14); rings(g, 218, 10, 5.5, ["#fff", "#fff", "#fff", "#fff", "#fff"], 1.4);
  g.textAlign = "center"; g.font = `bold 50px ${FONT}`; g.fillText(t.toFixed(2), 128, 66); g.font = `600 18px ${FONT}`; g.fillStyle = "#ffd100"; g.fillText(r.phase === "idle" ? "ATHLETICS  " + r.distance + " m" : Math.round(d) + " m  /  " + r.distance + " m", 128, 108);
  const p = rctx.createPattern(SCR, "no-repeat"); TEX.scr.pat = [p, p, p];
}
let sizedW = 0, lastT = 0, scrAt = 0, frameMs = 0, goAt = 0, ELX = 0, FINAT = 0, LASTPH = "";
const RS = {}, FIELD = Array.from({length: 6}, (_, i) => ({name: "", who: "", kmh: 0, d: 0, u: 0, w: 0, z: 0, vis: false, ph: 0, k: 0, b: RB[i], z3: new Float64Array(JN)}));
function fitRace() { const d = Math.min(2, window.devicePixelRatio || 1); rc.width = Math.round(rc.clientWidth * d); rc.height = Math.round(SCENE_H * d); sizedW = 0; }
window.addEventListener("resize", fitRace); fitRace();
function runner(e, name, who, kmh, dist, total, live, el0, tclk, fdt, cad, tf) {   // smoothed position, effort and stride phase for one runner
  const st = RS[name] || (RS[name] = {d: 0, ph: rand() * TAU, k: 0}), v = kmh / 3.6, ts = tclk - tf; let target = Math.min(total, dist + (live ? v * Math.max(0, tclk - el0) : 0)), va = live ? v : 0;
  if (live && ts > 0) { const dec = Math.exp(-ts / 1.8); target = total + v * 1.8 * (1 - dec); va = v * dec; }   // over the line: run out and ease off
  st.d = !live || Math.abs(target - st.d) > 12 ? target : st.d + (target - st.d) * (1 - Math.exp(-fdt * 8)); const kt = va > 0.3 ? Math.min(1, 0.22 + va / 7.5) : 0; st.k += (kt - st.k) * (1 - Math.exp(-fdt * 3));
  const f = cad > 0 && !(ts > 0) ? cad / 120 : va > 0.3 ? va / (2 * (0.6 + 0.16 * va)) : 0; st.ph = (st.ph + TAU * f * fdt) % TAU; e.name = name; e.who = who; e.kmh = kmh; e.d = st.d; e.ph = st.ph; e.k = st.k; e.u = U0 + st.d; e.w = (LANE[name] - 0.5) * LW; return e;
}
const TAGS = new Float64Array(18), BAR = new Float64Array(6), SV = new Int16Array(NSEG), SK = new Float64Array(NSEG), ORD = [0, 0, 0, 0, 0, 0, 0, 0];
function drawRace() { requestAnimationFrame(drawRace);
  const r = raceSnap; if (!r || rc.width === 0 || rc.clientWidth === 0) return; const now = performance.now(), fdt = Math.min(0.1, lastT ? (now - lastT) / 1000 : 0.016); lastT = now;
  if (Math.abs(rc.clientWidth * Math.min(2, window.devicePixelRatio || 1) - rc.width) > 2) fitRace(); const dpr = rc.height / SCENE_H, W = rc.width / dpr, H = SCENE_H, ctx = rctx; if (sizedW !== rc.width) setupSize(W, H, dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const run = r.phase === "running", fin = r.phase === "finished", dt = (now - raceSnapAt) / 1000; if (r.phase !== LASTPH) { LASTPH = r.phase; FINAT = now; ELX = 0; }
  const el = run ? (ELX = Math.max(ELX, r.elapsed + dt)) : r.elapsed, tclk = fin ? r.elapsed + (now - FINAT) / 1000 : el, live = run || fin;   // the clock runs on after the finish for the run-out
  // runners: you plus the ghosts (the Champion hides while your own record run is on show)
  let n = 0; const res = fin && r.result, me = runner(FIELD[n++], "YOU", r.who, res ? res.avg_kmh : r.speed_kmh, r.dist, r.distance, live, r.elapsed, tclk, fdt, run ? r.cadence || 0 : 0, fin ? (res ? res.time : r.elapsed) : 1e9);
  for (const g of r.ghosts) if (n < 6 && LANE[g.name] && !(g.name === "Champion" && fin && res && res.record))
    runner(FIELD[n++], g.name, g.who || "", g.kmh, g.dist, r.distance, live, r.elapsed, tclk, fdt, 0, g.name === "Champion" && r.champion ? r.champion.time : r.distance / Math.max(0.1, g.kmh / 3.6));
  const bob = -0.03 * me.k * Math.cos(2 * me.ph), sway = 0.025 * me.k * Math.sin(me.ph); setCam(me.u, me.w, bob, sway); if (now - scrAt > 100) { scrAt = now; updateScreen(r, el, me.d); } // sky (scrolls with the camera yaw), then the stands far to near
  const sw = SKY.width, sx = -((YAW * FOC) % sw + sw) % sw; ctx.drawImage(SKY, sx, HOR - SKY0); ctx.drawImage(SKY, sx + sw, HOR - SKY0); const tanH = W2 / FOC * 1.15; let nv = 0; for (let i = 0; i < NSEG; i++) { let front = 0, left = 0, right = 0;
    for (let c = 0; c < 4; c++) { trackPt(i * SEG + (c & 1 ? SEG : 0), c & 2 ? 58 : 10.5); const dx = TX - CX, dz = TZ - CZ, f = dx * CHX + dz * CHZ, x = dx * RX + dz * RZ; if (f > 1) { front++; if (x > f * tanH) right++; else if (x < -f * tanH) left++; } }
    if (!front || right === 4 || left === 4) continue; trackPt(i * SEG + SEG / 2, 24); const dx = TX - CX, dz = TZ - CZ; SK[i] = dx * dx + dz * dz; let j = nv++; while (j > 0 && SK[SV[j - 1]] < SK[i]) { SV[j] = SV[j - 1]; j--; } SV[j] = i; } for (let v = 0; v < nv; v++) {
    const i = SV[v], u = i * SEG + SEG / 2; trackPt(u, 0); const face = (HX * SUN[2] - HZ * SUN[0]) / 0.85;   // inward normal (left of the heading) . sun
    drawSeg(i, HAZE[Math.max(0, Math.min(4, Math.round(2 - 2.6 * face)))][Math.min(10, Math.round(10 * (1 - Math.exp(-Math.sqrt(SK[i]) / 230))))]); } // ground: infield with mowing stripes, track with grain, markings, kerb
  ctx.fillStyle = "#58a043"; ctx.beginPath(); strip(-0.9, -R, 0); ctx.fill(); ctx.fillStyle = "rgba(24,70,16,0.22)"; ctx.beginPath(); for (let w = -4.9; w > -R + 4; w -= 8) strip(w, w - 4, 0); ctx.fill();
  ctx.fillStyle = "#8c8784"; ctx.beginPath(); strip(-0.12, -0.9, 0); ctx.fill(); ctx.fillStyle = "#8a8f98"; ctx.beginPath(); strip(9.5, 10.5, 0); ctx.fill(); ctx.fillStyle = "#4b4d56"; ctx.beginPath(); strip(10.5, 12.5, 0); ctx.fill();
  ctx.beginPath(); strip(0, 9.5, 0); ctx.save(); ctx.translate(0, HOR - GR.hor); ctx.fillStyle = GR.track; ctx.fill(); ctx.restore(); ctx.save(); ctx.clip();
  for (let y = H, s = 1; y > HOR + 3 && s > 0.12;) {   // grain in screen bands, scaled to the ground distance and scrolled with your run
    const D = CY * FOC / Math.max(1, y - HOR); s = FOC / D / 64; const h = Math.max(6, (y - HOR) * 0.22); GM.a = GM.d = s; GM.e = W2; GM.f = y + ((me.u * FOC / D) % (128 * s));
    GRAINP.setTransform(GM); ctx.fillStyle = GRAINP; ctx.globalAlpha = Math.min(1, s * 1.6); ctx.fillRect(0, y - h, W, h); y -= h; } ctx.globalAlpha = 1; ctx.restore(); ctx.fillStyle = "#f3efe8"; ctx.beginPath(); for (let l = 1; l <= 6; l++) lineStrip(l * LW); ctx.fill();
  const uF = U0 + r.distance; ctx.beginPath(); quadUW(U0 - 0.05, U0 + 0.05, 0, 7.32, 0, 7.32, 0); quadUW(uF - 0.32, uF, 0, 7.32, 0, 7.32, 0); ctx.fill();
  for (const [du, col] of [[10, "#f5d000"], [20, "#2f7ad8"], [30, "#2fb84a"], [40, "#2f7ad8"]]) { ctx.fillStyle = col; ctx.beginPath(); for (let l = 1; l < 6; l++) quadUW(U0 + du, U0 + du + 0.45, l * LW - 0.06, l * LW + 0.06, l * LW - 0.06, l * LW + 0.06, 0); ctx.fill(); }
  ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = `bold 100px ${FONT}`;
  for (let l = 1; l <= 6; l++) { const w = (l - 0.5) * LW; if (pq(0, U0 - 1.2, w - 0.32, 0) && pq(1, U0 - 1.2, w + 0.32, 0) && pq(2, U0 - 1.9, w - 0.32, 0)) { affine(ctx, 0, 1, 2, 100, 100); ctx.fillText(String(l), 50, 52); } } ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  for (let c = 0; c < 2; c++) { ctx.fillStyle = c ? "#d8343a" : "#f6f4f0"; ctx.beginPath(); for (let d = 0; d < 70; d++) if (((Math.floor(CAMU + d) % 2) + 2) % 2 === c) quadUW(Math.floor(CAMU) + d, Math.floor(CAMU) + d + 1, -0.12, 0, -0.12, 0, 0.07); ctx.fill(); }
  ctx.fillStyle = "#e8dcd8"; ctx.beginPath(); for (let d = 70; d < LOOP - 8;) { const s = stepAt(d); quadUW(CAMU + d, CAMU + d + s, -0.12, 0, -0.12, 0, 0.07); d += s; } ctx.fill();
  ctx.fillStyle = "#8e8681"; ctx.beginPath(); for (let d = 0; d < 70; d += 1.5) quadV(CAMU + d, CAMU + d + 1.5, 0, 0, 0.07); ctx.fill();
  ctx.fillStyle = "#f3c300"; ctx.beginPath(); for (let m = 20; m < 400; m += 20) { const u = U0 + m; trackPt(u, -0.45); const dz = (TX - CX) * CHX + (TZ - CZ) * CHZ; if (dz < 2 || dz > 130) continue; setQ(0, u, -0.32, 0); setQ(1, u, -0.58, 0); setQ(2, u, -0.53, 0.32); setQ(3, u, -0.37, 0.32); polyW(4); } ctx.fill();
  ctx.fillStyle = GR.haze; ctx.save(); ctx.translate(0, HOR - GR.hor); ctx.fillRect(0, GR.hor - 8, W, H * 0.24); ctx.restore(); // runners: project, shadows (one soft layer), then bodies and the stadium props far to near
  let nr = 0; for (let i = 0; i < n; i++) { const e = FIELD[i]; pose(e.ph, e.k); for (let j = 0; j < JN; j++) e.z3[j] = JL[3 * j + 2]; e.vis = placeRunner(e.b, e.u, e.w); trackPt(e.u, e.w); e.z = (TX - CX) * CHX + (TZ - CZ) * CHZ; if (e.vis) ORD[nr++] = i; }
  SHG.setTransform(1, 0, 0, 1, 0, 0); SHG.clearRect(0, 0, SHC.width, SHC.height); SHG.setTransform(SHS, 0, 0, SHS, 0, 0); SHG.fillStyle = "#1c0505"; for (let i = 0; i < nr; i++) drawShadow(FIELD[ORD[i]].b.S);
  ctx.globalAlpha = 0.4; ctx.drawImage(SHC, 0, 0, W, H); ctx.globalAlpha = 1; const props = [[uF, "gantry"], [U0 + 28, "board"]]; let pz = [0, 0]; for (let p = 0; p < 2; p++) { trackPt(props[p][0], 3); pz[p] = (TX - CX) * CHX + (TZ - CZ) * CHZ; }
  for (let i = 1; i < nr; i++) { const k = ORD[i]; let j = i; while (j > 0 && FIELD[ORD[j - 1]].z < FIELD[k].z) { ORD[j] = ORD[j - 1]; j--; } ORD[j] = k; } let pi = pz[0] > pz[1] ? [0, 1] : [1, 0], q = 0;
  const prop = p => props[p][1] === "gantry" ? gantry(props[p][0], el) : scoreboard(props[p][0], r, el); for (let i = 0; i < nr; i++) { const e = FIELD[ORD[i]]; while (q < 2 && pz[pi[q]] > e.z) prop(pi[q++]); drawRunner(e); } while (q < 2) prop(pi[q++]);
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  let nt = 0; { const bw = Math.max(12, 15 * UI), bh = Math.max(70, 0.24 * H), bx = W * 0.6, by = HOR - bh * 0.78; BAR[0] = bx - 4; BAR[1] = by - 4; BAR[2] = bx + bw * 1.55 + 13; BAR[3] = by + bh + 4; BAR[4] = bw; BAR[5] = bh; }   // power bar rectangle
  for (let i = nr - 1; i >= 0; i--) { const e = FIELD[ORD[i]]; if (e.name === "YOU" || e.z > 90) continue; const B = e.b.J, fs = Math.max(9, Math.min(13 * UI, B[2] * 0.11)), label = e.name === "Champion" ? "Champion " + (e.who || "") : e.name;
    ctx.font = `600 ${fs}px ${FONT}`; const tw = ctx.measureText(label).width + fs * 0.9; let x = Math.max(tw / 2 + 3, Math.min(W - tw / 2 - 3, B[63])); let y = B[64] - fs * 1.1;
    for (let t = 0; t < nt; t++) if (Math.abs(TAGS[3 * t] - x) < (TAGS[3 * t + 2] + tw) / 2 && Math.abs(TAGS[3 * t + 1] - y) < fs * 1.4) { y = TAGS[3 * t + 1] - fs * 1.45; t = -1; }   // nearer tags keep their place, farther ones step up
    if (run && y + fs * 0.4 > BAR[1] && y - fs * 0.95 < BAR[3] && x + tw / 2 > BAR[0] && x - tw / 2 < BAR[2]) x = x < (BAR[0] + BAR[2]) / 2 ? BAR[0] - tw / 2 - 2 : BAR[2] + tw / 2 + 2;   // keep tags off the power bar
    if (nt < 6) { TAGS[3 * nt] = x; TAGS[3 * nt + 1] = y; TAGS[3 * nt + 2] = tw; nt++; } ctx.fillStyle = "rgba(20,14,30,0.55)"; ctx.beginPath(); rrect(ctx, x - tw / 2, y - fs * 0.95, tw, fs * 1.35, fs * 0.4); ctx.fill(); ctx.fillStyle = "#fff"; ctx.fillText(label, x, y); }
  ctx.fillStyle = GR.key; ctx.fillRect(0, 0, W, H); ctx.fillStyle = GR.vig; ctx.fillRect(0, 0, W, H); hud(ctx, r, W, H, el, me, n); frameMs = frameMs * 0.95 + (performance.now() - now) * 0.05; rc.dataset.ms = frameMs.toFixed(2);
}
function gantry(u, el) {   // the finish: a post pair carrying the photo-finish camera and the clock, and a FINISH sign outside
  const ctx = rctx; ctx.fillStyle = "#eef0f4"; ctx.beginPath(); for (const w of [-0.9, 9.9]) { setQ(0, u, w - 0.06, 0); setQ(1, u, w + 0.06, 0); setQ(2, u, w + 0.06, 1.75); setQ(3, u, w - 0.06, 1.75); polyW(4); } ctx.fill();
  ctx.fillStyle = "#1e2026"; ctx.beginPath(); for (const w of [-0.9, 9.9]) { setQ(0, u, w - 0.24, 1.7); setQ(1, u, w + 0.24, 1.7); setQ(2, u, w + 0.24, 2.08); setQ(3, u, w - 0.24, 2.08); polyW(4); } ctx.fill();
  ctx.fillStyle = "#c3338b"; ctx.beginPath(); setQ(0, u, 10.9, 0.55); setQ(1, u, 9.96, 0.55); setQ(2, u, 9.96, 1.25); setQ(3, u, 10.9, 1.25); polyW(4); ctx.fill();
  if (pq(0, u, 9.96, 1.25) && pq(1, u, 10.9, 1.25) && pq(2, u, 9.96, 0.55) && PQ[3] - PQ[0] > 14) { affine(ctx, 0, 1, 2, 100, 74); ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = `bold 26px ${FONT}`; ctx.fillText("FINISH", 50, 38); ctx.setTransform(DPR, 0, 0, DPR, 0, 0); }
  if (pq(0, u, -1.12, 2.02) && pq(1, u, -0.68, 2.02) && pq(2, u, -1.12, 1.76) && PQ[3] - PQ[0] > 10) { affine(ctx, 0, 1, 2, 100, 60); ctx.fillStyle = "#ffc20e"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = `bold 34px ui-monospace, Menlo, monospace`; ctx.fillText(el.toFixed(1), 50, 32); ctx.setTransform(DPR, 0, 0, DPR, 0, 0); }
}
function scoreboard(u, r, el) {   // the infield board by the kerb, as in the reference
  const ctx = rctx; ctx.fillStyle = "#3c3f46"; ctx.beginPath(); for (const w of [-4.6, -7.4]) { setQ(0, u, w - 0.07, 0); setQ(1, u, w + 0.07, 0); setQ(2, u, w + 0.07, 1); setQ(3, u, w - 0.07, 1); polyW(4); } ctx.fill();
  if (!pq(0, u, -8.0, 2.4) || !pq(1, u, -4.0, 2.4) || !pq(2, u, -8.0, 0.9)) return;
  ctx.fillStyle = "#16171c"; ctx.beginPath(); setQ(0, u, -4, 2.4); setQ(1, u, -8, 2.4); setQ(2, u, -8, 0.9); setQ(3, u, -4, 0.9); polyW(4); ctx.fill(); ctx.strokeStyle = "#8d9099"; ctx.lineWidth = Math.max(1, 0.06 * FOC / PQ[2]); ctx.stroke();
  affine(ctx, 0, 1, 2, 400, 150); ctx.fillStyle = "#ffc20e"; ctx.textBaseline = "middle"; ctx.font = `bold 52px ui-monospace, Menlo, monospace`;
  ctx.textAlign = "right"; ctx.fillText(r.distance + "m", 370, 42); ctx.fillText(el.toFixed(2), 370, 108); ctx.fillStyle = "#c3338b"; ctx.fillRect(30, 70, 120, 8); ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
function hud(ctx, r, W, H, el, me, n) { const U = UI, pad = 14 * U, idle = r.phase === "idle", live = r.phase === "running" || r.phase === "finished";
  // scoreboard, top right: LED dots
  const p = LEDPP, pw = 34 * p + 10, px = W - pw - pad, py = pad; ctx.drawImage(LEDP, px, py, pw, 19 * p + 10); ctx.save(); ctx.translate(px + 5, py + 5); ctx.beginPath(); ledText(ctx, r.distance + "M", 0, 0, p, false); ledText(ctx, el.toFixed(2), 33, 11, p, true);
  ctx.strokeStyle = "rgba(255,160,40,0.3)"; ctx.lineWidth = p * 0.6; ctx.stroke(); ctx.fillStyle = "#ffb627"; ctx.fill(); ctx.restore(); // power bar: speed as a fraction of 22 km/h, plus a cadence meter
  if (r.phase === "running") { const bw = BAR[4], bh = BAR[5], bx = BAR[0] + 4, by = BAR[1] + 4, frac = Math.min(1, Math.max(0, r.speed_kmh / 22)); ctx.fillStyle = "rgba(10,14,10,0.5)"; ctx.beginPath(); rrect(ctx, bx - 3, by - 3, bw + 6, bh + 6, 4); ctx.fill();
  const pg = ctx.createLinearGradient(0, by + bh, 0, by); pg.addColorStop(0, "#0a6b22"); pg.addColorStop(0.6, "#2fd23f"); pg.addColorStop(1, "#b6ff5a"); ctx.fillStyle = pg; ctx.fillRect(bx, by + bh * (1 - frac), bw, bh * frac);
  ctx.fillStyle = "rgba(255,255,255,0.32)"; ctx.fillRect(bx + 1.5, by + bh * (1 - frac), bw * 0.35, bh * frac);
  ctx.strokeStyle = "#f4f4f4"; ctx.lineWidth = 2; ctx.strokeRect(bx, by, bw, bh); ctx.fillStyle = "#f4f4f4"; for (let i = 1; i < 4; i++) ctx.fillRect(bx - 4, by + bh * i / 4 - 0.5, 6, 1);
  const cf = Math.min(1, (r.cadence || 0) / 200), sx = bx + bw + 7, sh = bh * 0.52, cy0 = by + bh - sh; ctx.fillStyle = "rgba(10,14,10,0.5)"; ctx.fillRect(sx - 2, cy0 - 2, bw * 0.55 + 4, sh + 4);
  for (let i = 0; i < 5; i++) { ctx.fillStyle = cf * 5 > i ? (i > 3 ? "#ffde00" : "#4be35a") : "rgba(255,255,255,0.12)"; ctx.fillRect(sx, cy0 + sh - (i + 1) * sh / 5 + 1.5, bw * 0.55, sh / 5 - 3); } } // position, bottom left
  let place = 1; for (let i = 1; i < n; i++) if (FIELD[i].d > me.d + 0.05) place++; if (r.phase === "finished" && r.result) place = r.result.place;
  ctx.save(); ctx.shadowColor = "rgba(0,0,0,0.55)"; ctx.shadowBlur = 8 * U; ctx.shadowOffsetX = 1.5 * U; ctx.shadowOffsetY = 2 * U; ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  if (live) { ctx.font = `600 ${15 * U}px ${COND}`; ctx.fillText("Position", pad + 4 * U, H - pad - 54 * U); ctx.font = `700 ${62 * U}px ${COND}`; ctx.fillText(String(place), pad, H - pad); } // runners behind the camera, with gaps
  ctx.font = `600 ${11.5 * U}px ${FONT}`; ctx.textAlign = "right"; let ly = H - pad - 52 * U;
  for (let i = 1; i < n; i++) { const e = FIELD[i]; if (e.vis && e.z > 1) continue; const gap = e.d - me.d; ctx.fillStyle = "#fff"; ctx.fillText((e.name === "Champion" ? "Champion " + (e.who || "") : e.name) + "  " + (gap >= 0 ? "+" : "") + gap.toFixed(1) + " m", W - pad - 14 * U, ly);
    ctx.fillStyle = LOOK[e.name][0]; ctx.fillRect(W - pad - 10 * U, ly - 9 * U, 8 * U, 9 * U); ly -= 15 * U; } ctx.restore(); // time bar with the rings, bottom right
  const tw = 236 * U, th = 40 * U, tx = W - tw - pad, ty = H - th - pad; ctx.drawImage(TBAR, tx, ty, tw + 4, th + 4);
  ctx.fillStyle = "#fff"; ctx.font = `700 ${24 * U}px ${COND}`; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(el.toFixed(2), tx + tw - 84 * U, ty + th * 0.47); // countdown and GO!
  ctx.textAlign = "center"; ctx.textBaseline = "middle"; if (r.phase === "countdown") { const c = Math.max(0, (r.countdown || 0) - (performance.now() - raceSnapAt) / 1000), num = Math.max(1, Math.ceil(c)), f = num - c, s = 1 + 1.2 * Math.max(0, 1 - f * 4) ** 2; goAt = 0;
    ctx.save(); ctx.globalAlpha = Math.min(1, f * 5); ctx.translate(W2, H * 0.4); ctx.scale(s, s); ctx.font = `800 ${92 * U}px ${COND}`; ctx.lineWidth = 7 * U; ctx.strokeStyle = "rgba(20,10,30,0.85)"; ctx.strokeText(String(num), 0, 0); ctx.fillStyle = "#fff"; ctx.fillText(String(num), 0, 0); ctx.restore(); }
  if (r.phase === "running" && el < 1.3) { const f = el / 1.3, s = 0.9 + 0.5 * f;
    ctx.save(); ctx.globalAlpha = 1 - f * f; ctx.translate(W2, H * 0.4); ctx.scale(s, s); ctx.font = `800 ${96 * U}px ${COND}`; ctx.lineWidth = 9 * U; ctx.strokeStyle = "#c3338b"; ctx.strokeText("GO!", 0, 0); ctx.fillStyle = "#fff36b"; ctx.fillText("GO!", 0, 0); ctx.restore();
    ctx.fillStyle = `rgba(255,255,255,${(0.35 * (1 - f) * (1 - f)).toFixed(3)})`; ctx.fillRect(0, 0, W, H); } if (idle) { const t = "Choose a distance and press Start"; ctx.font = `600 ${13 * U}px ${FONT}`; const w = ctx.measureText(t).width + 22 * U, y = H * 0.7;
    ctx.fillStyle = "rgba(18,14,28,0.62)"; ctx.beginPath(); rrect(ctx, W2 - w / 2, y - 12 * U, w, 24 * U, 12 * U); ctx.fill(); ctx.fillStyle = "#fff"; ctx.fillText(t, W2, y + 1); } // finish panel with a medal band
  if (r.phase === "finished" && r.result) { const res = r.result, medal = res.record ? ["#fff3a8", "#e8b923", "#9a7410"] : res.beat.includes("Runner") ? ["#ffffff", "#c9ccd3", "#7d828c"] : ["#ffd2a8", "#c9773a", "#7a3f16"];
    const pw2 = Math.min(360 * U, W - 2 * pad), ph = 118 * U, x = W2 - pw2 / 2, y = 0.13 * H; ctx.fillStyle = "rgba(18,14,28,0.88)"; ctx.beginPath(); rrect(ctx, x, y, pw2, ph, 10 * U); ctx.fill();
    const mg = ctx.createLinearGradient(x, 0, x + pw2, 0); mg.addColorStop(0, medal[2]); mg.addColorStop(0.5, medal[1]); mg.addColorStop(1, medal[2]); ctx.fillStyle = mg; ctx.fillRect(x, y + 26 * U, pw2, 4 * U);
    ctx.fillStyle = "#c3338b"; ctx.beginPath(); rrect(ctx, x, y, pw2, 26 * U, 10 * U); ctx.fill(); ctx.fillRect(x, y + 14 * U, pw2, 12 * U); ctx.fillStyle = "#fff"; ctx.font = `700 ${14 * U}px ${FONT}`; ctx.fillText("FINISH  " + r.distance + " m", W2, y + 14 * U);
    const mx = x + 42 * U, my = y + 72 * U, rg = ctx.createRadialGradient(mx - 6 * U, my - 6 * U, 2, mx, my, 24 * U); rg.addColorStop(0, medal[0]); rg.addColorStop(0.6, medal[1]); rg.addColorStop(1, medal[2]);
    ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(mx, my, 23 * U, 0, TAU); ctx.fill(); ctx.fillStyle = "rgba(40,24,0,0.75)"; ctx.font = `800 ${20 * U}px ${COND}`; ctx.fillText(String(res.place), mx, my + 1);
    ctx.textAlign = "left"; ctx.fillStyle = "#fff"; ctx.font = `700 ${30 * U}px ${COND}`; ctx.fillText(ordinal(res.place) + "   " + res.time.toFixed(2) + " s", x + 80 * U, y + 60 * U); ctx.font = `500 ${12.5 * U}px ${FONT}`; ctx.fillStyle = res.record ? "#ffd84a" : "#d8d8e0";
    ctx.fillText(res.record ? "NEW RECORD: you are the Champion ghost" : "beat " + (res.beat.length ? res.beat.join(", ") : "nobody"), x + 80 * U, y + 86 * U);
    ctx.fillStyle = "#a9a6b6"; ctx.fillText(res.avg_kmh.toFixed(1) + " km/h average" + (r.champion && !res.record ? "   record " + r.champion.time.toFixed(2) + " s" : ""), x + 80 * U, y + 104 * U); }
}
drawRace();
function render(s) {
  s.race.cadence = s.cad || 0;
  renderRace(s.race, s.speed_kmh);
  const act = $("act"); act.textContent = s.act; act.className = "verdict " + s.act.replace(" ", "");
  $("m1").textContent = s.m1;
  $("agree").textContent = s.agree == null ? "" : (s.agree ? "agrees" : "differs"); $("agree").className = s.agree ? "agree" : "differ";
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
</main>
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
POST /race/start?distance=100|400|1000&who=   POST /race/abort
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
            elif url.path == "/race/start":
                try:
                    self.gait.race.start(int(q.get("distance", ["400"])[0]), who)
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
            elif url.path == "/race/abort":
                self.gait.race.abort()
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

---

## Revision 4 (2026-10-26): racing on the spot (spec Revision 4)

### Task 21: `race.py`: speed model, result store, race state machine

**Files:**
- Create: `app/python/race.py`, `tests/test_race.py`

- [ ] **Step 1: Failing tests**

`tests/test_race.py`:
```python
import json

import pytest

from race import COUNTDOWN_S, DISTANCES, GHOSTS, WALK_CAP, Race, ResultStore, SpeedModel, raw_speed, replay_distance, stride_m


class Clock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


def test_stride_and_raw_speed_from_the_recordings():
    assert stride_m(1.0) == pytest.approx(0.6)
    assert stride_m(2.2) == pytest.approx(1.2)
    assert stride_m(10.0) == 2.2 and stride_m(0.0) == 0.5          # clamped
    assert raw_speed("WALK", 108, 1.3) == pytest.approx(108 / 60 * 0.75)   # 1.35 m/s = 4.9 km/h
    assert raw_speed("RUN", 160, 2.2) == pytest.approx(3.2)                 # 11.5 km/h
    assert raw_speed("RUN", 200, 4.0) == pytest.approx(200 / 60 * 2.1)      # 25 km/h
    assert raw_speed("WALK", 200, 4.0) == pytest.approx(WALK_CAP)           # walking is capped
    for act in ("STAND", "BIKE", "NO IMU", "---"):
        assert raw_speed(act, 160, 2.2) == 0.0
    assert raw_speed("RUN", None, 2.2) == 0.0 and raw_speed("RUN", 160, None) == 0.0


def test_speed_model_smooths_towards_the_target():
    m = SpeedModel()
    assert m.update("RUN", 160, 2.2, now=0.0) == pytest.approx(3.2)   # first sample lands at once
    v = m.update("STAND", None, None, now=0.5)
    assert 0 < v < 3.2                                                 # decays, does not drop to zero at once
    for i in range(1, 12):
        v = m.update("STAND", None, None, now=0.5 + i * 0.5)
    assert v < 0.02                                                    # gone after a few seconds


def test_replay_distance_integrates_a_profile():
    prof = [(0.5, 2.0), (1.0, 2.0), (1.5, 4.0)]         # (elapsed s, speed m/s) samples
    assert replay_distance(prof, 0.0) == 0.0
    assert replay_distance(prof, 1.0) == pytest.approx(2.0)
    assert replay_distance(prof, 1.5) == pytest.approx(4.0)
    assert replay_distance(prof, 9.0) == pytest.approx(4.0)   # holds the last sample


def test_race_runs_countdown_distance_finish_and_saves(tmp_path):
    clk = Clock()
    store = ResultStore(tmp_path / "races.json")
    r = Race(store, clock=clk)
    assert r.snapshot()["phase"] == "idle" and r.snapshot()["leaderboard"] == []
    with pytest.raises(ValueError):
        r.start(123, "ann")
    r.start(100, "ann")
    s = r.snapshot()
    assert s["phase"] == "countdown" and s["countdown"] == pytest.approx(COUNTDOWN_S) and s["distance"] == 100
    clk.t += COUNTDOWN_S
    r.tick(5.0, clk.t)
    assert r.snapshot()["phase"] == "running"
    for _ in range(19):                        # 19 x 1 s at 5 m/s = 95 m
        clk.t += 1.0
        r.tick(5.0, clk.t)
    s = r.snapshot()
    assert s["phase"] == "running" and s["dist"] == pytest.approx(95.0) and s["elapsed"] == pytest.approx(19.0)
    ghosts = {g["name"]: g["dist"] for g in s["ghosts"]}
    assert ghosts["Jogger"] == pytest.approx(9 / 3.6 * 19) and ghosts["Sprinter"] == 100   # sprinter already home
    clk.t += 2.0                               # 10 m in 2 s: crosses the line after 1 s of it
    r.tick(5.0, clk.t)
    s = r.snapshot()
    assert s["phase"] == "finished" and s["dist"] == 100
    assert s["result"]["time"] == pytest.approx(20.0) and s["result"]["avg_kmh"] == pytest.approx(18.0)
    assert s["result"]["beat"] == ["Jogger", "Runner"] and s["result"]["rank"] == 1
    assert s["result"]["place"] == 2 and s["result"]["record"] is True
    assert s["champion"]["who"] == "ann"
    assert store.leaderboard(100)[0]["who"] == "ann"
    assert json.loads((tmp_path / "races.json").read_text())[0]["distance"] == 100


def _finish(r, clk, speed):
    clk.t += COUNTDOWN_S
    r.tick(speed, clk.t)
    while r.snapshot()["phase"] == "running":
        clk.t += 1.0
        r.tick(speed, clk.t)


def test_champion_ghost_is_the_fastest_of_anyone(tmp_path):
    clk = Clock()
    store = ResultStore(tmp_path / "races.json")
    r = Race(store, clock=clk)
    r.start(100, "ann")
    _finish(r, clk, 4.0)
    assert r.snapshot()["phase"] == "finished"
    r.start(100, "bob")
    gh = r.snapshot()["ghosts"]
    assert [g["name"] for g in gh] == [n for n, _ in GHOSTS] + ["Champion"] and gh[-1]["who"] == "ann"
    clk.t += COUNTDOWN_S + 2.0
    r.tick(5.0, clk.t)
    champ = next(g for g in r.snapshot()["ghosts"] if g["name"] == "Champion")
    assert champ["dist"] == pytest.approx(8.0)
    while r.snapshot()["phase"] == "running":
        clk.t += 1.0
        r.tick(5.0, clk.t)
    res = r.snapshot()["result"]
    assert res["record"] is True and "Champion" in res["beat"]
    r.start(100, "ann")
    assert r.snapshot()["ghosts"][-1]["who"] == "bob"
    r.abort()
    assert r.snapshot()["phase"] == "idle"
    r.start(400, "bob")
    assert [g["name"] for g in r.snapshot()["ghosts"]] == [n for n, _ in GHOSTS]


def test_anon_can_hold_the_record(tmp_path):
    clk = Clock()
    r = Race(ResultStore(tmp_path / "races.json"), clock=clk)
    r.start(100, "")
    _finish(r, clk, 4.0)
    assert r.snapshot()["phase"] == "finished"
    r.start(100, "")
    assert r.snapshot()["ghosts"][-1]["who"] == "anon"


def test_slower_than_the_champion_is_no_record(tmp_path):
    clk = Clock()
    r = Race(ResultStore(tmp_path / "races.json"), clock=clk)
    r.start(100, "ann")
    _finish(r, clk, 5.0)
    r.start(100, "bob")
    _finish(r, clk, 4.0)
    res = r.snapshot()["result"]
    assert res["record"] is False and "Champion" not in res["beat"] and res["place"] == 3


def test_result_store_persists_and_ranks(tmp_path):
    p = tmp_path / "races.json"
    s1 = ResultStore(p)
    s1.save(400, "ann", 120.0, [(0.5, 3.3)])
    s1.save(400, "bob", 90.0, [(0.5, 4.4)])
    s1.save(100, "bob", 20.0, [(0.5, 5.0)])
    s2 = ResultStore(p)                        # re-read from disk
    board = s2.leaderboard(400)
    assert [e["who"] for e in board] == ["bob", "ann"] and board[0]["avg_kmh"] == pytest.approx(16.0)
    assert s2.champion(400)["who"] == "bob" and s2.champion(400)["time"] == 90.0 and s2.champion(1000) is None
    assert DISTANCES == (100, 400, 1000)


def test_no_distance_during_countdown_and_handoff_counts_from_t0(tmp_path):
    clk = Clock()
    r = Race(ResultStore(tmp_path / "races.json"), clock=clk)
    r.start(100, "ann")
    t0 = clk.t + COUNTDOWN_S
    r.tick(5.0, t0 - 1.0)
    assert r.snapshot()["dist"] == 0.0
    r.tick(5.0, t0 + 0.4)
    assert r.snapshot()["dist"] == pytest.approx(2.0)


def test_profile_is_decimated_to_half_seconds(tmp_path):
    clk = Clock()
    store = ResultStore(tmp_path / "races.json")
    r = Race(store, clock=clk)
    r.start(100, "ann")
    t0 = clk.t + COUNTDOWN_S
    i = 0
    while r.snapshot()["phase"] != "finished":
        i += 1
        r.tick(5.0, t0 + i * 0.05)
    entry = json.loads((tmp_path / "races.json").read_text())[0]
    assert 38 <= len(entry["profile"]) <= 44
    assert entry["profile"][-1][0] == pytest.approx(entry["time"], abs=0.01)


def test_start_during_save_does_not_leak_the_result(tmp_path):
    clk = Clock()
    store = ResultStore(tmp_path / "races.json")
    r = Race(store, clock=clk)
    orig = store.save

    def save(*a, **k):
        out = orig(*a, **k)
        r.start(100, "bob")
        return out

    store.save = save
    r.start(100, "ann")
    clk.t += COUNTDOWN_S
    r.tick(50.0, clk.t)
    clk.t += 5.0
    r.tick(50.0, clk.t)
    s = r.snapshot()
    assert s["phase"] == "countdown" and s["result"] is None


def test_corrupt_results_file_is_kept_aside(tmp_path):
    p = tmp_path / "races.json"
    p.write_text("{not json")
    store = ResultStore(p)
    assert store.leaderboard(100) == []
    bad = list(tmp_path.glob("races.json.bad-*"))
    assert len(bad) == 1 and bad[0].read_text() == "{not json"
    store.save(100, "ann", 20.0, [(0.5, 5.0)])
    assert json.loads(p.read_text())[0]["who"] == "ann"


def test_start_accepts_a_string_distance(tmp_path):
    r = Race(ResultStore(tmp_path / "races.json"), clock=Clock())
    r.start("400", "ann")
    assert r.snapshot()["distance"] == 400
```

Run: `.venv/bin/python -m pytest tests/test_race.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'race'`

- [ ] **Step 2: Implementation**

`app/python/race.py`:
```python
"""Racing on the spot: a speed from the gait, and a race against ghost runners.

speed = cadence/60 x stride, stride_m = clamp(0.6 + 0.5 x (peak_g - 1), 0.5, 2.2). The classifier gates it: STAND, BIKE
and no data give 0, WALK is capped at 7 km/h, RUN is uncapped. SpeedModel smooths it with a 1 s exponential average.
Race: start -> 3 s countdown -> running (distance integrates the smoothed speed every tick) -> finished when the line is
crossed (finish time interpolated inside the last step). Ghosts run at fixed paces, plus a replay of the runner's best
profile for that distance. ResultStore keeps every finish in a JSON file and serves leaderboards.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

STRIDE_AT_1G, STRIDE_PER_G, STRIDE_MIN, STRIDE_MAX = 0.6, 0.5, 0.5, 2.2
WALK_CAP = 7.0 / 3.6          # m/s
SMOOTH_TAU = 1.0              # s
COUNTDOWN_S = 3.0
DISTANCES = (100, 400, 1000)
GHOSTS = (("Jogger", 9.0), ("Runner", 13.0), ("Sprinter", 22.0))   # name, km/h
LEADERBOARD_N = 10
PROFILE_STEP_S = 0.5


def stride_m(peak_g):
    return max(STRIDE_MIN, min(STRIDE_MAX, STRIDE_AT_1G + STRIDE_PER_G * (peak_g - 1.0)))


def raw_speed(act, cad, pk):
    """m/s from one classified window, before smoothing."""
    if act not in ("WALK", "RUN") or not cad or pk is None:
        return 0.0
    v = cad / 60.0 * stride_m(pk)
    return min(v, WALK_CAP) if act == "WALK" else v


def replay_distance(profile, t):
    """Distance (m) a recorded (elapsed_s, speed_m_s) profile has covered after t seconds."""
    dist, last_t = 0.0, 0.0
    for elapsed, speed in profile:
        if elapsed >= t:
            dist += speed * (t - last_t)
            return dist
        dist += speed * (elapsed - last_t)
        last_t = elapsed
    return dist


class SpeedModel:
    def __init__(self):
        self.speed = 0.0
        self._t = None

    def update(self, act, cad, pk, now):
        target = raw_speed(act, cad, pk)
        if self._t is None:
            self.speed = target
        else:
            a = 1.0 - math.exp(-max(0.0, now - self._t) / SMOOTH_TAU)
            self.speed += a * (target - self.speed)
        self._t = now
        return self.speed


class ResultStore:
    """Every finished race, in a JSON list on disk."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._boards = {}
        self._entries = []
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                ok = isinstance(data, list) and all(
                    isinstance(e, dict) and "distance" in e and "time" in e and "who" in e for e in data)
            except (ValueError, OSError):
                ok = False
            if ok:
                self._entries = data
            else:
                try:
                    self.path.rename(self.path.with_name(self.path.name + ".bad-" + time.strftime("%Y%m%d-%H%M%S")))
                except OSError:
                    pass

    def save(self, distance, who, seconds, profile):
        entry = {"who": who or "anon", "distance": int(distance), "time": round(float(seconds), 2),
                 "avg_kmh": round(distance / seconds * 3.6, 2) if seconds > 0 else 0.0,
                 "when": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": [list(p) for p in profile]}
        with self._lock:
            self._entries.append(entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries))
            os.replace(str(tmp), str(self.path))
            self._boards = {}
            rank = 1 + sum(1 for e in self._entries if e["distance"] == entry["distance"] and e["time"] < entry["time"])
        return dict(entry, rank=rank)

    def leaderboard(self, distance, n=LEADERBOARD_N):
        with self._lock:
            key = (distance, n)
            if key not in self._boards:
                rows = sorted((e for e in self._entries if e["distance"] == distance), key=lambda e: e["time"])[:n]
                self._boards[key] = [{k: e[k] for k in ("who", "time", "avg_kmh", "when")} for e in rows]
            return [dict(r) for r in self._boards[key]]

    def champion(self, distance):
        """The fastest finish by anyone at this distance, or None."""
        with self._lock:
            mine = [e for e in self._entries if e["distance"] == distance]
        return min(mine, key=lambda e: e["time"]) if mine else None


class Race:
    """idle -> countdown -> running -> finished; abort() returns to idle without saving."""

    def __init__(self, store, clock=time.monotonic):
        self.store = store
        self._clock = clock
        self.distance = DISTANCES[1]
        self._lock = threading.RLock()
        self._gen = 0
        self._reset()

    def _reset(self):
        self._gen += 1
        self.phase = "idle"
        self.who = ""
        self.t0 = None
        self.dist = 0.0
        self.elapsed = 0.0
        self.speed = 0.0
        self.profile = []
        self.result = None
        self.champion = None
        self._last = None

    def start(self, distance, who=""):
        with self._lock:
            try:
                distance = int(distance)
            except (TypeError, ValueError):
                raise ValueError("distance must be one of %s" % (DISTANCES,))
            if distance not in DISTANCES:
                raise ValueError("distance must be one of %s" % (DISTANCES,))
            self._reset()
            self.distance = distance
            self.who = who or "anon"
            self.champion = self.store.champion(self.distance)
            self.t0 = self._clock() + COUNTDOWN_S
            self.phase = "countdown"

    def abort(self):
        with self._lock:
            self._reset()

    def tick(self, speed, now):
        """Advance with the current smoothed speed (m/s); call every loop pass."""
        pending = None
        with self._lock:
            self.speed = speed
            if self.phase == "countdown":
                if now >= self.t0:
                    self.phase = "running"
                    self._last = self.t0
                    self.tick(speed, now)
            elif self.phase == "running":
                dt = max(0.0, now - self._last)
                self._last = now
                self.dist += speed * dt
                self.elapsed = now - self.t0
                if self.dist >= self.distance:
                    if speed > 0:
                        self.elapsed -= (self.dist - self.distance) / speed
                    self.dist = float(self.distance)
                    self.phase = "finished"
                    self.profile.append((round(self.elapsed, 2), round(speed, 3)))
                    ghost_times = [(n, self.distance / (kmh / 3.6)) for n, kmh in GHOSTS]
                    if self.champion:
                        ghost_times.append(("Champion", self.champion["time"]))
                    beat = [n for n, gt in ghost_times if gt > self.elapsed]
                    place = 1 + sum(1 for _, gt in ghost_times if gt < self.elapsed)
                    record = self.champion is None or self.elapsed < self.champion["time"]
                    pending = (self._gen, self.distance, self.who, self.elapsed, list(self.profile), beat, place, record)
                elif not self.profile or self.elapsed - self.profile[-1][0] >= PROFILE_STEP_S:
                    self.profile.append((round(self.elapsed, 2), round(speed, 3)))
        if pending:
            gen, distance, who, elapsed, profile, beat, place, record = pending
            saved = self.store.save(distance, who, elapsed, profile)
            with self._lock:
                if self._gen == gen:
                    self.result = {"time": saved["time"], "avg_kmh": saved["avg_kmh"], "rank": saved["rank"],
                                   "beat": beat, "place": place, "record": record}
                    if record:
                        self.champion = saved

    def ghosts(self):
        with self._lock:
            return self._ghosts()

    def _ghosts(self):
        t = max(0.0, self.elapsed) if self.phase in ("running", "finished") else 0.0
        out = [{"name": n, "kmh": k, "dist": min(float(self.distance), k / 3.6 * t)} for n, k in GHOSTS]
        if self.champion:
            out.append({"name": "Champion", "who": self.champion["who"], "kmh": self.champion["avg_kmh"],
                        "dist": min(float(self.distance), replay_distance(self.champion["profile"], t))})
        return out

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        now = self._clock()
        return {"phase": self.phase, "distance": self.distance, "who": self.who,
                "countdown": round(max(0.0, self.t0 - now), 1) if self.phase == "countdown" else None,
                "elapsed": round(self.elapsed, 2), "dist": round(self.dist, 1), "speed_kmh": round(self.speed * 3.6, 1),
                "ghosts": self._ghosts(), "field": len(GHOSTS) + (1 if self.champion else 0) + 1,
                "champion": ({k: self.champion[k] for k in ("who", "time", "avg_kmh")} if self.champion else None), "result": self.result, "leaderboard": self.store.leaderboard(self.distance)}
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_race.py -q`
Expected: `6 passed`

- [ ] **Step 4: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python/race.py P6-Gait-Wearable/tests/test_race.py && git commit -m "P6-Gait-Wearable: race.py: speed from cadence and stride-from-impact gated by the classifier, smoothed; race state machine with countdown, ghosts and a best-run replay; JSON result store with leaderboards"
```

### Task 22: race in the pipeline, web routes, race panel on the page

**Files:**
- Modify: `app/python/pipeline.py`, `app/python/main.py`, `app/python/web.py`, `app/python/page.html`, `tests/test_pipeline.py`, `tests/test_web.py`

- [ ] **Step 1: Failing tests**

Append to `tests/test_pipeline.py`:
```python
def test_race_runs_on_the_gait(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    assert g.snapshot()["race"]["phase"] == "idle" and g.snapshot()["speed_kmh"] == 0.0
    g.race.start(100, "ann")
    feed(g, clk, "RUN", 6.0)                       # 2.8 Hz synth run: ~168 spm x stride(3.5 g) 1.85 m = ~5 m/s
    snap = g.snapshot()
    assert snap["speed_kmh"] > 10
    assert snap["race"]["phase"] in ("running", "finished") and snap["race"]["dist"] > 0
    assert len(snap["race"]["ghosts"]) == 3
    g.race.abort()
    assert g.snapshot()["race"]["phase"] == "idle"


def test_race_distance_does_not_grow_while_standing(tmp_path):
    clk = Clock()
    g = make(tmp_path, clk)
    g.race.start(100, "ann")
    feed(g, clk, "STAND", 6.0)
    snap = g.snapshot()
    assert snap["race"]["phase"] == "running" and snap["race"]["dist"] == 0.0 and snap["speed_kmh"] == 0.0
```

Append to `tests/test_web.py`:
```python
def test_race_routes(server):
    s, g = server
    status, _, body = req(s, "POST", "/race/start?distance=400&who=ann")
    snap = json.loads(body)
    assert status == 200 and snap["race"]["phase"] == "countdown" and snap["race"]["distance"] == 400
    assert g.race.who == "ann"
    status, _, body = req(s, "POST", "/race/start?distance=123")
    assert status == 400
    status, _, body = req(s, "POST", "/race/abort")
    assert status == 200 and json.loads(body)["race"]["phase"] == "idle"
```

Run: `.venv/bin/python -m pytest tests/test_pipeline.py tests/test_web.py -q`
Expected: FAIL on the three new tests (`AttributeError: 'Gait' object has no attribute 'race'`, 404s).

- [ ] **Step 2: pipeline.py**

- Import: `from race import Race, ResultStore, SpeedModel`.
- `Gait.__init__` gains a keyword `races_path=None`; after the recorder is stored: `self.speed = SpeedModel()` and
  `self.race = Race(ResultStore(races_path or (Path(recorder.dir) / "races.json")), clock=clock)` (import `Path`).
- `tick()`: the race advances on EVERY call, the classification only every HOP_S. Restructure the top:
```python
    def tick(self, now=None):
        """Advance the race every call; every HOP_S also classify the newest window. Returns True when it classified."""
        now = self._clock() if now is None else now
        self.race.tick(self.speed.speed, now)
        if now < self._next_tick:
            return False
        self._next_tick = now + HOP_S
        ...
```
  and after each state change at the end of the classification path (all three branches), update the speed model:
  `self.speed.update(self.state["act"], self.state["cad"], self.state["pk"], now)` (the two early-return branches
  pass `act` with `cad=None`, which gives 0). Simplest: compute `self._set(...)` as now, then one line before each
  `return True`: `self.speed.update(self.state["act"], self.state["cad"], self.state["pk"], now)`.
- `snapshot()`: add `"speed_kmh": round(self.speed.speed * 3.6, 1)` and `"race": self.race.snapshot()`.

- [ ] **Step 3: web.py**

In `do_POST`, before the `else` 404:
```python
        elif url.path == "/race/start":
            try:
                self.gait.race.start(int(q.get("distance", ["400"])[0]), who)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
        elif url.path == "/race/abort":
            self.gait.race.abort()
```
(`who` is already cleaned above.) Update the module docstring's route list.

- [ ] **Step 4: page.html**

Insert a Race panel between the `<h1>` and the gait verdict, and the JS to drive it. Markup (inside `<main>`):
```html
<section id="race" class="race">
  <div class="row" style="align-items:center">
    <h1 style="margin:0">Race</h1>
    <span class="muted">run on the spot</span>
    <span style="flex:1"></span>
    <button class="sec dist" data-d="100">100 m</button>
    <button class="sec dist" data-d="400">400 m</button>
    <button class="sec dist" data-d="1000">1 km</button>
    <button id="raceStart" onclick="raceStart()">Start</button>
    <button id="raceAbort" class="stop" onclick="post('/race/abort')">Abort</button>
  </div>
  <div class="track" id="track"></div>
  <div class="muted">speed = cadence x stride, stride = 0.6 m + 0.5 m per g of impact above 1 g; WALK capped at 7 km/h, STAND and BIKE do not move</div>
  <div class="row">
    <div class="card"><small>Speed</small><b id="rSpeed">0.0</b> km/h</div>
    <div class="card"><small>Distance</small><b id="rDist">0</b> m</div>
    <div class="card"><small>Time</small><b id="rTime">0.0</b> s</div>
  </div>
  <div id="rCountdown" class="countdown"></div>
  <div id="rResult" class="card result"></div>
  <table id="rBoard" class="board"></table>
</section>
```
CSS additions:
```css
  .race { background: #0d0d0d; border: 1px solid #222; border-radius: 12px; padding: 12px 14px; margin-bottom: 18px; }
  .track { position: relative; height: 150px; background: #161616; border-radius: 8px; margin: 10px 0 4px; overflow: hidden; }
  .lane { position: absolute; left: 0; right: 0; height: 28px; border-bottom: 1px dashed #2a2a2a; }
  .lane .name { position: absolute; left: 8px; top: 6px; font-size: 12px; color: #999; }
  .lane .dot { position: absolute; top: 6px; width: 16px; height: 16px; border-radius: 50%; margin-left: -8px; transition: left 0.1s linear; }
  .lane .time { position: absolute; right: 8px; top: 6px; font-size: 12px; color: #bbb; }
  .finish { position: absolute; top: 0; bottom: 0; right: 6%; width: 2px; background: repeating-linear-gradient(#fff 0 4px, #000 4px 8px); }
  .countdown { font-size: 64px; font-weight: 800; text-align: center; color: #fff; min-height: 0; }
  .result { display: none; margin-top: 8px; }
  .board { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }
  .board td, .board th { padding: 4px 6px; text-align: left; border-bottom: 1px solid #222; color: #ccc; }
  .board th { color: #888; font-weight: 500; text-transform: uppercase; font-size: 11px; }
  .dist.on { background: #0093e7; }
```
JS additions (keep the existing functions):
```js
let raceDistance = 400;
document.querySelectorAll(".dist").forEach(b => b.onclick = () => { raceDistance = +b.dataset.d; document.querySelectorAll(".dist").forEach(x => x.classList.toggle("on", x === b)); });
document.querySelector('.dist[data-d="400"]').classList.add("on");
function raceStart() { post("/race/start?distance=" + raceDistance); }
const ghostColor = {Jogger: "#16ec06", Runner: "#ffde00", Sprinter: "#ff0026", "Your best": "#0093e7"};
function renderRace(r, speedKmh) {
  $("rSpeed").textContent = speedKmh.toFixed(1);
  $("rDist").textContent = Math.round(r.dist);
  $("rTime").textContent = r.elapsed.toFixed(1);
  $("rCountdown").textContent = r.phase === "countdown" ? Math.ceil(r.countdown) : (r.phase === "running" ? "GO" : "");
  $("raceStart").style.display = r.phase === "idle" || r.phase === "finished" ? "" : "none";
  $("raceAbort").style.display = r.phase === "countdown" || r.phase === "running" ? "" : "none";
  const lanes = [{name: "You", dist: r.dist, color: "#fff"}].concat(r.ghosts.map(g => ({name: g.name + " " + g.kmh + " km/h", dist: g.dist, color: ghostColor[g.name] || "#888"})));
  const html = '<div class="finish"></div>' + lanes.map((l, i) => `<div class="lane" style="top:${i * 29}px"><span class="name">${l.name}</span><span class="dot" style="left:${8 + 86 * Math.min(1, l.dist / r.distance)}%;background:${l.color}"></span><span class="time">${Math.round(l.dist)} m</span></div>`).join("");
  if ($("track").dataset.html !== html) { $("track").innerHTML = html; $("track").dataset.html = html; }
  $("track").style.height = (lanes.length * 29 + 4) + "px";
  if (r.result && r.phase === "finished") {
    $("rResult").style.display = "block";
    $("rResult").innerHTML = `<small>${r.who}, ${r.distance} m</small><b>${r.result.time.toFixed(2)} s</b> &nbsp; ${r.result.avg_kmh.toFixed(1)} km/h &nbsp; rank #${r.result.rank}` + (r.result.beat.length ? `<div class="muted">beat ${r.result.beat.join(", ")}</div>` : `<div class="muted">beat nobody this time</div>`);
  } else { $("rResult").style.display = "none"; }
  const rows = r.leaderboard.map((e, i) => `<tr><td>${i + 1}</td><td>${e.who}</td><td>${e.time.toFixed(2)} s</td><td>${e.avg_kmh.toFixed(1)} km/h</td><td class="muted">${e.when}</td></tr>`).join("");
  const board = rows ? `<tr><th>#</th><th>who</th><th>${r.distance} m</th><th>avg</th><th>when</th></tr>` + rows : "";
  if ($("rBoard").dataset.html !== board) { $("rBoard").innerHTML = board; $("rBoard").dataset.html = board; }
}
```
and in `render(s)`: `renderRace(s.race, s.speed_kmh);` as the first line. Session links and the rest unchanged.

- [ ] **Step 5: main.py**

No change needed (`Gait(...)` default `races_path` lands in `data/races.json`). Just confirm `py_compile`.

- [ ] **Step 6: Run everything**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all green (41 + 6 race + 2 pipeline + 1 web = 50).

- [ ] **Step 7: Commit**

```bash
cd ~/ME193 && git add P6-Gait-Wearable/app/python P6-Gait-Wearable/tests && git commit -m "P6-Gait-Wearable: race mode: the pipeline runs a speed model and the race every loop pass, web routes start/abort a race, the page gets a Race panel with distance buttons, lanes with ghosts, countdown, finish card and leaderboard"
```

---

## Revision 5 (2026-10-26): Olympic final, first person, champion ghost (spec Revision 5)

### Task 23: champion ghost, record and place in `race.py`

**Files:** modify `app/python/race.py`, `tests/test_race.py`

- [ ] **Step 1: Tests.** In `tests/test_race.py`:
  - `test_race_runs_countdown_distance_finish_and_saves`: after the finish also assert `s["result"]["place"] == 2`
    (the Sprinter at 22 km/h runs 100 m in 16.4 s, faster than 20 s) and `s["result"]["record"] is True` (first ever),
    and `s["champion"]["who"] == "ann"` once finished.
  - Replace `test_second_race_gets_the_best_ghost_and_abort_resets` with `test_champion_ghost_is_the_fastest_of_anyone`:
    ann finishes 100 m at 4 m/s (25 s); then bob starts 100 m: ghosts are the three fixed ones plus
    `{"name": "Champion", "who": "ann", ...}`; at 2 s into bob's run the Champion's `dist == pytest.approx(8.0)`;
    bob finishes at 5 m/s (20 s): `result["record"] is True` and `result["beat"]` contains `"Champion"`; a third race
    by ann now shows the Champion as bob; `r.abort()` returns to idle; a race at 400 m shows only the three fixed ghosts.
  - Replace `test_anon_has_no_personal_ghost` with `test_anon_can_hold_the_record`: who="" finishes; the next start
    shows a Champion whose `who == "anon"`.
  - Add `test_slower_than_the_champion_is_no_record`: ann 100 m in 20 s, then bob at 4 m/s (25 s): `record is False`,
    `"Champion" not in result["beat"]`, `place == 3` (Sprinter and Champion ahead).
  - Keep every other test.

- [ ] **Step 2: Implementation** (`app/python/race.py`):
  - `ResultStore.champion(distance)`: the entry with the smallest `time` at that distance, or None. Remove `best()`.
  - `Race._reset` sets `self.champion = None` (replacing `self.best`); `start` sets `self.champion = self.store.champion(self.distance)`.
  - `ghosts()`: the fixed three, then when `self.champion`: `{"name": "Champion", "who": champion["who"], "kmh": champion["avg_kmh"], "dist": min(distance, replay_distance(champion["profile"], t))}`.
  - At the finish (under the lock, before the save): `ghost_times = [distance / (kmh / 3.6) for _, kmh in GHOSTS]` plus
    `champion["time"]` when there is one; `beat` = names of those with `time > elapsed` (the champion's name is
    `"Champion"`); `place = 1 + number of ghost times < elapsed`; `record = champion is None or elapsed < champion["time"]`.
    `self.result = {"time", "avg_kmh", "rank", "beat", "place", "record"}`. After the save, if the race is still the
    same one and `record`, set `self.champion = saved` (so the finish snapshot already shows the new champion).
  - `snapshot()` adds `"champion": None or {"who", "time", "avg_kmh"}` and `"field"`: the number of runners (ghosts + 1).

- [ ] **Step 3:** `.venv/bin/python -m pytest tests -q` green; commit as
  `P6-Gait-Wearable: race: one Champion ghost (the fastest finish by anyone at that distance) replaces the personal best, results carry place and record, the record holder becomes the ghost`.

### Task 24: first-person canvas race view on the page

**Files:** modify `app/python/page.html` only.

Replace the `<div class="track" id="track"></div>` with `<canvas id="raceCanvas" class="track"></canvas>` and delete the
`.lane`, `.finish` CSS rules (keep `.track` but make it `height: 360px; width: 100%; display: block;`). In `renderRace`,
delete the lanes/track HTML block and instead store the snapshot: `raceSnap = r; raceSnapAt = performance.now();`.
The result card gains place and record: replace its innerHTML line with
```js
    $("rResult").innerHTML = `<small>${r.who}, ${r.distance} m</small><b>${ordinal(r.result.place)} of ${r.field}</b> &nbsp; ${r.result.time.toFixed(2)} s &nbsp; ${r.result.avg_kmh.toFixed(1)} km/h` + (r.result.record ? `<div style="color:#ffde00;font-weight:700">NEW RECORD: you are the Champion ghost now</div>` : (r.champion ? `<div class="muted">Champion ${r.champion.who} holds ${r.champion.time.toFixed(2)} s</div>` : ""));
```
with `const ordinal = n => n + (["th","st","nd","rd"][(n % 100 > 10 && n % 100 < 14) ? 0 : Math.min(n % 10, 4) % 4] || "th");` hmm, simpler: `const ordinal = n => n + ({1: "st", 2: "nd", 3: "rd"}[n] || "th");`.
Remove the `$("rCountdown")` line (the canvas draws the countdown) and the `#rCountdown` div.

Add this script block after the existing `renderRace` (keep everything else on the page):
```js
// ---- first-person Olympic final ----
const rc = $("raceCanvas"), rctx = rc.getContext("2d");
let raceSnap = null, raceSnapAt = 0;
const LANES = 5, YOU_LANE = 2, VIEW_M = 80, K = 14;
const runnerColor = {Jogger: "#16ec06", Runner: "#ffde00", Sprinter: "#ff0026", Champion: "#0093e7"};
const crowd = Array.from({length: 400}, (_, i) => [Math.random(), Math.random(), ["#c33", "#36c", "#eee", "#fc3", "#3a3"][i % 5]]);
function fitRace() { rc.width = rc.clientWidth * devicePixelRatio; rc.height = 360 * devicePixelRatio; }
window.addEventListener("resize", fitRace); fitRace();
function runnerFigure(ctx, x, y, s, color, phase) {   // feet at (x, y), height ~ 60 * s px
  const h = 60 * s;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = Math.max(1, 5 * s); ctx.lineCap = "round";
  const swing = Math.sin(phase) * 0.6;
  ctx.beginPath(); ctx.moveTo(x, y - h * 0.5); ctx.lineTo(x + Math.sin(swing) * h * 0.3, y - h * 0.1); ctx.stroke();           // leg 1
  ctx.beginPath(); ctx.moveTo(x, y - h * 0.5); ctx.lineTo(x - Math.sin(swing) * h * 0.3, y - h * 0.1); ctx.stroke();           // leg 2
  ctx.beginPath(); ctx.moveTo(x, y - h * 0.5); ctx.lineTo(x, y - h * 0.85); ctx.stroke();                                       // torso
  ctx.beginPath(); ctx.moveTo(x - Math.sin(swing) * h * 0.25, y - h * 0.5); ctx.lineTo(x, y - h * 0.8); ctx.lineTo(x + Math.sin(swing) * h * 0.25, y - h * 0.5); ctx.stroke(); // arms
  ctx.beginPath(); ctx.arc(x, y - h * 0.95, h * 0.1, 0, 7); ctx.fill();                                                          // head
}
function drawRace() {
  requestAnimationFrame(drawRace);
  const r = raceSnap; if (!r || rc.width === 0) return;
  const ctx = rctx, W = rc.width, H = rc.height, dpr = devicePixelRatio;
  const dt = r.phase === "running" ? (performance.now() - raceSnapAt) / 1000 : 0;
  const you = Math.min(r.distance, r.dist + r.speed_kmh / 3.6 * dt);
  const horizon = H * 0.38, vx = W / 2;
  const laneX = i => W * (0.02 + 0.96 * i / LANES);
  const proj = (xb, d) => { d = Math.max(0, d); const t = d / (d + K); return [xb + (vx - xb) * t, H - (H - horizon) * t, 1 - t]; };
  // sky, crowd, track
  ctx.fillStyle = "#0b1020"; ctx.fillRect(0, 0, W, horizon);
  for (const [cx, cy, c] of crowd) { ctx.fillStyle = c; ctx.fillRect(cx * W, horizon * (0.45 + 0.55 * cy), 3 * dpr, 3 * dpr); }
  ctx.fillStyle = "#8b2f1f"; ctx.fillRect(0, horizon, W, H - horizon);
  ctx.strokeStyle = "#f4f4f4"; ctx.lineWidth = 2 * dpr;
  for (let i = 0; i <= LANES; i++) { const [x1, y1] = proj(laneX(i), 0), [x2, y2] = proj(laneX(i), VIEW_M); ctx.beginPath(); ctx.moveTo(laneX(i), H); ctx.lineTo(x2, y2); ctx.stroke(); }
  // 10 m marks scrolling towards you
  ctx.font = `${12 * dpr}px system-ui`; ctx.fillStyle = "#ddd"; ctx.textAlign = "left";
  for (let m = Math.floor(you / 10) * 10; m <= you + VIEW_M; m += 10) {
    const d = m - you; if (d < 0) continue;
    const [xl, y] = proj(laneX(0), d), [xr] = proj(laneX(LANES), d);
    ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke();
    if (m % 50 === 0) ctx.fillText(m + " m", xr + 4 * dpr, y);
  }
  // finish line
  const df = r.distance - you;
  if (df >= 0 && df <= VIEW_M) {
    const [xl, y, s] = proj(laneX(0), df), [xr] = proj(laneX(LANES), df);
    ctx.lineWidth = Math.max(2, 10 * s) * dpr; ctx.strokeStyle = "#fff"; ctx.setLineDash([8 * s * dpr, 8 * s * dpr]);
    ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.font = `bold ${Math.max(10, 28 * s) * dpr}px system-ui`; ctx.fillText("FINISH", vx, y - 14 * s * dpr);
  }
  // competitors in the other lanes, drawn far to near
  const lanes = [0, 1, 3, 4];
  const runners = r.ghosts.map((g, i) => ({name: g.name, who: g.who, lane: lanes[i % lanes.length], dist: Math.min(r.distance, g.dist + g.kmh / 3.6 * dt), kmh: g.kmh}));
  runners.sort((a, b) => b.dist - a.dist);
  for (const g of runners) {
    const gap = g.dist - you;
    if (gap < -2 || gap > VIEW_M) continue;
    const xb = (laneX(g.lane) + laneX(g.lane + 1)) / 2, [x, y, s] = proj(xb, gap);
    runnerFigure(ctx, x, y, s * dpr, runnerColor[g.name] || "#aaa", performance.now() / 1000 * g.kmh / 3.6 * 3);
    ctx.fillStyle = runnerColor[g.name] || "#aaa"; ctx.textAlign = "center"; ctx.font = `${Math.max(9, 12 * s) * dpr}px system-ui`;
    ctx.fillText(g.name === "Champion" ? "Champion " + (g.who || "") : g.name, x, y - 66 * s * dpr);
  }
  // you, bottom centre, legs driven by cadence
  const cad = (raceSnap.cadence || 0);
  runnerFigure(ctx, vx, H - 6 * dpr, 1.15 * dpr, "#fff", performance.now() / 1000 * Math.PI * 2 * (cad / 60));
  // HUD
  ctx.textAlign = "left"; ctx.fillStyle = "#fff"; ctx.font = `bold ${14 * dpr}px system-ui`;
  ctx.fillText(`OLYMPIC FINAL  ${r.distance} m`, 10 * dpr, 20 * dpr);
  const order = [{name: "YOU", dist: you}].concat(runners).sort((a, b) => b.dist - a.dist);
  ctx.font = `${12 * dpr}px system-ui`;
  order.forEach((o, i) => { ctx.fillStyle = o.name === "YOU" ? "#fff" : (runnerColor[o.name] || "#aaa"); ctx.fillText(`${i + 1}. ${o.name === "Champion" ? "Champion " + (o.who || "") : o.name}  ${Math.round(o.dist)} m`, 10 * dpr, (40 + 16 * i) * dpr); });
  ctx.textAlign = "right"; ctx.fillStyle = "#fff"; ctx.font = `bold ${22 * dpr}px system-ui`;
  ctx.fillText((r.elapsed + dt).toFixed(1) + " s", W - 10 * dpr, 26 * dpr);
  ctx.font = `${14 * dpr}px system-ui`; ctx.fillText(r.speed_kmh.toFixed(1) + " km/h", W - 10 * dpr, 46 * dpr);
  // countdown / go / finish
  const big = r.phase === "countdown" ? String(Math.ceil(r.countdown)) : (r.phase === "running" && r.elapsed + dt < 1.5 ? "GO!" : (r.phase === "finished" ? "FINISH" : (r.phase === "idle" ? "" : "")));
  if (big) { ctx.textAlign = "center"; ctx.fillStyle = "#fff"; ctx.font = `bold ${72 * dpr}px system-ui`; ctx.fillText(big, vx, H * 0.6); }
  if (r.phase === "idle") { ctx.textAlign = "center"; ctx.fillStyle = "#ddd"; ctx.font = `${16 * dpr}px system-ui`; ctx.fillText("pick a distance and press Start", vx, H * 0.6); }
}
drawRace();
```
`raceSnap.cadence` comes from the main snapshot: in `render(s)` set `s.race.cadence = s.cad || 0;` before calling `renderRace(s.race, s.speed_kmh)`.

- [ ] **Step 2:** sanity: `.venv/bin/python -c "import html.parser; html.parser.HTMLParser().feed(open('app/python/page.html').read()); print('ok')"`, the suite stays green (`tests/test_web.py` only checks for `EventSource`), then commit as
  `P6-Gait-Wearable: race view is a first-person Olympic final on a canvas: perspective lanes, crowd, scrolling marks, approaching finish line, competitors as animated figures in the lanes beside you, HUD with positions and time, the Champion ghost named`.


### Task 25 (done in-session, 2026-10-26): race view in the SEGA London 2012 look

Spec Revision 6. `page.html`'s race script block was rewritten twice after Task 24 (doodle side view, then the chase-camera stadium view) directly in the file; the Task 11 `page.html` block above is synced to the final file, so read the file, not Tasks 22-24, for the current view.

### Task 26 (done in-session, 2026-10-26): realism pass

A design agent rebuilt the race scene (commits 1379be7, 29de2d8): world-space 400 m oval with six 1.22 m lanes and 36.5 m bends turning LEFT (anticlockwise, infield on the left), pinhole chase camera 6 m behind and 2.2 m up, offscreen pre-rendered stadium/crowd/track grain, skeletal back-view runners with bibs and a stride cycle, sun from front-left with limb shadows, LED scoreboard, glossy power bar at 0.60 width with tag collision avoidance, medal-banded finish panel. The Task 11 `page.html` block above is synced to the final file.
