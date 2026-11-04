# P7: Finger Lock

A safe that opens for a three-digit finger code. Push the button, the LEDs come on, hold up three different numbers
of fingers (1 to 5) one after the other under the webcam, and the latch opens or stays shut. The number of fingers
comes from **FingerNet**, a convolutional neural network built from scratch in PyTorch and trained on my own hand with
the USB webcam; the trained weights then run on the **Arduino UNO Q** in plain numpy with the same webcam.
ME 193 midterm, Tufts, November 2026.

| | |
|---|---|
| Net | 3 x (conv 3x3 → ReLU → max-pool 2) with 16 / 32 / 64 filters, linear 4096 → 64 → 6, cross-entropy; 285,894 weights |
| Input | the desk square, grayscale 64 x 64 |
| Decision | same class ≥ 0.85 confidence for 8 frames in a row = a digit; hand must leave before the next; 3 digits vs the code |
| Board | UNO Q: webcam → numpy net → policy → 13x8 matrix + 2 LEDs + servo latch; page on port 8080 |
| Results | `docs/results.md` (written by the training script), `docs/comparison.md` (UNO Q vs Mac) |

## What's where

| Path | What |
|---|---|
| `lock/` | Shared by Mac and board: `config.py`, `preprocess.py` (ROI crop → 64x64), `cnn_np.py` (numpy inference), `policy.py` (the lock state machine), `matrix.py` (13x8 pictures), `engine.py`, `timing.py` |
| `train/` | `model.py` (FingerNet), `dataset.py` (session split + augmentation), `train.py`, `plots.py`, `export.py` (→ JSON), `synth.py` (generated hands for pipeline checks) |
| `tools/` | `capture.py` (take the pictures), `lock_app.py` (the lock on the Mac, with saliency), `bench.py` (timings) |
| `app/` | The UNO Q App Lab app: `python/main.py`, `python/web.py` + `page.html`, `sketch/sketch.ino` |
| `scripts/` | `deploy.sh` (USB cable), `compile.sh` (sketch on the Mac), `board_stats.sh` (pull the board's timings) |
| `models/` | `fingers.pt` (PyTorch), `fingers.json` (for the board), `meta.json` |
| `docs/movie/` | The stop-action explainer: `frames.py`, `build.sh`, `script.md` (voice-over) |
| `docs/` | `notion.md` (page draft with the four midterm sections), `results.md`, `comparison.md`, `figures/`, `superpowers/` (spec + plan) |
| `tests/` | 46 pytest tests |

## Run it

Setup once: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Tests: `.venv/bin/python -m pytest tests -q`.

```bash
.venv/bin/python tools/capture.py --list            # find the USB webcam's index; put it in config.json ("camera")
.venv/bin/python tools/capture.py                   # 0-5 = label, r = burst, arrows = move the square, w = save it
.venv/bin/python train/train.py --ablate            # train (with and without augmentation), figures, results.md, export
.venv/bin/python tools/lock_app.py                  # the lock on the Mac: b = button, s = saliency, q = quit
.venv/bin/python tools/bench.py --camera 1          # Mac timings -> docs/bench_mac.json
scripts/deploy.sh                                   # board on the USB cable; app starts on power from then on
scripts/board_stats.sh                              # board timings -> docs/bench_board.json
docs/movie/build.sh                                 # the explainer frames + movie_silent.mp4
```

Taking pictures: the webcam on a stand looking down at the desk, a square drawn on the desk inside the green box
(move the box with the arrow keys and press `w` so the board gets the same square). About 150 pictures per class
over three or more sessions (different light, slightly different hand positions); class 0 is the empty square and
a fist. Training holds out whole sessions, so the accuracy in `docs/results.md` is for pictures the net never saw.

## Wiring (UNO Q)

| Part | Pin | Note |
|---|---|---|
| Push button | D2 to GND | internal pull-up, 30 ms debounce |
| Red LED | D3 → 220 Ω → GND | denied / lockout (blinks) |
| Green LED | D4 → 220 Ω → GND | unlocked; both LEDs on while listening |
| Servo (latch) | signal D5, 5V, GND | open 2.0 ms pulse for `servo_open_ms`, closed 1.0 ms; change `kOpenUs` in the sketch to suit the latch |
| USB webcam | the USB port (hub if needed) | the same webcam as on the Mac; `board_camera` in config.json picks it (`usb:0`) |
| 13x8 LED matrix | built in | the digits entered, hold-progress bar, tick / cross / skull |

Page over the cable: `adb forward tcp:8080 tcp:8080` then http://localhost:8080 (state, the crop the net sees, the
probabilities, a button for testing without wiring one). Logs: `adb shell tail -f '~/ArduinoApps/finger-lock/data/app.log'`.

## The lock policy

`lock/policy.py`, fed one probability vector per frame. IDLE → button → LISTENING (LEDs on, 20 s to enter). A digit
is accepted when the same class 1..5 is the arg-max with confidence ≥ 0.85 for 8 consecutive frames (0.8 s at 10 fps);
then the hand must release (3 frames of no hand or another class) before the next digit can start, so a held hand is
one digit. Three digits are compared with the code in order: UNLOCKED (servo open 5 s, green) or DENIED (red, 2 s).
Three failures in a row → LOCKOUT 30 s (button ignored, red blinks, the failing hand's picture is saved to
`data/intruders/`). The code must be three different digits in 1..5; `config.json` is refused otherwise.

## Fun parts not done in class

- **Saliency overlay** in `tools/lock_app.py` (`s`): the gradient of the winning score with respect to the pixels,
  drawn as a heat map on the live picture, shows which pixels the net is counting.
- **Intruder log + skull**: after the third wrong code the matrix shows a skull, the button is dead for 30 s, and the
  hand that failed is saved for the owner.
- The explainer is a **stop-action movie** drawn from the real weights and the real activations of my hand.

## The camera hood (SolidCraft)

`enclosure/camera_box.py` writes a [SolidCraft](https://github.com/tangency-labs/solidcraft) script and runs it headless:
`enclosure/out/camera_box.sc` (open it in SolidCraft, every number is a parameter), `.step`, `.stl` and three renders.
A 328 x 328 mm hood, open at the front and the bottom, roof 304 mm up; the webcam (70 x 18.5 x 25 mm) drops lens-down
into a snug pocket in a boss on the roof and looks through a 62 x 14 window in the pocket floor, strapped down through
two slots; a 1/4-20 hole in front of the boss is the alternative for a camera with a tripod thread. The base plate
carries the engraved 177 mm square (the net's region of interest at that height for a 70° lens) centred under the
lens; `config.json`'s `roi` is centred to match. Cable notch at the top of the back wall. Change `P` in the script,
or the parameters in SolidCraft, to resize it. To let Claude drive SolidCraft directly in later sessions:
`claude mcp add solidcraft -- /Applications/SolidCraft.app/Contents/MacOS/solidcraft-cli mcp`.
