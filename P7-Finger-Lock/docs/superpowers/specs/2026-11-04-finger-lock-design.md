# Finger Lock: design (full-scale plan)

ME 193 midterm, Tufts, November 2026. A safe that opens for a three-digit finger code read by a convolutional neural
network, trained in PyTorch on the Mac and run on the Arduino UNO Q.

## 1. What it has to do (from the assignment)

1. Walk up, **push a button**: the LEDs come on and the lock starts listening.
2. Show **three different numbers (1 to 5) with your fingers** under the camera, one after the other.
3. The lock **opens** (servo swings the latch, green LED) if the three digits are the code, otherwise it **stays locked**
   (red LED).
4. The number of fingers comes from a **PyTorch neural net built from scratch** on the Mac and the user's **new USB
   webcam** (never the phone or the laptop camera); the trained model is then **moved to the UNO Q** and run there with
   the same webcam.
5. Deliverables: Notion page (video, GitHub, intro, explanations), a **stop-action movie** of how the net works with the
   user's own voice-over, something **fun that was not done in class**, four midterm sections (lock policy,
   limitations, how the net works, UNO Q vs computer) and the usual reflection questions.

Point 1 of the assignment (the V-data gradient-descent section) is out of scope by the user's instruction.

## 2. The system

```
                 Mac (training + demo)                          UNO Q (the safe)
 ┌──────────────────────────────────────┐        ┌─────────────────────────────────────────────┐
 │ tools/capture.py  → data/raw/<n>/*.png│        │ Linux side (App Lab container, Python 3.13) │
 │ train/train.py    → models/fingers.pt │ deploy │  app/python/main.py                         │
 │ train/export.py   → models/fingers.json│ ─────▶ │   Camera("usb:0") ─▶ lock/preprocess        │
 │ tools/lock_app.py (same policy, torch)│        │   ─▶ lock/cnn_np (numpy) ─▶ lock/policy     │
 └──────────────────────────────────────┘        │   ─▶ Bridge.call draw / leds / unlock       │
                                                  │   web page :8080 (state, ROI picture, stats)│
                                                  ├─────────────────────────────────────────────┤
                                                  │ MCU side (sketch.ino)                       │
                                                  │  button D2 ─▶ Bridge.notify("button")       │
                                                  │  provide draw (13x8 matrix), leds, unlock   │
                                                  │  red LED D3, green LED D4, servo D5         │
                                                  └─────────────────────────────────────────────┘
```

One shared Python package, `lock/`, holds everything that must behave identically on both machines: the preprocessing
(ROI crop, grayscale, 64x64), the numpy inference engine, the lock policy and the LED-matrix renderer. It is numpy +
stdlib only (OpenCV when present) because the board has no PyTorch. The Mac uses the same package, so the policy the
user demonstrates on the laptop is the policy on the safe.

## 3. Camera, region of interest, data

- The webcam sits on a stand at a fixed height looking down at the desk; a square drawn on the desk marks where the
  hand goes (the instructor's hint). `config.json` holds the **ROI** as fractions of the frame: `roi: [x, y, size]`
  so the crop is the same on both machines and the capture tool draws it on screen.
- The crop is saved at 128x128 colour PNG in `data/raw/<label>/<session>-<n>.png`, label 0..5 (0 = no hand or a
  fist). The session stamp is the capture run, so validation can hold out whole sessions (an honest number: the same
  session's neighbouring frames are near-duplicates).
- Training input: grayscale 64x64, scaled to 0..1, then standardised with the training mean and std (stored with the
  weights). Grayscale keeps the model small and the board fast; lighting robustness comes from augmentation.
- `tools/capture.py` keys: `0`..`5` pick the label, `r` toggles a 5 per second burst, `c` one shot, `q` quits;
  `--list` prints the camera indices so the USB webcam can be told from the laptop camera.

## 4. The neural net (FingerNet)

| stage | shape | parameters |
|---|---|---|
| input | 1 x 64 x 64 | |
| conv 3x3, 16 filters, pad 1, ReLU, max-pool 2 | 16 x 32 x 32 | 160 |
| conv 3x3, 32 filters, pad 1, ReLU, max-pool 2 | 32 x 16 x 16 | 4,640 |
| conv 3x3, 64 filters, pad 1, ReLU, max-pool 2 | 64 x 8 x 8 | 18,496 |
| flatten, linear 4096 → 64, ReLU, dropout 0.3 | 64 | 262,208 |
| linear 64 → 6 | 6 scores | 390 |

Loss: cross entropy over the six classes (0..5 fingers). Optimiser: Adam, lr 1e-3, 30 epochs, batch 64, cosine decay.
Augmentation (on the fly, OpenCV warps): rotation ±20°, shift ±8 px, scale 0.85..1.15, horizontal flip, brightness
±25 %, contrast 0.8..1.25, mild Gaussian noise. `train.py --no-aug` trains the same net without it so the write-up can
show the difference the instructor's hint makes.

Outputs of a training run: `models/fingers.pt`, `models/fingers.json` (plain weights for the board),
`docs/figures/train_curves.png` (loss and accuracy per epoch, train and validation), `docs/figures/confusion.png`,
`docs/figures/samples.png`, `docs/figures/augment.png`, `docs/results.md` (the numbers), `models/meta.json` (dataset
sizes, split, accuracy, date).

Inference on the board: `lock/cnn_np.py` reads the JSON and runs conv (im2col matmul), ReLU, max-pool, linear in
numpy; a test checks it matches PyTorch to 1e-4 on random inputs and on real samples.

## 5. The lock policy (midterm section 1)

A state machine in `lock/policy.py`, fed one probability vector per camera frame (about 10 per second):

- **IDLE**: LEDs off, matrix dim. Button → LISTENING.
- **LISTENING** (LEDs on, matrix shows three empty slots and a countdown bar). A digit is **accepted** when the same
  class 1..5 is the arg-max with confidence ≥ 0.85 for `hold_frames` consecutive frames (8 frames ≈ 0.8 s). After an
  acceptance the hand must **release**: `gap_frames` (3) frames of class 0 or of a different class before the next
  digit can start, so a hand held still is one digit, not three. Each accepted digit appears on the matrix. The
  entry times out after 20 s → IDLE.
- **Three digits in** → compare with the code in order. Match → **UNLOCKED**: servo to the open angle, green LED,
  tick on the matrix for `unlock_s` (5 s), then IDLE and the servo closes. No match → **DENIED**: red LED and a cross
  for 2 s, failed attempts + 1, then IDLE.
- `max_attempts` (3) failures in a row → **LOCKOUT** for 30 s: the button is ignored and the red LED blinks; a success
  resets the counter.
- The code must be three *different* digits in 1..5 (validated when the config loads); repeated or zero digits are
  rejected at load time.

The policy never looks at the raw image; it only sees probabilities, so the Mac demo, the tests and the board run the
exact same decisions.

## 6. Board app

- `app/python/main.py`: opens the webcam (`arduino.app_peripherals.camera.Camera("usb:0")`, 640x480, 10 fps), crops
  the ROI, runs the numpy net, feeds the policy, and drives the MCU with `Bridge.call("draw", bytes)`,
  `Bridge.call("leds", mask)`, `Bridge.call("unlock", ms)`. `Bridge.provide("button", ...)` receives presses.
  It keeps per-frame timings (capture, preprocess, inference, total) and serves them with the state and the latest
  ROI picture on port 8080 (`/`, `/state`, `/roi.jpg`, `/stats`), reachable over `adb forward` or the hotspot.
- `app/sketch/sketch.ino`: button on D2 (`INPUT_PULLUP`, 30 ms debounce) → `Bridge.notify("button", 1)`; `draw`
  puts 104 brightness bytes on the 13x8 matrix (3-bit grayscale, as the P5 scoreboard sketch does); `leds(mask)` sets
  D3 (red) and D4 (green); `unlock(ms)` drives a hobby servo on D5 with a software 50 Hz pulse train to the open angle
  for `ms` milliseconds, then back to closed. No external libraries beyond the Bridge.
- `scripts/deploy.sh` stages `app/`, `lock/`, `models/fingers.json` and `config.json` and pushes them over the USB
  cable with App Lab's adb (the P6 pattern); `scripts/compile.sh` compiles the sketch on the Mac in ~20 s.

## 7. Mac demo app

`tools/lock_app.py`: webcam window with the ROI box, the live class probabilities as bars, the matrix preview, two
LED dots, the servo state and the policy state. `b` is the button, `q` quits. `--engine torch|numpy` picks the engine
(numpy proves the board engine on the Mac). It logs the same timings as the board to `docs/bench_mac.json`
(`tools/bench.py` does it headless for the comparison table).

**Fun thing not done in class**: `s` toggles a live **saliency overlay**, the gradient of the winning score with
respect to the pixels, so you can see which pixels the net is counting; and the lock keeps an **intruder log**: after
the third failed attempt it saves the hand picture that failed (`data/intruders/`) and shows a skull on the matrix.

## 8. The stop-action movie (midterm section 3)

`docs/movie/frames.py` draws numbered PNG frames in a hand-drawn (xkcd/Comic Sans) style at 3 frames per second, using
the real trained weights and a real sample: image → pixel numbers → a 3x3 kernel sliding one step per frame →
ReLU → 2x2 max-pool → the three-layer stack → flatten → linear → softmax → cross entropy → gradient descent (a ball in a
bowl plus the real loss curve) → back-propagation arrows → augmentation → the lock policy. `docs/movie/build.sh` turns
the frames into `movie_silent.mp4`; `docs/movie/script.md` is the voice-over, one paragraph per scene with the frame
range, for the user to record over the playing movie (Zoom or QuickTime).

## 9. Write-up (midterm sections and template questions)

`docs/notion.md` is the page draft: intro, the setup picture list, how it works, the four midterm sections (lock
policy; at least four limitations; how the net works pointing at the movie; UNO Q vs computer with the measured
table), the fun part, the GitHub link, and the template questions left as `(your words)` placeholders. `README.md`
is the GitHub front page with wiring, commands and results.

## 10. Verification

- `tests/` (pytest): policy state machine (accept, release, timeout, deny, lockout, config validation), preprocessing
  (ROI maths, output shape and range), numpy vs torch parity, export round trip, matrix glyphs, synthetic dataset,
  a one-epoch training smoke test on synthetic data, web handlers, config loading.
- `train/train.py --synth` trains on generated hands end to end (figures, results, JSON) so the whole pipeline is proven
  before the first real picture is taken.
- The sketch compiles with `scripts/compile.sh`.
- What only the user can do: capture real sessions, train, deploy, measure the board, film.

## 11. Limitations (to feed midterm section 2)

Lighting and skin/background contrast; only the trained hand (one person, one camera height); the fixed ROI (a hand
outside the square is class 0); 3 vs 4 fingers at a grazing angle; a photo of a hand fools it (no liveness); a 10 fps
board loop means fast hand changes are missed; the code space is small (5·4·3 = 60 codes) so the lockout matters; the
servo latch is only as strong as the box.
