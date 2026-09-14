# lego-pose-car

Drive a LEGO Education car by moving your arms in front of a webcam.

MediaPipe reads your pose from the camera, each arm's height becomes a speed for
the motor on that side, and those speeds go to a LEGO Double Motor over
Bluetooth Low Energy.

Built for ME 193 – AI & Robotics.

## The control scheme

Tank drive: **each arm controls the wheel on its own side.**

| Your arm | That wheel |
|---|---|
| straight up | full forward |
| held out, shoulder height (T-pose) | stopped |
| down at your side | full reverse |

So both arms up is forward, both arms down is reverse, and one up / one down
spins the car in place. Because the car drives away from you, its left wheel and
your left arm are on the same side — no mental flip needed.

The preview window is mirrored, so it behaves like a mirror, and the two bars at
the edges of the frame show the speed each wheel is being given.

## Run it

```bash
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt

python pose_car.py --no-motor    # vision only, no hardware needed
python pose_car.py               # connect to a motor and drive
```

Keys, in the preview window:

- **`d`** — arm / disarm the motors. **Starts disarmed**, so you can see what the
  car *would* do before it does it. Worth having when the car is pointed at a
  door.
- **`q`** or **Esc** — quit (stops the motor and drops the BLE link on the way out).

With several cars in one room, a bare `connect()` grabs whichever motor answers
the scan first — possibly someone else's. Use the Connection Card that came with
the motor to name yours:

```bash
python pose_car.py --card-color azure --card-serial 3683
```

Other flags: `--camera N` to pick a different camera.

On macOS the first run triggers Camera and Bluetooth permission prompts for your
terminal — allow both.

## How it works

One loop, roughly 30 times a second:

1. **Grab a frame** from the webcam with OpenCV and mirror it.
2. **Find the body.** MediaPipe's `PoseLandmarker` returns 33 landmarks, each
   with an `x`, `y` (as a fraction of the frame) and a `visibility` score. Four
   of them matter here: the two shoulders and the two wrists.
3. **Turn arms into speeds.** For each arm, measure how far the wrist sits above
   its shoulder and divide by the distance between the shoulders:

   ```
   offset = (shoulder_y - wrist_y) / shoulder_width
   ```

   Dividing by shoulder width is what makes this work at any distance from the
   camera: standing closer scales every pixel measurement up, but their *ratio*
   holds. A dead zone near zero keeps a level arm from creeping, and the result
   is clamped to ±70%.
4. **Smooth it.** An exponential moving average over the last few frames, so a
   single bad landmark doesn't jolt the car.
5. **Send it.** `movement_move_tank(left, right)` over BLE.

Each step also has a way to say "stop": a landmark below 0.5 visibility, a
driver standing side-on to the camera (their shoulders overlap, which would
otherwise amplify noise into full throttle), or no pose at all for 0.4s all
drive the speeds to zero.

## Questions

### 1. How is Python talking to the LEGO hardware?

Over **Bluetooth Low Energy — there is no cable and no custom firmware.** The
motor runs LEGO's stock firmware and my laptop acts as the BLE central.

The stack, top to bottom:

```
pose_car.py
  └─ legoeducation 1.1.1      LEGO's own Python package (PyPI)
      └─ bleak 3.x            cross-platform BLE library
          └─ CoreBluetooth    macOS's system Bluetooth stack
              └─ ~~~ 2.4 GHz ~~~  LEGO Double Motor
```

`legoeducation` scans for advertisements carrying LEGO's service UUID
`0000FD02-…`, and that service has exactly two characteristics it uses:

| Characteristic | Direction |
|---|---|
| `0000FD02-0001-…` | write — laptop → motor (commands) |
| `0000FD02-0002-…` | notify — motor → laptop (acknowledgements, telemetry) |

Commands are not text. Each one is serialized into a small binary RPC message:
`movement_move_tank(60, -20)` becomes a `MovementMoveTankCommand` struct that is
written to the write characteristic, and the motor answers on the notify
characteristic with a status and with live telemetry (position, speed, motor
state).

Optionally the scan is filtered by the colour and serial printed on the
Connection Card in the box, which is how you pick one specific motor out of a
room full of them.

### 2. Is it synchronous or asynchronous?

**Both, at different layers — and the interesting part is where the boundary
sits.**

**My program is synchronous.** It is a single-threaded loop: grab a frame, run
inference, compute speeds, send them, repeat. I deliberately used MediaPipe's
`RunningMode.VIDEO`, whose `detect_for_video()` call blocks until the result is
ready, rather than `RunningMode.LIVE_STREAM`, which hands results back through a
callback on another thread. Blocking here is the right call: the results arrive
in frame order, and a pose can never overtake a stale motor command. Inference
on the lite model is only a few milliseconds, so it isn't the bottleneck.

**The library underneath is asynchronous.** `legoeducation` exposes a plain
synchronous API but is async internally. On import it starts a persistent
`asyncio` event loop in a background daemon thread; every public method wraps
its coroutine in `asyncio.run_coroutine_threadsafe()` and blocks the calling
thread on the resulting future. Running the loop on its own thread is what lets
BLE notifications from the motor keep being processed even while the main thread
is busy.

**So the one decision I actually had to make was where to block**, and every
motor command takes a `blocking=` argument that decides it:

- `blocking=True` (the default) waits for the motor's acknowledgement to come
  back over BLE before returning.
- `blocking=False` hands the command off and returns immediately.

I use **`blocking=False`**. A BLE round trip is tens of milliseconds; waiting for
one on every frame would drag the camera loop down and make the controls feel
laggy. I don't need the acknowledgement — the next frame is about to send a new
speed anyway.

The last piece is rate limiting. The camera produces ~30 poses a second, which is
more than BLE wants to carry, so `Car.drive()` only sends when at least 80 ms
have passed *and* a speed actually changed by more than a few percent — about 12
writes a second. The one exception is a stop, which always goes out immediately
and never waits its turn.

### 3. How did you train it, and what are its limitations?

**I didn't train anything, and that's the honest answer.** The pose model is
`pose_landmarker_lite.task` — a pre-trained model Google ships as part of
MediaPipe (the BlazePose GHUM family), trained on large human-pose datasets and
used here exactly as downloaded. No fine-tuning, no dataset of my own, no
training run.

What I wrote instead of a model is **geometry plus tuning**. The step from "33
body landmarks" to "two motor speeds" is hand-written trigonometry, and the
constants in it — the dead zone, what counts as a fully raised arm, the
smoothing weight, the send interval — were set by trying them and watching the
bars move. That is tuning, not learning: nothing adapts, and the numbers are the
same every run.

That choice is defensible for a gesture this simple. "Is the wrist above the
shoulder" is a comparison, not a classification problem, and a trained
classifier here would be strictly worse: it would need collected and labelled
data, it would only output discrete classes where I want a continuous speed, and
it would fail in ways that are much harder to debug than a ratio I can print.

**Limitations:**

*From the pre-trained model:*

- It knows human bodies in general and nothing about *me* in particular. There is
  no calibration step, so a person with a different build drives slightly
  differently.
- `lite` is the least accurate of the three model sizes, and wrists are among
  the noisiest landmarks it produces. I traded accuracy for frame rate.
- It degrades in low light, with baggy sleeves, and when arms cross the torso.

*From my mapping:*

- **Front-facing only.** I use 2D image coordinates, so an arm pointed toward or
  away from the camera looks identical to one hanging straight down. MediaPipe
  does estimate a `z` per landmark, but it is far less reliable than `x`/`y`, so
  I ignore it. A driver standing side-on is rejected outright rather than
  guessed at.
- **One driver.** `num_poses=1`, with no identity tracking. If someone walks
  behind me the detector can latch onto them instead, and the car will follow
  whoever it picked.
- **Smoothing costs latency.** The moving average adds roughly 3–4 frames
  (~100 ms) of lag, on top of the BLE send interval. Good for stability, bad for
  quick corrections.
- **The dead zone costs fine control.** Speeds near zero are unreachable by
  design — you get stop, or you get real movement.
- **It's open-loop.** The car reports its motor positions back, but I don't use
  them. Nothing corrects for a wheel slipping, a battery draining, or the car
  drifting off a straight line. It does what you say, not what you meant.
- **Range.** Drive out of Bluetooth range and commands simply stop landing. The
  motor keeps doing whatever it was last told, which is why `d` (disarm) is worth
  reaching for before that happens.

## Note on versions

The pinned versions in `requirements.txt` are not arbitrary — see the comment at
the top of that file. Briefly: mediapipe 1.x crashes on Apple silicon inside the
pose graph (`DrishtiMetalHelper … Service is unavailable`), so this is held at
0.10.21, which in turn pins `numpy<2`, which forces OpenCV back below 4.11.

`pose_landmarker_lite.task` (5.5 MB) is committed alongside the code on purpose,
so a demo doesn't depend on having internet to download a model. To refetch it:

```bash
curl -L -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```
