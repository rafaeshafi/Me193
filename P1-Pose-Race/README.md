# P1: Pose Race

**Drive a LEGO Education car by moving your arms in front of a webcam.**

ME 193 – AI & Robotics, Project 1.

A webcam watches you. MediaPipe turns each frame into a skeleton of 33 body
landmarks. Your arm positions become a pair of motor speeds, and those speeds
go to a LEGO Double Motor over Bluetooth Low Energy — no cable, no custom
firmware. Raise both arms and the car drives away from you; drop one and it
spins. The whole loop runs about thirty times a second on a laptop CPU.

The project is deliberately two systems in one, because the interesting
question in this assignment is *what actually needs to be learned*:

- A **hand-written geometric mapping** — trigonometry on your arm angles, no
  training data, smooth continuous speed control.
- A **classifier you train yourself** — you record examples of each pose, it
  learns to tell them apart, and the car drives from its predictions.

Both drive the same hardware and you switch with one flag, so they can be
compared directly. The write-up in [Questions](#questions) covers how Python
reaches the hardware, where the synchronous/asynchronous boundary sits, and
what was and was not trained.

## Contents

| File | What it is |
|---|---|
| [`pose_car.py`](pose_car.py) | The driving program — both control modes, Bluetooth, the preview window |
| [`pose_features.py`](pose_features.py) | Shared: landmark indices, the feature vector, the pose detector |
| [`collect_poses.py`](collect_poses.py) | Records labelled pose examples to `pose_data.csv` |
| [`train_poses.py`](train_poses.py) | Trains on that CSV, reports results, writes `pose_model.joblib` |
| [`requirements.txt`](requirements.txt) | Pinned dependencies (the pins matter — see the bottom of this file) |
| `pose_landmarker_lite.task` | Google's pre-trained pose model, committed so a demo needs no internet |

## Two ways to steer it

You can switch between them with a flag:

| Mode | How it decides | Flag |
|---|---|---|
| **Geometry** (default) | Hand-written trigonometry on your arm angles. Smooth and continuous. | *(none)* |
| **Trained classifier** | A model trained on pose examples you recorded of yourself. | `--model` |

## The control scheme

**Geometry mode** is tank drive: **each arm controls the wheel on its own side.**

| Your arm | That wheel |
|---|---|
| straight up | full forward |
| held out, shoulder height (T-pose) | stopped |
| down at your side | full reverse |

Both arms up is forward, both down is reverse, and one up / one down spins the
car in place. Because the car drives away from you, its left wheel and your left
arm are on the same side — no mental flip needed.

**Classifier mode** recognises whichever poses you trained it on, and applies a
fixed speed for each — by default `forward`, `back`, `left`, `right` and `stop`,
listed in `CLASS_SPEEDS` in [`pose_features.py`](pose_features.py).

The preview window is mirrored so it behaves like a mirror, and the bars at the
edges show the speed each wheel is being given.

## Run it

```bash
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt

python pose_car.py --no-motor    # vision only, no hardware needed
python pose_car.py               # connect to a motor and drive
```

If you get `ModuleNotFoundError: No module named 'cv2'`, you ran the system
Python instead of this project's. Either activate the venv as above, or call it
directly: `my_env/bin/python pose_car.py`. In VS Code: Cmd+Shift+P →
**Python: Select Interpreter** → `P1-Pose-Race/my_env/bin/python`.

Keys, in the preview window:

- **`d`** — arm / disarm the motors. **Starts disarmed**, so you can see what the
  car *would* do before it does it. Worth having when the car is pointed at a
  door.
- **`q`** or **Esc** — quit (stops the motor and drops the BLE link on the way out).

With several cars in one room, a bare `connect()` grabs whichever motor answers
the scan first — possibly someone else's. Use the Connection Card that came with
the motor to name yours:

```bash
python pose_car.py --card-color orange --card-serial 1129
```

Other flags: `--camera N` to pick a different camera, `--min-confidence` to make
the classifier more or less cautious.

On macOS the first run triggers Camera and Bluetooth permission prompts for your
terminal — allow both.

### If the car drives the wrong way

How the motors sit in the chassis decides whether a positive speed drives the
car forwards or backwards, and there is no way to detect that from software.
`MOTOR_DIRECTION` in [`pose_car.py`](pose_car.py) flips it:

```python
MOTOR_DIRECTION = -1   # this build; use +1 if arms-up already drives forwards
```

It is applied at the single point where the command goes out over Bluetooth, so
everything else in the code stays written in the driver's terms — forward is
positive — and both control modes are corrected at once.

Flipping the sign of both wheels also mirrors turning: a left spin becomes a
right spin. That is correct when the whole chassis is mounted backwards. If
driving is now right but turning is mirrored, the two motors are swapped
left-for-right instead, which is a separate fix.

## Training your own pose classes

Instead of the built-in geometry, you can teach the car poses by example — you
record yourself holding each one, train a classifier on those recordings, and
drive from it.

### 1. Record examples

```bash
python collect_poses.py
```

A number key per class. Press one, get into the pose during the 3-second
countdown, hold it while the counter climbs, press any key to stop. Move around
between takes — step closer and further, stand left and right of frame, turn
slightly — so the model learns the *pose* and not the spot you were standing in.

**Aim for 100–200 samples per class**, which is about 5 seconds of holding each,
a few times over. Samples append to `pose_data.csv`, so you can record in
several sittings, and the script prints which classes are still thin.

Custom classes work too:

```bash
python collect_poses.py --classes forward,back,left,right,stop,boost
```

Anything new needs an entry in `CLASS_SPEEDS` in `pose_features.py`, or the car
won't know what speeds it means.

### 2. Train

```bash
python train_poses.py
```

Prints how many samples it has per class, holds out 25% to test on, and reports
precision/recall and a confusion matrix, then a 5-fold cross-validated accuracy.
Writes `pose_model.joblib`.

Read the confusion matrix rather than just the accuracy — it tells you *which*
poses get mixed up, which is what you'd go record more of. Two poses that keep
swapping usually just look too similar from the front, and are better fixed by
picking a more distinct pose than by collecting more data.

### 3. Drive from it

```bash
python pose_car.py --model pose_model.joblib
```

The predicted class and the model's confidence show in the corner of the
preview. Below `--min-confidence` (default 0.6) the car stops instead of acting
on a guess, which is what happens when you're between poses or doing something
it never saw.

Commit `pose_data.csv` and `pose_model.joblib` along with the code — they're the
evidence of what you trained on.

## How it works

One loop, roughly 30 times a second:

1. **Grab a frame** from the webcam with OpenCV and mirror it.
2. **Find the body.** MediaPipe's `PoseLandmarker` returns 33 landmarks, each
   with an `x`, `y` (as a fraction of the frame) and a `visibility` score.
3. **Decide what the car should do** — one of the two modes below.
4. **Smooth it** with an exponential moving average, so a single bad landmark
   doesn't jolt the car.
5. **Send it:** `movement_move_tank(left, right)` over BLE.

**Geometry mode** measures how far each wrist sits above its shoulder and
divides by the distance between the shoulders:

```
offset = (shoulder_y - wrist_y) / shoulder_width
```

Dividing by shoulder width is what makes this work at any distance from the
camera: standing closer scales every pixel measurement up, but their *ratio*
holds. A dead zone near zero keeps a level arm from creeping, and the result is
clamped to ±`MAX_SPEED`.

**Classifier mode** builds a feature vector from 9 upper-body landmarks — nose,
shoulders, elbows, wrists, hips — re-expressed relative to the midpoint of your
shoulders and divided by your shoulder width. That's the same normalization
trick, for the same reason, applied to every point instead of just the wrists:
what survives is the *shape* of your pose, with where you stood and how close
you were divided out. Those 18 numbers go to the classifier, which returns a
class and a probability.

Either way, each step also has a way to say "stop": a landmark below 0.5
visibility, a driver standing side-on to the camera (their shoulders overlap,
which would otherwise amplify noise into full throttle), no pose at all for
0.4s, or — in classifier mode — a prediction the model isn't confident about.

### One definition of a feature vector

The feature vector is defined in exactly one place on purpose. If recording,
training and driving disagreed about what the 18 numbers mean, the model would
still load and still predict — it would just predict nonsense, with nothing to
tell you why. So all three import the same definition, and both the CSV and the
saved model carry the column names so a mismatch is caught out loud.

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
inference, decide, send, repeat. I deliberately used MediaPipe's
`RunningMode.VIDEO`, whose `detect_for_video()` call blocks until the result is
ready, rather than `RunningMode.LIVE_STREAM`, which hands results back through a
callback on another thread. Blocking here is the right call: results arrive in
frame order, and a pose can never overtake a stale motor command. Inference on
the lite model is only a few milliseconds, so it isn't the bottleneck.

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

**It's two models, and I only trained one of them.** That split is the whole
design, so it's worth being precise about which is which.

**Stage 1 — finding the body. Pre-trained, not by me.** `pose_landmarker_lite.task`
is a model Google ships with MediaPipe (the BlazePose GHUM family), trained on
large human-pose datasets and used here exactly as downloaded. Turning raw pixels
into 33 body landmarks is the genuinely hard, data-hungry part of this problem,
and it is already solved.

**Stage 2 — deciding what a pose means. Trained by me, on my own data.** This is
`collect_poses.py` → `train_poses.py`: I record myself holding each pose, each
frame becomes one labelled example, and a scikit-learn pipeline
(`StandardScaler` → `LogisticRegression`) learns to tell them apart. Trained on
my recordings, evaluated on a held-out 25% and by 5-fold cross-validation, then
refit on everything for the model that ships.

**The reason this works with a few hundred examples instead of a few hundred
thousand is the feature vector.** Stage 1 has already reduced an image to 18
meaningful numbers, and I normalize those against your own shoulders — every
point re-expressed relative to your shoulder midpoint and divided by your
shoulder width. So "where in the frame you stood" and "how close to the camera
you were" are divided out *before* the model ever sees the data, instead of being
something it would have to learn to ignore by seeing examples from every position
and distance. That is why five seconds of holding a pose is enough. Using a
pre-trained model as a feature extractor and learning only a small classifier on
top of it is transfer learning, and it's what makes this trainable in a
classroom rather than a data centre.

**I also kept the untrained version**, which is the default. The geometric
mapping needs no data at all, gives smooth continuous speeds rather than a
handful of discrete commands, and can't be wrong in ways I can't inspect. The
classifier is more flexible — it can learn a pose that would be painful to write
a formula for — but it only outputs the classes you recorded, and it fails
silently where the geometry fails visibly. Having both means I can compare them
on the same hardware, and demo the one that's behaving.

#### Results

*Fill this in from your own `train_poses.py` output — how many samples per class,
the cross-validated accuracy, and which classes the confusion matrix shows
getting mixed up.*

**Limitations:**

*From the pre-trained stage:*

- It knows human bodies in general and nothing about *me* in particular.
- `lite` is the least accurate of the three model sizes, and wrists are among the
  noisiest landmarks it produces. I traded accuracy for frame rate.
- It degrades in low light, with baggy sleeves, and when arms cross the torso.

*From the stage I trained:*

- **It only knows what I showed it.** Every sample is me, in one room, in one
  outfit, in front of one camera. A different person or a very different setting
  is out-of-distribution, and the model has no way to say "I've never seen this"
  beyond a low probability.
- **Confidence gating helps but doesn't solve that.** On a pose it was never
  trained on, the 0.6 threshold catches roughly 4 out of 5 frames — the rest it
  acts on. Raising the threshold trades responsiveness for caution.
- **Discrete, not continuous.** It returns one of five commands. You cannot ease
  into a turn the way you can in geometry mode.
- **Near-perfect test accuracy is not as good as it sounds.** Poses chosen to be
  easy to tell apart are easy to tell apart, and consecutive frames of one
  recording are nearly identical, so a random split puts near-duplicates on both
  sides of it. The honest test is driving the car, not the number.

*From the system as a whole:*

- **Front-facing only.** I use 2D image coordinates, so an arm pointed toward or
  away from the camera looks identical to one hanging straight down. MediaPipe
  does estimate a `z` per landmark, but it is far less reliable than `x`/`y`, so
  I ignore it. A driver standing side-on is rejected outright rather than
  guessed at.
- **One driver.** `num_poses=1`, with no identity tracking. If someone walks
  behind me the detector can latch onto them instead, and the car will follow
  whoever it picked.
- **Smoothing costs latency.** The moving average adds roughly 3–4 frames
  (~100 ms) of lag, on top of the BLE send interval.
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
