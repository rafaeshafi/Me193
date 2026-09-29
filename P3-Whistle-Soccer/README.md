# P3: Whistle Soccer

**Steer a LEGO Education car by whistling, then play World Cup over MQTT.**

ME 193 – AI & Robotics, Project 3.

The laptop microphone streams audio through PyAudio. Each ~46 ms block is
turned into a spectrum. If it holds a clean whistle, the pitch of that whistle
steers a LEGO Double Motor over Bluetooth. Your comfortable middle note drives
straight. Higher notes turn left and lower notes turn right, harder the further
you go. Silence stops the car.

On game day the car waits for `start` on the MQTT topic `ME193`. As the
**ball**, it tries to reach the goal. If the goalie gets close to the Color
Sensor on its nose, the car stops, publishes that it was caught, and plays a
death song. If it reaches the goal, you hold the top **goal whistle**. The car
then publishes the goal and plays a victory song.

The **goalie** is one robot run by **two people on two laptops**. Each person
streams audio from their own microphone, and the two laptops coordinate over
MQTT:

- **Robot laptop:** its whistle slides the car back and forth along the goal
  line. Above the middle note it drives forward, below it drives backward, and
  it never turns. It connects to the Double Motor over Bluetooth.
- **Glove laptop:** connects over Bluetooth to the **glove**, a Single Motor
  standing upright with a large LEGO piece on it. Its whistle aims the glove,
  using the same pitch system.
- **MQTT between them:** the robot laptop sends the game state, and the glove
  may only move while the robot says DRIVING. The glove laptop reports the
  glove's angle back, and the robot laptop shows it.

When the ball reports that it was caught or scored, the goalie plays the
opposite song on both laptops.

## Contents

| File | What it is |
|---|---|
| [`whistle_car.py`](whistle_car.py) | Ball car, and the goalie's **robot laptop**: game states, wheels, sensor, MQTT |
| [`glove.py`](glove.py) | Goalie's **glove laptop**: whistle to glove angle, glove motor, MQTT |
| [`live_audio.py`](live_audio.py) | Shared by both laptops: PyAudio microphone stream, live display, calibration |
| [`whistle_policy.py`](whistle_policy.py) | Whistle detection and the steering policy. Pure numpy, no hardware |
| [`mqtt_chat.py`](mqtt_chat.py) | MQTT chat window: subscribe to topics, watch and send messages |
| [`songs.py`](songs.py) | Plays the songs; `python songs.py death` previews one |
| [`config.py`](config.py) | **Everything you might change:** MQTT topic and messages, songs, card, speeds, bands |
| [`requirements.txt`](requirements.txt) | Pinned dependencies |

## Setup

```sh
brew install portaudio        # PyAudio's C library (macOS)
cd P3-Whistle-Soccer
python3.12 -m venv my_env && my_env/bin/pip install -r requirements.txt
```

Every device pairs from the kit's Connection Card, set in `config.py`
(`CARD_COLOR = "green"`, `CARD_SERIAL = "0997"`).

- **Ball:** Double Motor, plus the Color Sensor facing forward, open, at the
  front of the car.
- **Goalie:** Double Motor, plus the glove, a Single Motor mounted vertically
  with a large LEGO piece on its hub. The robot laptop connects to the Double
  Motor and the glove laptop connects to the Single Motor. Point the glove
  straight ahead before launching `glove.py`, because wherever it points then
  becomes 0°.

Both goalie laptops need this folder and its setup. On a Windows glove laptop,
skip `brew`: install Python 3.12 from python.org, then run
`py -3.12 -m venv my_env` and
`my_env\Scripts\python -m pip install pyaudio numpy matplotlib paho-mqtt legoeducation`,
and run the programs with `my_env\Scripts\python`. Each person calibrates on
their own laptop, because `calibration.json` belongs to one whistle and one
microphone, and it is gitignored.

## Running it

```sh
my_env/bin/python whistle_car.py --calibrate        # once per laptop: learn your whistle
my_env/bin/python whistle_car.py --role ball        # ball car
my_env/bin/python whistle_car.py --no-motor --no-sensor   # practise with no hardware
```

**Goalie, two laptops:**

```sh
# Laptop 1, robot driver: Double Motor (wheels) over Bluetooth
my_env/bin/python whistle_car.py --role goalie

# Laptop 2, glove: Single Motor (glove) over Bluetooth
my_env/bin/python glove.py --calibrate     # once
my_env/bin/python glove.py
```

The glove laptop's headline shows **NO ROBOT** until it hears the robot laptop
over MQTT, and the glove stays frozen until the robot says DRIVING. To practise:

- **Glove on its own:** `glove.py --solo` moves the glove without waiting for
  the robot laptop.
- **Without hardware:** `glove.py --no-motor` and
  `whistle_car.py --role goalie --no-motor`.

Stay quiet for the first 3 seconds after launch while it measures the room
(`NOISE_SECONDS` in `config.py`). Keys
in the plot window:

| Key | Does |
|---|---|
| `s` | start locally (no MQTT needed, for practice) |
| `r` | reset to waiting |
| `c` / `g` | pretend caught / goal, to test the messages and songs |
| `q` | quit |

On the glove laptop, `c` re-centres the glove and `q` quits. Ctrl+C in the
terminal also quits cleanly on both.

### MQTT chat window

```sh
my_env/bin/python mqtt_chat.py
```

Click **Connect**. The window starts subscribed to the game topic and the
goalie team topic. To add a topic, type it and press **Subscribe**; wildcards
like `ME193/#` work. Every message on your topics appears live with its time
and topic, and your own messages are shown in green.

To send, pick a topic in **To**, type, and press Enter. The **send 'start' /
'caught' / 'goal'** buttons publish the game messages to `ME193`, which
is handy for testing the robots without the instructor. Messages go out
exactly as typed, so tick **add my name** only for human chat: `Rafae: start`
is not a start command. **Hide robot heartbeats** (on by default) hides the
once-a-second `state` / `glove` messages between the goalie laptops.

### Changing the messages or the songs

Open [`config.py`](config.py):

- `TOPIC`, `MSG_START`, `MSG_CAUGHT`, `MSG_GOAL` are plain strings. Set them to
  whatever you and your opponent agree on.
- `DEATH_SONG` and `VICTORY_SONG` are either a list of `(note, beats)` pairs,
  like `[("C5", 1), ("R", 0.5), ("G4", 2)]`, or the path to any `.wav` file,
  like `"sounds/sad_trombone.wav"`. Preview a song with `python songs.py death`.
- `TEAM_TOPIC`, `GLOVE_CMD` and `STATE_CMD` are the goalie laptops' private
  channel and words. The `GLOVE_*` values set how far and fast the glove
  swings, and `GLOVE_DIRECTION = -1` flips it if it swings the wrong way.

## Goalie: two laptops, one robot, MQTT between them

```
 person 1 whistles                                 person 2 whistles
       |                                                  |
 [robot laptop]  -------- "state DRIVING" ------------>  [glove laptop]
  mic -> pitch -> drive         ME193/Cucurella            mic -> pitch -> glove angle
       |         <------- "glove -45" -----------------       |  (plays the songs too)
       | Bluetooth                                           | Bluetooth
       v                                                     v
 Double Motor (wheels)  ====== same goalie robot ======  Single Motor (glove)
       ^
       |  "start" / "ball:caught" / "ball:goal" on ME193
 instructor + ball
```

| Topic | Message | From → to | Meaning |
|---|---|---|---|
| `ME193` | `start` | instructor → everyone | begin |
| `ME193` | `ball:caught` / `ball:goal` | ball → goalie | goalie won / lost |
| `ME193` | anything in `HEAR_GOAL` / `HEAR_CAUGHT` | other team's ball → our goalie | their wording also counts (e.g. `goal`, `failed`), any case |
| `ME193/Cucurella` | `state <STATE>` | robot laptop → glove laptop | game state, on every change plus once a second as a heartbeat. The glove only moves while it is `DRIVING` |
| `ME193/Cucurella` | `glove <deg>` | glove laptop → robot laptop | where the glove is (−90 … +90), on every change plus once a second |

Design choices:

- **Each laptop owns its own motor.** A LEGO device takes one Bluetooth
  connection at a time, so each motor has exactly one owner: the robot laptop
  has the wheels and the glove laptop has the glove. The glove responds to its
  whistle with no network delay.
- **Angles, not speeds.** The glove runs in position mode
  (`motor_run_to_relative_position`) with the motor set to hold. A glove
  command means "be at −45°", so a late or repeated message can't make it
  spin away, and silence leaves it where it is.
- **The robot laptop is the referee.** The glove laptop doesn't listen to the
  instructor directly. It moves the glove only while the robot laptop reports
  DRIVING. So both halves of the goalie start and stop together, and if the
  robot laptop goes quiet for 3 s, the glove freezes.
- **Traffic stays low but recovers from lost messages.** Each side sends on
  every change, plus a once-a-second heartbeat, in case a message is lost on
  the public broker. Glove angles are rounded to 5° steps.

## The live display

The window has three panels, redrawn continuously:

1. **Microphone signal:** the raw waveform of the current block.
2. **Spectrum:** the block's spectrum in dB. Shaded regions show the decision
   bands: blue = RIGHT, green = STRAIGHT, orange = LEFT, purple = GOAL. The red
   dashed line is the room's noise floor plus the 25 dB gate, and a green dot
   marks an accepted whistle.
3. **Decision:** a big headline (STRAIGHT / LEFT / RIGHT / STOP / WON / LOST),
   plus the pitch, why the block was accepted or rejected, the value of every
   noise gate, steering, both motor speeds, and the sensor reading. On the
   goalie robot laptop, the last line shows the glove angle reported by the
   glove laptop, and how long ago that report arrived.

The glove laptop shows the same three panels, with the bands relabelled SWING
RIGHT / CENTRE / SWING LEFT. Its headline is the glove angle, HOLD when
silent, or the robot's state when the glove is frozen.

## Questions

### Describe the policy: how does it make decisions?

Every audio block goes through three steps.

1. **Hear:** Hann-window the block and take its FFT. Find the strongest peak
   between 500 and 3500 Hz, refined to about 2 Hz with parabolic
   interpolation. If the block fails any noise gate (below), it counts as "no
   whistle".
2. **Smooth:** take the median of the last 5 accepted pitches, so one bad
   estimate cannot jerk the wheel.
3. **Decide:** compare the smoothed pitch to your calibrated notes
   (`calibration.json`):

   | Pitch | Decision | Motors (base = 50 %) |
   |---|---|---|
   | within ±100 Hz of your middle note | **STRAIGHT** | both at base |
   | above that, up to your high note | **LEFT**, in proportion | left wheel slows, reaching 0 at your high note |
   | below that, down to your low note | **RIGHT**, in proportion | right wheel slows, reaching 0 at your low note |
   | above the goal threshold, held 0.75 s | **GOAL** (ball only) | stop, publish `ball:goal`, victory song |
   | no whistle for 0.3 s | **STOP** | both 0 |

   Steering is continuous. A whistle a little above the middle is a gentle
   left and a whistle at your top steering note is a pivot, so it behaves like
   a steering wheel rather than three buttons. Leaving STRAIGHT needs 25 % more
   than the dead-band (hysteresis), so a note right on the edge doesn't
   twitch. While the goal whistle is building up, the car holds still, so
   whistling up to the goal note doesn't make it lurch left.

Around the policy sits a game state machine. **WAITING** keeps the motors at 0
until `start` arrives. In **DRIVING**, whistles steer. **LOST** and **WON** stop
the car and play a song. As the ball, the car switches to LOST when the Color
Sensor's reflection rises 20 above its value at `start` for 0.1 s. That means
something is right in front of it, which is the goalie. It then publishes
`ball:caught`. The goalie role switches on the ball's messages instead.

**The goalie robot** only moves along the goal line. The same steering value
becomes one speed for both wheels: steer × `GOALIE_SPEED` (60 %). Above the
middle note it drives forward, below it drives backward, faster the further
from the middle. The middle note or silence stops it, and the goal band is
ignored. Because both wheels always get the same speed, it cannot turn off the
line.

**The glove** uses the same hearing and smoothing steps, and the same bands,
on the second laptop. The steering value −1 … +1 becomes an angle: steer ×
90°, rounded to 5°. Your middle note centres the glove, a higher note swings it
left and a lower note swings it right, further the further you go from the
middle. So the glove person can follow the ball smoothly instead of flicking
between fixed positions.

### What does your code do if no whistle is detected?

The car **stops**. A gap shorter than 0.3 s keeps the last command, so taking a
breath between whistles doesn't make it stutter. After 0.3 s of silence, the
policy clears its history and outputs 0/0. The stop goes to the motor at once,
skipping the rate limit that applies to other motor commands. Silence is the
safe state: if the mic dies, the laptop freezes, or you stop whistling, the car
stops. Before `start` and after the game ends, the motors are held at 0 whatever
the microphone hears.

The glove does the opposite: with no whistle it **holds its current angle**.
The motor gets no new command, and its hold mode keeps it where it is. For a
goalkeeper, staying in position is the safe default, while snapping back to
the centre would leave a gap. If the glove laptop drops out entirely, the
robot laptop still drives normally and shows how long ago the glove laptop was
last heard. If the robot laptop's heartbeat stops, the glove laptop shows
**NO ROBOT** and freezes the glove.

### How did you try to mask out unwanted noise?

A block only counts as a whistle if it passes **every** gate. The display shows
which gate rejected each block and by how much.

1. **Band limit (500–3500 Hz).** Whistles live here. Motor whine, desk thumps,
   HVAC and the fundamental of most voices sit below it.
2. **Loudness gate.** Only loud sounds are read: a block quieter than RMS 0.03
   (`MIN_LOUDNESS`) is treated as silence, so distant sounds are ignored.
3. **Adaptive noise floor.** At launch, the program records 3 seconds of the
   room and keeps its level at every frequency. The peak must be 25 dB above
   the room's own level *at that frequency*. A fan or buzzing light raises the
   bar only where it is noisy.
4. **Tonality.** A whistle is one narrow spectral line. The peak must be 20 dB
   above the median of the band. Claps, crowd noise and hiss spread their
   energy and fail this test.
5. **Out-of-band check.** Speech and singing are stacks of harmonics, and one
   of them can land in the band and look like a whistle. Their strongest line
   is the fundamental below 500 Hz, though. If anything outside the band is
   louder than the peak, the block is rejected as voice or hum.
6. **Time filtering.** The 5-block median ignores isolated false detections.
   The goal needs a 0.75 s hold, so a squeak cannot end the game.
7. **Self-mute.** While our own song plays, the detector ignores the mic so the
   car doesn't hear its own speaker.

Offline tests with synthetic audio: pure tones from 900 to 3000 Hz were
measured within 1 Hz. Room noise, loud white noise, a clap, 120 Hz hum and a
voice-like 180 Hz harmonic stack were all rejected.
