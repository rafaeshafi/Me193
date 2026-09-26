# P3: Whistle Soccer

**Steer a LEGO Education car by whistling, then play World Cup over MQTT.**

ME 193 – AI & Robotics, Project 3.

The laptop microphone streams audio through PyAudio. Each ~46 ms block is
turned into a spectrum. If it holds a clean whistle, the pitch of that whistle
steers a LEGO Double Motor over Bluetooth. Your comfortable middle note drives
straight. Higher notes turn left and lower notes turn right, harder the further
you go. Silence stops the car.

On game day the car waits for `start` on the MQTT topic `ME193/Rogers`. As the
**ball**, it tries to reach the goal. If the goalie gets close to the Color
Sensor on its nose, the car stops, publishes that it was caught, and plays a
death song. If it reaches the goal, you hold the top **goal whistle**. The car
then publishes the goal and plays a victory song. The **goalie's** laptop
listens to the same topic and plays the opposite song.

## Contents

| File | What it is |
|---|---|
| [`whistle_car.py`](whistle_car.py) | The program: audio stream, live display, game states, motor, sensor, MQTT, calibration |
| [`whistle_policy.py`](whistle_policy.py) | Whistle detection and the steering policy. Pure numpy, no hardware |
| [`songs.py`](songs.py) | Plays the songs; `python songs.py death` previews one |
| [`config.py`](config.py) | **Everything you might change:** MQTT topic and messages, songs, card, speeds, bands |
| [`requirements.txt`](requirements.txt) | Pinned dependencies |

## Setup

```sh
brew install portaudio        # PyAudio's C library (macOS)
cd P3-Whistle-Soccer
python3.12 -m venv my_env && my_env/bin/pip install -r requirements.txt
```

The Double Motor and Color Sensor both pair from the kit's Connection Card,
set in `config.py` (`CARD_COLOR = "green"`, `CARD_SERIAL = "0997"`). The Color
Sensor must face forward, open, at the front of the car.

## Running it

```sh
my_env/bin/python whistle_car.py --calibrate        # once: learn your whistle
my_env/bin/python whistle_car.py --role ball        # game day (or --role goalie)
my_env/bin/python whistle_car.py --no-motor --no-sensor   # practise with no hardware
```

Stay quiet for the first second after launch while it measures the room. Keys
in the plot window:

| Key | Does |
|---|---|
| `s` | start locally (no MQTT needed, for practice) |
| `r` | reset to waiting |
| `c` / `g` | pretend caught / goal, to test the messages and songs |
| `q` | quit |

### Changing the messages or the songs

Open [`config.py`](config.py):

- `TOPIC`, `MSG_START`, `MSG_CAUGHT`, `MSG_GOAL` are plain strings. Set them to
  whatever you and your opponent agree on.
- `DEATH_SONG` and `VICTORY_SONG` are either a list of `(note, beats)` pairs,
  like `[("C5", 1), ("R", 0.5), ("G4", 2)]`, or the path to any `.wav` file,
  like `"sounds/sad_trombone.wav"`. Preview a song with `python songs.py death`.

## The live display

The window has three panels, redrawn continuously:

1. **Microphone signal:** the raw waveform of the current block.
2. **Spectrum:** the block's spectrum in dB. Shaded regions show the decision
   bands: blue = RIGHT, green = STRAIGHT, orange = LEFT, purple = GOAL. The red
   dashed line is the room's noise floor plus the 15 dB gate, and a green dot
   marks an accepted whistle.
3. **Decision:** a big headline (STRAIGHT / LEFT / RIGHT / STOP / WON / LOST),
   plus the pitch, why the block was accepted or rejected, the value of every
   noise gate, steering, both motor speeds, and the sensor reading.

## Questions

### Describe the policy: how does it make decisions?

Every audio block goes through three steps.

1. **Hear:** Hann-window the block and take its FFT. Find the strongest peak
   between 700 and 3500 Hz, refined to about 2 Hz with parabolic
   interpolation. If the block fails any noise gate (below), it counts as "no
   whistle".
2. **Smooth:** take the median of the last 5 accepted pitches, so one bad
   estimate cannot jerk the wheel.
3. **Decide:** compare the smoothed pitch to your calibrated notes
   (`calibration.json`):

   | Pitch | Decision | Motors (base = 50 %) |
   |---|---|---|
   | within ±80 Hz of your middle note | **STRAIGHT** | both at base |
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

### What does your code do if no whistle is detected?

The car **stops**. A gap shorter than 0.3 s keeps the last command, so taking a
breath between whistles doesn't make it stutter. After 0.3 s of silence, the
policy clears its history and outputs 0/0. The stop goes to the motor at once,
skipping the rate limit that applies to other motor commands. Silence is the
safe state: if the mic dies, the laptop freezes, or you stop whistling, the car
stops. Before `start` and after the game ends, the motors are held at 0 whatever
the microphone hears.

### How did you try to mask out unwanted noise?

A block only counts as a whistle if it passes **every** gate. The display shows
which gate rejected each block and by how much.

1. **Band limit (700–3500 Hz).** Whistles live here. Motor whine, desk thumps,
   HVAC and the fundamental of most voices sit below it.
2. **Loudness gate.** An RMS under 0.005 counts as silence.
3. **Adaptive noise floor.** At launch, the program records one second of the
   room and keeps its level at every frequency. The peak must be 15 dB above
   the room's own level *at that frequency*. A fan or buzzing light raises the
   bar only where it is noisy.
4. **Tonality.** A whistle is one narrow spectral line. The peak must be 20 dB
   above the median of the band. Claps, crowd noise and hiss spread their
   energy and fail this test.
5. **Out-of-band check.** Speech and singing are stacks of harmonics, and one
   of them can land in the band and look like a whistle. Their strongest line
   is the fundamental below 700 Hz, though. If anything outside the band is
   louder than the peak, the block is rejected as voice or hum.
6. **Time filtering.** The 5-block median ignores isolated false detections.
   The goal needs a 0.75 s hold, so a squeak cannot end the game.
7. **Self-mute.** While our own song plays, the detector ignores the mic so the
   car doesn't hear its own speaker.

Offline tests with synthetic audio: pure tones from 900 to 3000 Hz were
measured within 1 Hz. Room noise, loud white noise, a clap, 120 Hz hum and a
voice-like 180 Hz harmonic stack were all rejected.
