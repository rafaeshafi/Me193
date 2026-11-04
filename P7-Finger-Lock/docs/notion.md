# Finger Lock: a safe that opens for my hand

*Notion page draft. Each section pastes as a block. Figures are in `docs/figures/`, the movie in `docs/movie/`. Lines
marked (your words) are for you to write.*

## Intro

My midterm safe opens for a three-digit code I show with my fingers. I push a button, the LEDs come on, I hold up 2,
then 5, then 3 fingers under a webcam on a stand, and a servo swings the latch open. The number of fingers in each
picture comes from FingerNet, a convolutional neural network I built from scratch in PyTorch and trained on about
a thousand pictures of my own hand, taken with the same USB webcam. The trained weights then moved onto the Arduino
UNO Q, where they run in plain numpy next to the same lock policy, the same button, LEDs and servo. The whole
pipeline, from pixels to "open", is mine: no vision library does the counting.

| deliverable | where |
|---|---|
| demo video | below |
| GitHub | https://github.com/rafaeshafi/Me193/tree/main/P7-Finger-Lock |
| stop-action movie (with my voice-over) | section 3 |
| the four midterm sections | sections 1 to 4 |
| something not done in class | saliency overlay, intruder log, numpy port |
| reflection questions | at the end |

## The build

- USB webcam on a stand, a fixed height above the desk, pointing straight down; a square drawn on the desk is the
  only part of the picture the net sees (the "region of interest", stored in `config.json` so the Mac and the board
  crop the same square).
- Arduino UNO Q in the safe: push button on D2, red LED on D3, green LED on D4, a hobby servo on D5 as the latch,
  the built-in 13x8 LED matrix showing the digits as they are read.
- Software: PyTorch on the Mac for training; the board runs the exported weights in numpy because the App Lab
  container has no PyTorch.

*(photo of the stand and the square)*  *(photo of the box with the servo latch)*

## Demo video

*(video to come: button, LEDs on, 2-5-3, servo opens; then a wrong code, red; then three wrong codes, skull)*

## How it works

Every tenth of a second: the webcam frame is cropped to the square, turned to grey and shrunk to 64x64; FingerNet
turns those 4,096 numbers into six probabilities (0 to 5 fingers); the lock policy turns the stream of probabilities
into digits and digits into a decision; the MCU shows the state on the matrix and the LEDs and drives the servo.

```
webcam → crop the square → 64x64 grey → FingerNet (conv 16 → conv 32 → conv 64 → linear 64 → 6) → softmax
       → lock policy (hold 8 frames, release, 3 digits, compare) → matrix / LEDs / servo
```

## 1. The lock policy: how it decides

The policy never looks at the picture; it only sees the six probabilities, ten times a second, and it is the same
code on the Mac, in the tests and on the board.

- **Idle.** LEDs off, a dim dot walks the bottom of the matrix. The button starts an entry (20 s to finish it).
- **Reading a digit.** A digit counts only when the same class 1 to 5 is the most likely answer with at least 85 %
  confidence for 8 frames in a row (0.8 s). A flicker, a hand moving in, a half-raised finger: no digit. The matrix
  shows the candidate glowing and a bar filling as the frames accumulate.
- **Release.** After a digit is accepted the hand has to go away (3 frames of "no hand" or of a different number)
  before the next digit can start, so holding up two fingers for five seconds is one "2", not three.
- **Decide.** Three digits are compared with the code in order. Match: servo open for 5 s, green LED, a tick on
  the matrix, then back to idle. No match: red LED, a cross, 2 s, idle, and the failed-attempt counter goes up.
- **Lockout.** Three wrong codes in a row and the safe ignores the button for 30 s, the red LED blinks and the matrix
  shows a skull; the picture of the hand that failed is saved for me. One right code clears the counter.
- **Zero is never a digit.** Class 0 (empty square, fist) is the separator between digits. The code itself must be
  three different digits from 1 to 5; the config file is refused otherwise, so there are 60 possible codes.

## 2. Limitations

1. **It knows one hand, one camera, one height.** Trained on my hand on my stand; someone else's hand, a different
   webcam or a stand ten centimetres higher changes the pictures and the accuracy drops until I retrain.
2. **Lighting.** The net works on grey pixels; a shadow across the square or a lamp from the other side moves every
   number it sees. Augmentation (brightness, contrast) helps, it does not cure.
3. **Three versus four.** Fingers that touch read as one wide finger; 3 and 4 are the pair the net confuses most
   (the confusion matrix shows it), so the code should avoid relying on a casual 4.
4. **No liveness.** A printed photo of my hand showing 2, 5, 3 opens it. The camera cannot tell skin from paper.
5. **The square.** A hand outside the drawn square is class 0; the lock does not search for the hand.
6. **Speed and timing.** The board reads about ten frames a second and needs eight in a row; a fast flick between
   numbers can miss a digit or, if a finger is half down, read the wrong one.
7. **Small code space and a weak latch.** 60 codes; the lockout makes guessing slow, not impossible, and a hobby
   servo latch stops an honest person, not a screwdriver.

## 3. How the neural net works

*(the stop-action movie with my voice-over goes here: `docs/movie/movie_silent.mp4` plus the recording)*

In short: a 64x64 grey picture goes through three rounds of convolution (a 3x3 window of learned weights sliding
over the picture, 16 then 32 then 64 of them), ReLU (negatives to zero) and 2x2 max-pooling (keep the biggest,
halve the size). The 64 little 8x8 maps are flattened into 4,096 numbers, a linear layer makes 64 hidden values
and another makes six scores; softmax turns the scores into probabilities. Training compares the probabilities with
the label I typed (cross-entropy), back-propagation gives the slope of that loss for each of the 285,894 weights,
and gradient descent (Adam) steps every weight a little downhill, 40 passes over the pictures, each picture
rotated, shifted, scaled and re-lit differently each time.

## 4. UNO Q versus my computer

| | MacBook Pro (M1 Pro) | UNO Q |
|---|---|---|
| inference per frame, PyTorch | 0.53 ms | no PyTorch there |
| inference per frame, numpy | 0.42 ms | *(fill from `scripts/board_stats.sh`)* |
| frames per second | webcam-bound (30) | 10 asked for; *(measured)* |
| training | ~1 min for 40 epochs | not done on the board |
| moving the model | `export.py` → JSON; numpy engine agrees with PyTorch to 1e-4 | `deploy.sh` copies weights + config + the shared `lock/` package |
| seeing what it does | OpenCV window | web page on port 8080 over the USB cable |
| turnaround | seconds | about 2 minutes per deploy |

*(your words: what was different about working on the board: the camera API, no PyTorch, the Bridge, debugging over the
page, how the accuracy compared on the box.)*

## Fun things not done in class

- **Saliency.** Pressing `s` in the Mac app paints the gradient of the winning score over the live picture: the
  pixels the net is counting light up, so you can see it "looking" at the finger tips and the gaps.
- **Intruder log.** Three wrong codes: a skull on the matrix, 30 s lockout, and the failing hand's picture is saved.
- **The numpy port.** The board has no PyTorch, so I wrote the forward pass (im2col convolution, pooling, linear)
  myself and proved it equals PyTorch with a test.

## Results

*(replace with the real numbers from `docs/results.md` once trained on real pictures; the figures are
`docs/figures/train_curves.png`, `confusion.png`, `samples.png`, `augment.png`)*

| run | validation accuracy (held-out sessions) |
|---|---|
| with augmentation | *(x)* % |
| without augmentation | *(y)* % |

## Reflection

- What I am most proud of: *(your words)*
- What was hardest: *(your words)*
- What I would do differently: *(your words)*
- What I learned: *(your words)*
