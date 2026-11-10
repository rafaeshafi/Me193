# Finger Lock: a safe that opens for my hand

*Paste this into Notion. Every `##` line is meant to be a toggle heading: paste the page, then click each heading's
block menu and pick Turn into → Toggle heading 1. Lines starting `(photo:` are spots for pictures of the setup; lines
starting `(upload:` are figures from the repo, all in `docs/figures/`. Lines marked (my words) are yours to write.*

This is a safe that opens when I hold up the right numbers with my fingers. You press the button, show three numbers one at a time, and if they match the code the bolt turns. The counting is done by a small neural net I trained on pictures of my own hand. All of it runs on an Arduino UNO Q, with a webcam pointed down at a square on my desk.

| | |
|---|---|
| code | three different digits from 1 to 5, each held for 3 seconds |
| brain | FingerNet, built from scratch in PyTorch, running in numpy on the board |
| body | UNO Q, USB webcam, stepper bolt, push button, two LEDs, 3.2" TFT screen |
| GitHub | https://github.com/rafaeshafi/Me193/tree/main/P7-Finger-Lock |

(photo: the whole safe from the front, screen and button visible)

## Demo

(video: button, show 1, 4, 3, bolt opens for 10 s and closes; then a wrong code, red LED and a cross; then three wrong codes, skull and lockout)

## How it works in one picture

(upload: pipeline.png)

About ten times a second the board does this:

- grabs a frame from the webcam, crops it to the square on the desk and shrinks it to 128 x 128 greyscale
- runs it through the net, which gives back six numbers: how sure it is that it sees 0, 1, 2, 3, 4 or 5 fingers
- waits until the same answer has won for 3 seconds straight. That counts as one digit
- after three digits, checks them against the code
- tells the MCU what to put on the screen, the matrix and the LEDs, and whether to move the bolt

## The build

### Parts

- Arduino UNO Q. The Linux side runs the net and the web page, the MCU side runs the hardware
- a USB webcam on a stand, looking straight down at a square I drew on the desk
- a 4-wire bipolar stepper (200 steps a turn) on an HW-627 driver board, which is a DRV8833 chip. This is the bolt
- a push button with COM / NO / NC terminals
- a red LED and a green LED
- a 3.2 inch SPI TFT screen, 240 x 320, ILI9341
- a powered USB-C hub, so the board can run the webcam without my laptop

(photo: the stand and the square on the desk)

(photo: inside the box, the stepper and the bolt)

### Wiring

| part | UNO Q pins | note |
|---|---|---|
| button | COM to GND, NO to D2 | internal pull-up, 30 ms debounce. NC stays unused |
| red LED | D3 through 220 Ω | denied, lockout blinks |
| green LED | D4 through 220 Ω | unlocked; both on while listening |
| stepper driver | IN1 D6, IN2 D7, IN3 D8, IN4 D9, EEP D10 | coil A on OUT1/OUT2, coil B on OUT3/OUT4; EEP high wakes the chip |
| motor supply | driver VCC, 3 to 10 V, ground shared | not the board's 5 V pin |
| TFT screen | SCK D13, MOSI D11, CS D5, DC A0, RST A1 | 3.3 V logic, backlight on 3.3 V |
| webcam | the board's USB, through the powered hub | |

(photo: the driver board with the four motor wires and the five signal wires)

### Finding the stepper's coils

A 4-wire stepper is really two coils, and the wire colours tell you nothing about which wires belong together. A multimeter does. Two wires on the same coil read a few ohms, and two wires on different coils read open. If you pair them wrong the motor just sits there and buzzes. If it turns the wrong way, swap the two wires of one coil.

## The bolt

(upload: latch_timing.png)

When the code is right, the stepper turns 50 full steps, which is a quarter turn, at one step every 20 ms. So it takes about a second to open. It stays open for 10 seconds and then turns the 50 steps back. After that I cut power to the coils so the motor is not sitting there getting hot.

My first version stepped every 5 ms and it did not work. The motor hummed, skipped steps and never made it to 90 degrees. It turns out a stepper cannot go from standing still straight to 200 steps a second. I slowed it down to 20 ms and it has not missed since.

For setting it up I added a motor panel to the web page. I can nudge the bolt by 1, 10 or 50 steps, hold a button to keep it turning, stop it, or let the coils go. There is also a "here = closed" button I press once the bolt is sitting where closed should be.

## The screen and the lights

The TFT shows what state the lock is in, colour coded, plus the digits it has read so far, like `1 4 _`. It never shows the actual code.

| state | screen | LEDs | matrix |
|---|---|---|---|
| idle | grey IDLE | off | a dot walking the bottom row |
| listening | yellow LISTENING | both on | digits and a hold bar filling |
| unlocked | green UNLOCKED | green | a tick |
| denied | red DENIED | red for 2 s | a cross |
| lockout | red LOCKOUT | red blinking 30 s | a skull |

(photo: the screen showing LISTENING with one digit in)

## The lock policy

The net gets single frames wrong sometimes, so the rules around it do a lot of the work.

- A digit only counts if the net says the same number, 1 to 5, with at least 85 % confidence for 30 frames in a row. At 10 fps that is 3 seconds of holding still.
- You have to take your hand out of the square before the next digit. Otherwise holding up a 4 for ten seconds would read as 4, 4, 4.
- Zero fingers is never a digit. A fist or an empty square is just the pause between digits.
- You get 20 seconds to enter all three, and the order matters.
- Get it wrong three times in a row and it locks you out for 30 seconds, ignores the button, and saves a picture of the hand that tried.

## The neural net, briefly

(upload: samples.png)

The net is called FingerNet and it is small. It takes the 128 x 128 grey picture of the square and puts it through three rounds of 3 x 3 convolution, ReLU and 2 x 2 max-pooling, with 16, then 32, then 64 filters. Then two linear layers take it from 16,384 numbers down to 64 and then to 6, and a softmax turns those 6 into probabilities. That is about a million weights in total.

I trained it with cross-entropy loss and Adam for 100 passes over the pictures. On every pass each picture gets rotated, shifted, scaled, flipped and re-lit a bit differently, so the net never sees exactly the same image twice.

It is the same gradient descent as the single-neuron homework. There are just a million weights to nudge instead of two.

## Training data and results

(upload: augment.png)

I took 896 pictures of my own hand with a capture tool I wrote, all in one sitting, between 130 and 161 for each number. Each picture is the whole square, so my hand could be anywhere inside it. For validation I held out the last quarter of every burst.

(upload: train_curves.png)

(upload: confusion.png)

| run | validation accuracy |
|---|---|
| with augmentation, the one on the board | 100 % |
| without augmentation | 100 % |

I do not fully trust that 100 %. The held-out pictures come from the same session as the training ones, so they look a lot alike. The real test would be a second session on another day, in different light.

## UNO Q versus my laptop

| | MacBook Pro M1 | UNO Q |
|---|---|---|
| net per frame | 0.5 ms, PyTorch or numpy | 59 ms in numpy, 7 ms more to prepare the frame |
| frames a second | whatever the camera gives | 10, still camera-bound |
| loading the weights | instant | 2.5 s for a 24 MB JSON |
| training | a minute | not on the board |
| seeing what it does | an OpenCV window | a web page over the USB cable or WiFi |
| a change | run it | about 2 to 3 minutes, the board compiles the sketch itself |

To get the model onto the board, `export.py` saves the PyTorch weights as JSON. On the board, a forward pass I wrote in plain numpy reads that file. I have a test that checks the numpy version and the PyTorch version give the same answer to within 1e-4.

(my words: what felt different about working on the board)

## Things that bit me

- The UNO Q has one USB-C port, so it cannot be plugged into my laptop and run the webcam at the same time. I deploy over the cable, then unplug it and move it to the powered hub.
- I unplugged the board once while it was still compiling the sketch. That left a half-built sketch behind, and the app refused to start until I cleared the cache.
- The board had no internet when I needed the screen libraries, so I copied them over USB into its library folder by hand.
- My capture tool let me drag the square bigger than the camera frame. The config check caught it and the app would not start, but nothing on the board said why. I had to go digging in the log.

## Not done in class

A few things I added that we did not cover:

- A saliency view on the laptop. Press a key and it lights up the pixels the net is actually looking at.
- The intruder log, and the skull on the matrix.
- Rewriting the whole net in numpy so the board does not need PyTorch.
- A web page served from the board with the live camera crop, six bars for the six probabilities, and the motor panel.

## Reflection

- Most proud of: (my words)
- Hardest part: (my words)
- Would do differently: (my words)
- Learned: (my words)
