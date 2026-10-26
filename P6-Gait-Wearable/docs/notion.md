# Gait Wearable: stand, walk, run or bike from one thigh sensor

*Notion page draft. Each section pastes as a block. Figures are in `docs/figures/`.*

## Intro

I built a wearable from an Arduino UNO Q and an MPU6050 inertial sensor that tells, live, whether the person wearing it is standing still, walking, running or biking (including lying on their back and air-pedalling). It straps to the thigh with two elastic bands and runs off a power bank in your pocket; the result shows on a web page served by the board itself, which any phone on the same hotspot can open. Two classifiers run side by side on every two-second window. Model 1 is allowed exactly one number, the peak acceleration along the thigh. Model 2 sees all six channels of the IMU, three accelerations and three rotation rates. Both are small decision trees, no neural nets. On two recorded sessions, each scored by a model that never saw it, Model 2 gets 98 % right and Model 1 gets 72 %, and the 26 % gap is almost entirely biking, which is the interesting part.

## How it works

- **Sensor to board.** The MPU6050 samples at 100 Hz into its own hardware buffer; the microcontroller empties it and sends 10 samples at a time to the Linux side of the UNO Q.
- **Windows and features.** Every half second the board takes the last two seconds and turns them into a handful of numbers: peak thigh-axis g, how much the thigh is tilted from vertical, the spread of acceleration along and across the thigh, rotation energy, the swing frequency (one cycle per stride or pedal revolution) and a rotation-per-impact ratio.
- **Two trees.** Model 1 reads only peak g. Model 2 reads the rest, behind one fixed question: if the gyro says the leg is not moving (under 40 deg/s, a still leg reads about 10), it is STAND, no matter what the posture is. The learned part only has to sort WALK, RUN and BIKE.
- **Vote and display.** The last three windows vote, which stops the label flickering on a stride boundary. The page shows both verdicts, whether they agree, cadence, peak g and the live acceleration trace.
- **Training loop.** The same page records labelled sessions: it counts you through STAND 20 s, WALK 30 s, RUN 30 s, BIKE 30 s. The files are pulled over USB, scikit-learn fits the trees on the Mac using the exact feature code the board runs, and the trees go back as small JSON files.

## Which model is better, and why

Model 2, by a wide margin, and the reason is physical rather than statistical. Peak thigh acceleration is really a measure of impact: standing has none (0.8 g, just gravity), walking has heel strikes around 1.3 g, running around 2.2 g. Biking has no impacts either, so its peak sits at 1.2 g, the same as walking. One number cannot tell "leg hitting the ground gently" from "leg going round with no ground at all". In the data, 76 % of walking windows were labelled BIKE by Model 1, and the only reason its biking looks decent (84 %) is that it learned a thin slice between standing and walking and called it BIKE.

Model 2 does not need impacts to see biking. Two things give it away: the thigh is horizontal (gravity along the thigh drops from 1 g to about 0.3 g), and the gyro traces a clean sine wave once per pedal stroke instead of the spiky bursts of a step. The deployed tree reads out loud: is the leg moving at all; is there an impact above 2.8 g (RUN); is the thigh horizontal (BIKE); otherwise WALK.

The honest cost is that Model 2 needs nine numbers and a tree you cannot set by hand, and that its first version was wrong in an instructive way. Trained on only one session it learned that "walking" means "leaning slightly forward", because in that one recording the lean separated standing from walking as cleanly as motion did. Trained on a second session, scored against the first, it dropped to 77 %. Fixing it meant removing orientation-dependent features and forcing the "is it moving" question to the top of the tree. That is also why peers can wear it with the sensor rotated a little and it still works.

## Figures

- `features.png`: left, the only thing Model 1 sees, with WALK and BIKE overlapping; right, the same windows against Model 2's first learned split, where BIKE separates.
- `tree_model1.png`: Model 1, four leaves on one feature.
- `tree_model2.png`: the learned part of Model 2 that runs after the motion gate.
- `confusion.png`: both models scored leave-one-session-out. Rows are what I was doing, columns what the model said.

## Demo video shot list

1. Strap on, bank in pocket, phone showing the page with STAND.
2. Walk across the room: the big word turns WALK, cadence appears, trace shows the steps.
3. Jog back: RUN, peak g over 2.
4. Lie down and air-pedal: Model 2 says BIKE while the Model 1 line says WALK in amber, "differs".
5. Stop pedalling, still lying down: STAND (the motion gate), then stand up.
6. Hand it to a friend for the same loop.

## GitHub

https://github.com/rafaeshafi/Me193/tree/main/P6-Gait-Wearable

## Reflection questions

*(answers go here once the four questions are in)*
