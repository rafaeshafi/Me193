# Step counter: a neural net on the thigh wearable

*Notion page draft. Each section pastes as a block. Figures are in `docs/figures/`, the animation in `docs/animation/`.*

## Intro

The P6 wearable now counts steps. The same thigh-strapped Arduino UNO Q and MPU6050 feed a small neural network, trained in PyTorch on my Mac and run on the board in plain Python. The Steps tab on the phone shows the count live with a ± band that comes from measured errors, not a guess. The net learned from nothing but takes where I walked, counted every footfall out loud, and typed the total when I stopped.

## How the net works

**What goes in.** Every sample is six numbers: three accelerations in g and three rotation rates in degrees per second. The board averages every four 100 Hz samples into one, so the net sees 25 Hz. Walking never gets near 12 Hz, so nothing is lost and the board does a quarter of the work. Each channel is then standardised, subtract its mean and divide by its spread, with the means and spreads measured on the training takes and shipped with the weights. All six go in because a step shows up differently in each: a swing in one gyro axis, a heel strike in the accelerations, a roll as the hip shifts.

**What it computes.** Three causal dilated 1-D convolutions, each with 12 filters of width 5, at dilations 1, 2 and 4, so the last layer sees 1.16 seconds of history and nothing from the future. A final one-tap layer with a Softplus squashes that into a non-negative step rate in steps per second. Summing the rate times 0.04 s gives the count. I chose a rate rather than a "step or not" classifier because the label I can honestly provide is a total per take, not the moment of each footfall, and a rate integrates to a total. On the board the same 1,849 parameters run as nested loops in plain Python 25 times a second.

**How it learned.** One number per take: the count I typed. The loss is the smooth-L1 distance between the integrated rate and that number. Back-propagation in PyTorch pushes each weight a little against its gradient, and Adam repeats that for 400 passes. Four augmentations stop the net from cheating: rotating the sensor frame by up to 15°, scaling the signal, concatenating two takes so the counts add, and always stretching time, which is what makes it count footfalls instead of measuring how long I moved. Stretches of the earlier guided sessions where I stood or air-pedalled are added as exact zeros. Then the weights are written to a JSON file and the board loads them.

## How good is it?

Every number below is leave-one-recording-out: the net is trained without a take, then asked to count that take.

| take | true | predicted |
|---|---|---|
| walk, first test | 30 | 37.8 |
| walk, second test | 32 | 25.9 |
| walk | 31 | 30.2 |
| walk, short | 7 | 10.7 |
| walk, next day | 40 | 39.4 |
| walk, next day | 27 | 19.0 |
| stand ×3 | 0 | 0.1, 0.2, 0.1 |

| metric | value | what it means for the wearer |
|---|---|---|
| mean absolute error | 3.0 steps (walks 4.5, stands 0.1) | a typical 30-step walk is off by about 4 |
| mean absolute % error | 22 % | relative miss, averaged |
| bias | −0.4 steps | it does not systematically over- or under-count |
| 95th percentile of the % error | 47 % | the live ± band: 19 out of 20 takes land inside it |
| seed-to-seed spread | 0.4 steps | retraining with a different random start barely moves a prediction |

Standing reads zero every time; walking carries the error, and two things explain it. A label audit, counting swing cycles straight off the thigh gyro, matches what I typed on every walk except the first test take: 40 cycles for a typed 30, and held out the net says 37.8, so it counts what the sensor saw. The ± band is set by the 7-step take, where 3.7 steps off is 53 %. And with six walks each fold trains on five, so the held-out number is noisy: six variants of the net and augmentation, and dropping the suspect takes, all scored the same or worse. More counted walks is the only lever.

The first version learned time spent moving times an average cadence: it fit four training totals and failed on any new pace. Adding the standing and air-pedalling stretches of the earlier guided sessions as exact zeros, plenty of leg motion and no steps, forced it onto footfalls and halved the walk error without a new recording.

## Figures

- `steps_scatter.png`: predicted against typed count, one point per held-out take; the diagonal is perfect.
- `steps_errors.png`: the signed error per take, which shows the two label outliers.
- `steps_rate.png`: the rate curve over a walk; one spike per footfall and flat at zero while standing.
- `steps_training.png`: the loss falling over the 400 passes.

## Demo takes

Recorded after training was frozen, so the net had never seen them. These are the takes in the video; the numbers are what the phone showed live.

| take | true | predicted | error |
|---|---|---|---|
| demo 1 | 24 | 26.1 | +2.1 |
| demo 2 | 19 | 20.9 | +1.9 |
| demo 3 | 20 | 21.5 | +1.5 |

All within about 2 steps, with a slight over-count of about 8 %.

## Video captions

- **Demo video** (phone screen recording, 24 s, demo take 3): starts at 0, 20 counted footfalls, the count climbs one per step with a spike in the rate trace under each, and ends at 21 ± 10. Standing before and after holds it still.
- **Animation** (`stepnet_explainer_small.mp4`, 88 s, made with Manim by Claude): from the sensor traces through decimation, the three convolution layers and the rate curve to the loss, back-propagation and the weights file landing on the board.

## GitHub

https://github.com/rafaeshafi/Me193/tree/main/P6-Gait-Wearable

## Reflection questions

*(answers go here)*
