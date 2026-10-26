# Notion page: humanised text (2026-10-26)

Sections the user wrote (intro, London 2012, reflections) are theirs and stay untouched. These replace the sections I wrote.
Headings get emojis; page icon 🏃.

## ⚙️ So how does it all work

- **Sensor to board.** A little motion chip on my thigh takes 100 readings a second and tucks them into its own buffer. The microcontroller scoops them out and hands them to the board's Linux side, ten at a time.
- **Windows and features.** Every half second the board grabs the last two seconds of leg wobble and boils it down to a handful of numbers: the biggest jolt along my thigh, how tilted the thigh is, how much it's shaking along and across, how fast it's spinning, how many swings per second, and a "spin versus impact" ratio.
- **Two trees.** Model 1 only gets one of those numbers, the jolt. Model 2 gets the lot, but it opens with one hard-coded question: is the leg even moving? If the gyro says no, it's STAND, however I happen to be lying. After that it only has to tell WALK, RUN and BIKE apart.
- **Vote and display.** The last three answers vote, so the label doesn't flicker every time a stride lands on a window edge. The phone page shows both verdicts, whether they agree, my cadence, the peak g, and a live trace of my thigh.
- **Training loop.** The same page records me: it counts me through STAND, WALK, RUN and BIKE while I actually do them. I pull the files over USB, scikit-learn grows the two trees using the exact feature code the board runs, and the trees go back as two tiny JSON files.

## 🧠 Which model is better, and why

Model 2, and it isn't close. The reason is physics, not statistics. Peak thigh acceleration is really just a measure of impact: standing has none (0.8 g, that's gravity), walking gives a small heel-strike bump around 1.3 g, running a big one around 2.2 g. Biking has no impacts at all, so it sits at 1.2 g, right on top of walking. One number can't tell "leg gently hitting the ground" from "leg going round in circles with no ground". In my data Model 1 called 76 % of my walking windows BIKE, and the only reason its biking score looks fine (84 %) is that it found a thin sliver between standing and walking and labelled it BIKE. Cheeky, not smart.

Model 2 doesn't need an impact to spot biking. Two things give it away: my thigh goes horizontal (gravity along the thigh drops from 1 g to about 0.3 g) and the gyro traces a smooth sine wave once per pedal stroke instead of the spiky bursts of a step. The tree it learned reads like a flow chart: is the leg moving? is there a big impact (RUN)? is the thigh horizontal (BIKE)? otherwise WALK.

The honest cost: Model 2 needs nine numbers and a tree you can't set by hand, and its first version was wrong in a way I found pretty funny. Trained on just one session, it learned that "walking" means "leaning slightly forward", because in that one recording the lean separated standing from walking just as well as motion did. Tested on a second session it fell to 77 %. The fix was throwing out the orientation-dependent features and forcing "is it moving?" to the top of the tree. Bonus: a friend can wear it with the sensor twisted a bit and it still works.

## 📊 Figures

- `features.png`: on the left, the one thing Model 1 ever sees, with WALK and BIKE sitting on top of each other; on the right the same windows against Model 2's first learned split, where BIKE drifts off on its own.
- `tree_model1.png`: Model 1, four leaves on a single number.
- `tree_model2.png`: the learned half of Model 2, the part that runs after the "is it moving?" gate.
- `confusion.png`: both models scored on recordings they'd never seen. Rows are what I was actually doing, columns what the model guessed.

## 🎬 Demo video

(the video block; file: Desktop/P6 demo (4.5 MB for Notion).mp4, 1024x672, 91 s)

## Other headings

🏊‍♂️ Training for an Ironman (user's) · 🎮 London 2012 (user's) · 💻 GitHub · 💭 Reflection questions (user's)
