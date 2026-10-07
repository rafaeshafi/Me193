# Journal

One line per surprise, failure or fix. This is the raw material for the Notion
reflection bullets (What we learned / most proud of / What took a while / Most
spectacular failure), so write it down when it happens, in your own words.

- 2026-10-06 — The plan went through two adversarial reviews. The schedule reviewer said my first compressed plan was "not realistic as written" (G0 by Thursday only ~35–45% likely); the plan was rebuilt around 4.5 h/days, Claude night shifts and a clock ladder.
- 2026-10-06 — Found a `P4-Door-To-Door-Service` folder committed at 19:56 by another session, so this project is **P5**, and P4 may compete for the same hours.
- 2026-10-06 — A magnitude-only swing detector fires on the *backswing* 100% of the time in a reviewer's simulation; the detector projects onto a learned signed forward axis instead.
- 2026-10-06 — Bluetooth cannot be used from the Claude desktop app's shell (macOS aborts the process, TCC); all BLE/camera runs happen in Terminal.app, and the tools now say so instead of crashing.
- 2026-10-06 — My own `FakeEnv` quantised the hub rate (66 Hz measured as 60, 33 Hz as 20) because it dropped fractional time between sleeps; the env_check tests caught it before any hardware did.
- 2026-10-06 — The first whole-pipeline run on fake hardware passed 14 of 14 tests at once, which I did not trust. Mutating the wiring (no haptic blank windows, no poses fed to the judge) made exactly the right tests fail, so they were measuring something.
- 2026-10-06 — The fake rally showed that in a first rally every hit is technically a "new record", so the 5-pulse record buzz replaced the hit-quality cue on every single hit. The record cue now fires once per rally, only when a previous best is passed.
- 2026-10-06 — A hub that went quiet for the stale limit (300 ms) was scored as a *miss*: the miss deadline passed on the wall clock before the pause started. Two fixes: a miss now waits until the IMU data has reached its deadline, and a pause is back-dated to the last sample, so the ball resumes where the player last saw it.
- 2026-10-06 — The haptics actuator held its lock during BLE writes, so the 60 Hz game loop could stall behind a buzz. Split into a schedule lock and a device lock; a test with a deliberately slow device pins it.
- 2026-10-06 — A swing plus its backswing looks periodic, so a bare FFT "shake" check would lock the paddle after normal swings. The shake rule needs a dominant 3-8 Hz bin, a narrow peak, enough amplitude and at least four sign reversals; white noise lands its loudest bin in the band about one time in five, and only the narrow-peak test stops it (checked by mutation: 92 false locks without it).
