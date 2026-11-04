# Finger Lock: implementation plan

Spec: `docs/superpowers/specs/2026-11-04-finger-lock-design.md`. Each task names its files, what it does and how it
is verified. Tasks 1-11 are built here; tasks 12-15 need the user's hands, webcam and board.

| # | Task | Files | Verify |
|---|---|---|---|
| 1 | Scaffold: layout, venv, `config.json`, `requirements.txt`, `.gitignore`, `tests/conftest.py` | root | `test_smoke` |
| 2 | Shared config + preprocessing: ROI maths, grayscale 64x64, standardise | `lock/config.py`, `lock/preprocess.py` | `test_preprocess` |
| 3 | Lock policy state machine (accept / release / timeout / deny / unlock / lockout / intruder hook) | `lock/policy.py` | `test_policy` |
| 4 | Matrix renderer: 3x5 digits, slots, progress bar, tick, cross, skull, 104-byte frames | `lock/matrix.py` | `test_matrix` |
| 5 | PyTorch FingerNet + JSON export + numpy engine with parity | `train/model.py`, `train/export.py`, `lock/cnn_np.py` | `test_cnn` |
| 6 | Dataset: loader by session, augmentation (cv2 warps), synthetic hands generator | `train/dataset.py`, `train/synth.py` | `test_dataset` |
| 7 | Training script: loop, curves, confusion, samples, results.md, meta, `--synth`, `--no-aug`, `--ablate` | `train/train.py`, `train/plots.py` | `train.py --synth --epochs 3` + `test_train` |
| 8 | Capture tool and Mac lock app (torch/numpy engines, on-screen LEDs + matrix, saliency, intruder log) + headless bench | `tools/capture.py`, `tools/lock_app.py`, `tools/bench.py` | `bench.py --synth`, `test_lock_app` (headless pieces) |
| 9 | UNO Q app: `app.yaml`, `main.py` (camera, numpy net, policy, Bridge), `web.py` + `page.html`, sketch (button, draw, leds, servo), `deploy.sh`, `compile.sh` | `app/`, `scripts/` | `test_web`, `compile.sh` |
| 10 | Stop-action movie: frame generator with real weights, build script, voice-over script | `docs/movie/` | frames render, mp4 plays |
| 11 | Write-up: README, `docs/notion.md` (4 midterm sections + template Qs), `docs/comparison.md`, commit | `README.md`, `docs/` | read-through |
| 12 | (user) Capture ~150 pictures per class over 3+ sessions, light variation; `train.py`; judge `docs/results.md` | `data/raw/` | val accuracy ≥ 95 % |
| 13 | (user) Wire button/LEDs/servo, `scripts/deploy.sh`, test on the board, read `/stats`, fill the board column | board | unlock on the real box |
| 14 | (user) Record voice-over over `movie_silent.mp4`; film the demo; Notion page from `docs/notion.md` | Notion | page complete |
| 15 | (user) Reflection answers in own words; push | GitHub | pushed |

## Conventions (from P5/P6)

- One venv per project (`.venv`, Python 3.12 from Homebrew), gitignored, rebuilt from `requirements.txt`.
- Board Python: numpy + OpenCV + PIL are in the App Lab container; no torch. Everything in `lock/` must import on
  both machines; `lock/preprocess.py` falls back to PIL when cv2 is missing.
- Tests: `.venv/bin/python -m pytest tests -q`; `tests/conftest.py` puts the repo root, `train/` and `app/python` on
  the path.
- Deploy over the USB cable with App Lab's adb (push a tarball, then unpack); keep `data/` on the board; make the app
  the default so it starts on power.
- Commit as work lands; never push unless asked; no attribution trailer.
