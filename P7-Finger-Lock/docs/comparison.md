# UNO Q vs the computer (midterm section 4)

Same webcam, same 64x64 input, same weights, same policy; the Mac runs PyTorch (or the numpy engine), the board runs
the numpy engine. Numbers are per frame; `tools/bench.py` writes the Mac's, `scripts/board_stats.sh` pulls the board's.

| | MacBook Pro (M1 Pro, 16 GB) | Arduino UNO Q (Qualcomm QRB2210, 4x Cortex-A53, 2 GB) |
|---|---|---|
| inference, PyTorch | 0.53 ms mean, 0.58 ms p95 | no PyTorch on the board |
| inference, numpy | 0.42 ms mean, 0.44 ms p95 | 22 ms mean, 18 ms median, 51 ms p95 (about 50x the Mac) |
| preprocess (crop, grey, resize, standardise) | well under 1 ms | 5.8 ms mean, 6.6 ms p95 |
| weights load (6.4 MB JSON) | instant | 0.7 s |
| loop rate | camera-bound (30 fps webcam) | camera-bound, 10 fps asked for |
| training 40 epochs on ~1,000 pictures | about a minute | not done on the board (the instructor's hint; it would take far longer and the board has no PyTorch) |
| what it took to move the model | `train/export.py` writes the weights as JSON; `lock/cnn_np.py` runs them in numpy | `scripts/deploy.sh` copies `lock/`, `models/fingers.json`, `config.json`; parity test says the two engines agree to 1e-4 |
| camera | OpenCV `VideoCapture` | `arduino.app_peripherals.camera.Camera("usb:0")` (V4L2 under the hood) |
| outputs | drawn on screen | matrix, LEDs, servo over the Bridge to the MCU |
| iteration speed | edit, run, see | deploy ~2 min for the first start (sketch compiles on the board) |

Mac numbers: `docs/bench_mac.json`, 300 inferences of random 64x64 inputs after a 10-call warm-up, 2026-11-04. Board
numbers: `docs/bench_board.json`, 200 runs inside the app container on 2026-11-07 (4x Cortex-A53, numpy 2.5.1); at 28 ms
for preprocess plus inference the board still has 70 ms to spare per frame at 10 fps, so the loop is camera-bound there too. The
Mac's loop is dominated by the camera, not the net: at 0.5 ms the net could run two thousand times a second.
