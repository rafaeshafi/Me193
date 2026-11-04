#!/usr/bin/env bash
# Render the frames and join them into movie_silent.mp4 (H.264 if OpenCV can, else mp4v). Then record the voice-over
# over it with QuickTime (File > New Screen Recording, microphone on) or Zoom, reading docs/movie/script.md.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../../.venv/bin/python}"
"$PY" frames.py "$@"
"$PY" - <<'EOF2'
import glob, cv2
frames = sorted(glob.glob("frames/f*.png"))
h, w = cv2.imread(frames[0]).shape[:2]
out = None
for codec in ("avc1", "mp4v"):
    out = cv2.VideoWriter("movie_silent.mp4", cv2.VideoWriter_fourcc(*codec), 12, (w, h))
    if out.isOpened():
        break
for f in frames:                      # 3 frames a second, each written 4 times at 12 fps (every player likes 12 fps)
    img = cv2.imread(f)
    for _ in range(4):
        out.write(img)
out.release()
print(f"movie_silent.mp4: {len(frames)} frames, {len(frames) / 3:.0f} s, {w}x{h}, codec {codec}")
EOF2
