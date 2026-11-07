#!/usr/bin/env bash
# Train on the pictures in data/raw, export the weights, deploy to the UNO Q if it is on the USB cable, and say how
# it went. One command after a capture session, or left running to do it by itself:
#
#   scripts/auto_train.sh            train + export + deploy once (refuses politely if there are too few pictures)
#   scripts/auto_train.sh --watch    every time tools/capture.py quits, do the above; Ctrl-C to stop
#   scripts/auto_train.sh --movie    also re-render docs/movie with the new weights (slow, a few minutes)
#
# Needs: every label 0..5 with at least MIN_PER_CLASS pictures and at least 2 capture sessions (validation holds out
# whole sessions). Training is train/train.py --ablate --deploy best (with and without augmentation; the better one is deployed).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
MIN_PER_CLASS="${MIN_PER_CLASS:-40}"
EPOCHS="${EPOCHS:-60}"
AUG="${AUG_STRENGTH:-0.5}"   # augmentation strength; 1.0 was too wild for the real pictures (train acc below val acc)
LOG=data/auto_train.log
WATCH=0; MOVIE=0
for arg in "$@"; do case "$arg" in --watch) WATCH=1;; --movie) MOVIE=1;; esac; done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; echo "$(date '+%F %T') $*" >> "$LOG"; }

enough() {  # prints the per-label counts; returns 0 when there is enough to train
    "$PY" - "$MIN_PER_CLASS" <<'EOF'
import re, sys
from pathlib import Path
need = int(sys.argv[1]); raw = Path("data/raw"); ok = True; covered = True; sessions = set()
for label in range(6):
    files = list((raw / str(label)).glob("*.png")) if (raw / str(label)).exists() else []
    mine = {re.sub(r"-\d+\.png$", "", f.name) for f in files}
    sessions |= mine
    flag = "" if len(files) >= need else f"   <- need {need - len(files)} more"
    if len(mine) < 2:
        flag += "   <- in only %d session; shoot it again in another run" % len(mine)
    print(f"  {label} fingers: {len(files):4d} pictures in {len(mine)} session(s){flag}")
    ok &= len(files) >= need
    covered = covered and len(mine) >= 2
print(f"  sessions: {len(sessions)}")
if ok and not (covered and len(sessions) >= 2):
    print("  some label is in one session only, so the honest whole-session hold-out is impossible: validating on the")
    print("  last quarter of every burst instead (optimistic; shoot every label in a second session for the real number)")
sys.exit(0 if ok and covered and len(sessions) >= 2 else 3 if ok else 1)
EOF
}

train_once() {
    say "pictures in data/raw:"
    enough; rc=$?
    SPLIT=session; [ "$rc" = 3 ] && SPLIT=within
    if [ "$rc" = 1 ]; then say "not enough yet; keep taking pictures"; return 2; fi
    say "training (${EPOCHS} epochs, augmentation strength ${AUG} and none, validation split: ${SPLIT})..."
    if ! "$PY" train/train.py --ablate --deploy best --epochs "$EPOCHS" --split "$SPLIT" --aug-strength "$AUG" 2>&1 | tee -a "$LOG" | grep -E "training /|per class|^--- |epoch +(1|5|10|20|30|40)\b|deployed run|Error|error"; then
        say "training failed; see $LOG"; return 1
    fi
    say "results (docs/results.md):"
    sed -n '/^| run/,/^$/p' docs/results.md
    "$PY" - <<'EOF'
import json
m = json.load(open("models/meta.json"))
acc = m["val_accuracy"]; pc = m.get("per_class_val_accuracy") or {}
verdict = "good: ready to demo" if acc >= 0.95 else "usable, but more pictures in new lighting would help" if acc >= 0.85 else "weak: take more pictures (different hand poses, distances, light) and retrain"
print(f"  held-out accuracy {acc * 100:.1f} % -> {verdict}")
if pc:
    worst = min(pc.items(), key=lambda kv: kv[1])
    print(f"  weakest class: {worst[0]} fingers at {float(worst[1]) * 100:.0f} %")
EOF
    if "$ADB" devices 2>/dev/null | grep -q "device$"; then
        say "board on USB: deploying..."
        if bash scripts/deploy.sh >> "$LOG" 2>&1; then say "deployed; the lock on the board now uses the new weights"; else say "deploy failed; see $LOG"; fi
    else
        say "board not on USB; run scripts/deploy.sh when it is"
    fi
    if [ "$MOVIE" = 1 ]; then
        say "re-rendering the movie with the real weights (a few minutes)..."
        (cd docs/movie && bash build.sh >> "../../$LOG" 2>&1) && say "movie: docs/movie/movie_silent.mp4" || say "movie render failed; see $LOG"
    fi
    return 0
}

if [ "$WATCH" = 0 ]; then train_once; exit $?; fi

say "watching: I train each time the capture tool quits (Ctrl-C here to stop)"
while true; do
    while ! pgrep -f tools/capture.py >/dev/null; do sleep 3; done
    say "capture session running... (quit it with q when done)"
    while pgrep -f tools/capture.py >/dev/null; do sleep 3; done
    say "capture quit"
    train_once
    say "waiting for the next capture session"
done
