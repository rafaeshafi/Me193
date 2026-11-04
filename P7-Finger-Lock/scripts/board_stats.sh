#!/usr/bin/env bash
# Pull the board's timing summary (the 'UNO Q vs computer' table) over the USB cable into docs/bench_board.json.
set -euo pipefail
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
cd "$(dirname "$0")/.."
"$ADB" forward tcp:18080 tcp:8080 >/dev/null
curl -s http://localhost:18080/stats | python3 -m json.tool > docs/bench_board.json
python3 - <<'PY'
import json; s = json.load(open("docs/bench_board.json"))
for k in ("capture", "preprocess", "inference", "loop"):
    v = s.get(k); print(f"{k:12s}", v and f"mean {v['mean_ms']} ms  p95 {v['p95_ms']} ms  n={v['n']}")
print("fps", s.get("fps"), "machine", s.get("machine"))
PY
