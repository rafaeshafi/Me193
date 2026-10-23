#!/usr/bin/env bash
# Compile the sketch on the Mac with App Lab's own arduino-cli (~20 s) instead of on the board (~2 min).
set -euo pipefail
cd "$(dirname "$0")/.."
WS="$HOME/Library/Application Support/arduino_applab_workspace"
CLI="$WS/resources/arduino/arduino-cli/arduino-cli"
"$CLI" --config-file "$WS/arduino15/arduino-cli.yaml" compile --fqbn arduino:zephyr:unoq \
  --build-path /tmp/claude-501/gait-build app/sketch "$@"
