#!/usr/bin/env bash
# Copy the board's recordings into ./data (gitignored) over the USB cable.
set -euo pipefail
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
cd "$(dirname "$0")/.."
mkdir -p data
"$ADB" pull /home/arduino/ArduinoApps/gait-wearable/data/. data/
ls -la data/*.csv | tail -20
