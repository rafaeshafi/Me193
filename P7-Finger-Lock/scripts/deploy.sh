#!/usr/bin/env bash
# Deploy the finger lock to the UNO Q over the USB cable (App Lab's adb) and (re)start it.
#   app/ + lock/ + models/fingers.json + config.json  ->  ~/ArduinoApps/finger-lock  (/app in the app container)
# The board's data/ (log, intruder pictures) and .cache/ are kept. Any other running user app is stopped (App Lab runs
# one at a time) and this app is made the default so it comes up on power alone. First start compiles ~2 min.
#   Page:  adb forward tcp:8080 tcp:8080  ->  http://localhost:8080      Logs: adb shell tail -f '~/ArduinoApps/finger-lock/data/app.log'
set -euo pipefail
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
APP_DIR='~/ArduinoApps/finger-lock'
cd "$(dirname "$0")/.."
[ -f models/fingers.json ] || { echo "models/fingers.json missing: train first (.venv/bin/python train/train.py)"; exit 1; }

remote() { "$ADB" shell "$1"; }

STAGE="$(mktemp -d -t lock-stage)"
TGZ="$(mktemp -t lock-app).tgz"
trap 'rm -rf "$STAGE" "$TGZ"' EXIT
cp -R app/. "$STAGE"/
mkdir -p "$STAGE/python/lock" "$STAGE/models"
cp lock/*.py "$STAGE/python/lock/"
cp models/fingers.json "$STAGE/models/"
cp config.json "$STAGE/"
COPYFILE_DISABLE=1 tar -C "$STAGE" --exclude .cache --exclude __pycache__ -czf "$TGZ" .
# adb shell does not carry stdin reliably: push the tarball, then unpack it.
"$ADB" push "$TGZ" /tmp/lock-app.tgz
remote "mkdir -p $APP_DIR && cd $APP_DIR &&
    find . -mindepth 1 -maxdepth 1 ! -name .cache ! -name data -exec rm -rf {} + &&
    tar --warning=no-timestamp -xzf /tmp/lock-app.tgz && rm /tmp/lock-app.tgz"

remote "arduino-app-cli app list | awk '\$1 ~ /^user:/ && \$0 ~ / running / && \$1 != \"user:finger-lock\" {sub(/^user:/, \"\", \$1); print \$1}' |
  while read -r other; do echo \"stopping \$other\"; arduino-app-cli app stop ~/ArduinoApps/\$other; done"
remote "arduino-app-cli app restart $APP_DIR"
remote "arduino-app-cli properties set default $APP_DIR" || echo "could not set the default app (older arduino-app-cli?)"
echo "deployed; first start compiles the sketch (~2 min). Page: $ADB forward tcp:8080 tcp:8080 -> http://localhost:8080"
