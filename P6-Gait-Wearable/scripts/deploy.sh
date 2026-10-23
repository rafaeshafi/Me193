#!/usr/bin/env bash
# Deploy the gait wearable to the UNO Q over the USB cable (App Lab's adb) and (re)start it.
#   app/ + models/  ->  ~/ArduinoApps/gait-wearable   (mounted at /app in the app container; first start compiles ~2 min)
# Recordings in ~/ArduinoApps/gait-wearable/data are kept. Any other running user app is stopped (App Lab runs one at a
# time) and this app is made the default so it comes up on power alone.
# Logs:  adb shell tail -f '~/ArduinoApps/gait-wearable/data/app.log'
set -euo pipefail
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
APP_DIR='~/ArduinoApps/gait-wearable'
cd "$(dirname "$0")/.."

remote() { "$ADB" shell "$1"; }

STAGE="$(mktemp -d -t gait-stage)"
TGZ="$(mktemp -t gait-app).tgz"
trap 'rm -rf "$STAGE" "$TGZ"' EXIT
cp -R app/. "$STAGE"/
mkdir -p "$STAGE/models" && cp models/model1.json models/model2.json "$STAGE/models/"
COPYFILE_DISABLE=1 tar -C "$STAGE" --exclude .cache --exclude __pycache__ -czf "$TGZ" .
# adb shell does not carry stdin reliably: push the tarball, then unpack it.
"$ADB" push "$TGZ" /tmp/gait-app.tgz
remote "mkdir -p $APP_DIR && cd $APP_DIR &&
    find . -mindepth 1 -maxdepth 1 ! -name .cache ! -name data -exec rm -rf {} + &&
    tar --warning=no-timestamp -xzf /tmp/gait-app.tgz && rm /tmp/gait-app.tgz"

remote "arduino-app-cli app list | awk '\$1 ~ /^user:/ && \$0 ~ / running / && \$1 != \"user:gait-wearable\" {sub(/^user:/, \"\", \$1); print \$1}' |
  while read -r other; do echo \"stopping \$other\"; arduino-app-cli app stop ~/ArduinoApps/\$other; done"
remote "arduino-app-cli app restart $APP_DIR"
remote "arduino-app-cli properties set default $APP_DIR" || echo "could not set the default app (older arduino-app-cli?)"
echo "deployed; first start compiles the sketch (~2 min). Logs: $ADB shell tail -f '$APP_DIR/data/app.log'"
