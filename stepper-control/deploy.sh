#!/usr/bin/env bash
# Deploy the stepper control app to the UNO Q over the USB cable (App Lab's adb) and (re)start it.
#   app.yaml + python/ + sketch/  ->  ~/ArduinoApps/stepper-control    page: adb forward tcp:8080 tcp:8080 -> http://localhost:8080
# Any other running user app (the finger lock) is stopped: App Lab runs one at a time. The default app is left alone,
# so the lock still comes up on power alone; run P7-Finger-Lock/scripts/deploy.sh to put the lock back now.
set -euo pipefail
ADB="${ADB:-$HOME/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb}"
APP_DIR='~/ArduinoApps/stepper-control'
cd "$(dirname "$0")"

remote() { "$ADB" shell "$1"; }

TGZ="$(mktemp -t stepper-app).tgz"
trap 'rm -f "$TGZ"' EXIT
COPYFILE_DISABLE=1 tar --exclude .cache --exclude __pycache__ --exclude deploy.sh -czf "$TGZ" app.yaml python sketch
"$ADB" push "$TGZ" /tmp/stepper-app.tgz
remote "mkdir -p $APP_DIR && cd $APP_DIR &&
    find . -mindepth 1 -maxdepth 1 ! -name .cache -exec rm -rf {} + &&
    tar --warning=no-timestamp -xzf /tmp/stepper-app.tgz && rm /tmp/stepper-app.tgz"

remote "arduino-app-cli app list | awk '\$1 ~ /^user:/ && \$0 ~ / running / && \$1 != \"user:stepper-control\" {sub(/^user:/, \"\", \$1); print \$1}' |
  while read -r other; do echo \"stopping \$other\"; arduino-app-cli app stop ~/ArduinoApps/\$other; done"
remote "arduino-app-cli app restart $APP_DIR"
echo "deployed; first start compiles the sketch (~2 min). Page: $ADB forward tcp:8080 tcp:8080 -> http://localhost:8080"
