#!/usr/bin/env bash
# Render the explainer: 1080p for the repo/YouTube and a small 480p cut for Notion (free plan: 5 MB per upload).
set -euo pipefail
cd "$(dirname "$0")"
MANIM="${MANIM:-../../.venv/bin/manim}"
"$MANIM" -qh --media_dir media -o stepnet_explainer.mp4 stepnet_explainer.py StepNetExplainer
"$MANIM" -ql --media_dir media -o stepnet_explainer_small.mp4 stepnet_explainer.py StepNetExplainer
cp media/videos/stepnet_explainer/1080p60/stepnet_explainer.mp4 . 2>/dev/null || cp media/videos/stepnet_explainer/1080p30/stepnet_explainer.mp4 .
cp media/videos/stepnet_explainer/480p15/stepnet_explainer_small.mp4 .
ls -la stepnet_explainer*.mp4
