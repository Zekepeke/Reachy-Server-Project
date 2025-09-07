#!/usr/bin/env bash
set -euo pipefail

# Requires: yq, libcamera-vid, ffmpeg
CLOUD_HOST=$(yq '.cloud.host' edge/config.yaml)
URL="rtsp://${CLOUD_HOST}:8554/reachy"

echo "[video] publishing to ${URL}"
libcamera-vid -t 0 \
  --width 1280 --height 720 --framerate 30 \
  --codec h264 --inline --profile high --level 4.2 --bitrate 2000000 \
  --listen -o - \
| ffmpeg -hide_banner -loglevel warning -re -i - -c copy -f rtsp "$URL"