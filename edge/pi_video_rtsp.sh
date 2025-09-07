#!/usr/bin/env bash
set -euo pipefail

# Start RTSP server (once per device)
# docker run -d --name rtsp -p 8554:8554 aler9/rtsp-simple-server:latest

# Stream camera → RTSP server
libcamera-vid -t 0 \
  --width 1280 --height 720 --framerate 30 \
  --codec h264 --inline --profile high --level 4.2 --bitrate 2000000 \
  --listen -o - | ffmpeg -re -i - -c copy -f rtsp rtsp://CLOUD_HOST:8554/reachy