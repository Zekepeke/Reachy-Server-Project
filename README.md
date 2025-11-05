# Reachy Server Project

A lightweight FastAPI + Uvicorn backend that streams live video from the Raspberry Pi 5 camera using the **Picamera2** and **libcamera** stack.  
Built to support my DIY Reachy Mini–style project with MediaPipe/OpenCV gesture recognition and future LLM/TTS integration. Works with the frontend:
https://github.com/Zekepeke/ReachyCloneWebApp

---

## Features
- Serves **live MJPEG video stream** over HTTP.
- Integrates with **FastAPI** for a clean REST interface.
- Leverages **Picamera2** (libcamera) for Raspberry Pi 5 camera capture.
- Ready for **MediaPipe + Torch**–based hand/gesture recognition.
- Designed to run on Raspberry Pi, with Python virtual environment on external storage to avoid SD card space issues.

---

## Requirements
System packages (install via `apt`):
```bash
sudo apt update
sudo apt install -y python3-venv python3-libcamera python3-picamera2 libcamera-apps libcap-dev
```

Python dependencies (installed via pip inside a venv):
- `fastapi`
- `uvicorn[standard]`
- `mediapipe`
- `torch`
- `opencv-python-headless`

See [`requirements.txt`](requirements.txt) for exact versions.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Zekepeke/Reachy-Server-Project.git
cd Reachy-Server-Project

# Create a venv (put it on your big drive if your SD is small)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

---

## ▶Running the Server

```bash
# Activate environment
source .venv/bin/activate

# Run FastAPI server with Uvicorn
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open your browser at:  
`http://<raspberry-pi-ip>:8000/mjpeg`

---

## Project Structure
```
Reachy-Server-Project/
├── server/
│   ├── app.py                    # FastAPI entrypoint
│   ├── pipeline.py               # Hand landmark / gesture processing
│   ├── landmarks_points_classifier.py
│   └── label_map.json
├── model/                        # ML models
├── hand_landmarker.task          # MediaPipe hand model
├── requirements.txt
└── README.md
```

---

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'libcamera'` → make sure `python3-libcamera` and `python3-picamera2` are installed via `apt`.
- If pip errors with `No space left on device` → move your `.venv` to your external drive (`/dev/sdb2`) and symlink it back.

---

## Running Later
When you reboot or come back later, just do:

```bash
# 1) Go to your project
cd ~/Computer-Vision-Project

# 2) Activate the venv that lives on the big drive
source .venv/bin/activate

# 3) Run your server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Future Work
- Integrate custom **gesture recognition model**.
- Connect to a local **LLM/TTS agent** for voice interaction.
- Auto-start service on boot with `systemd`.

---

## 📜 License
MIT License © 2025 Esequiel Linares (Zeke)
