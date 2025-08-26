# Converts Mediapipe 21-hand landmarks to a normalized feature vector
# (63 floats: (x,y,z)*21), wrist-centered, scale-normalized

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

NUM_POINTS = 21
WRIST = 0
MIDDLE_MCP = 9  # used for scale estimate with wrist

def mp_landmarks_to_np(landmarks):
    # landmarks: mediapipe.tasks.vision.HandLandmarkerResult...
    # We take the first detected hand only for classification.
    if not landmarks.handedness or not landmarks.hand_landmarks:
        return None
    pts = landmarks.hand_landmarks[0]  # list of 21 NormalizedLandmark
    arr = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in pts], dtype=np.float32)  # (21,3)

    # center on wrist
    center = arr[WRIST].copy()
    arr[:, :2] -= center[:2]

    # scale by wrist→middle_mcp distance (robust-ish)
    scale = np.linalg.norm(arr[MIDDLE_MCP, :2])
    if scale < 1e-6:
        scale = 1.0
    arr[:, :2] /= scale

    # (21*3,) flat vector
    return arr.reshape(-1)