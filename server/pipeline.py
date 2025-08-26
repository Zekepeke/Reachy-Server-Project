# server/pipeline.py
import copy
import itertools
from typing import List

import cv2 as cv
import numpy as np
import mediapipe as mp

# Mediapipe Tasks imports
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class Pipeline:
    """
    Hand-landmark pipeline.

    Usage:
      pipe = Pipeline(model_path="hand_landmarker.task")
      out_jpg = pipe(in_jpg_bytes)

    - Keeps a single HandLandmarker instance alive.
    - VIDEO mode is used for steady per-frame calls.
    """

    def __init__(self, model_path: str = "hand_landmarker.task") -> None:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0  # must strictly increase

    def __call__(self, frame_bytes: bytes) -> bytes:
        """Decode JPEG -> detect -> draw -> encode JPEG."""
        # Decode
        np_buf = np.frombuffer(frame_bytes, dtype=np.uint8)
        bgr = cv.imdecode(np_buf, cv.IMREAD_COLOR)
        if bgr is None:
            return frame_bytes  # not a valid JPEG, pass-through

        # Mirror (like webcam preview)
        bgr = cv.flip(bgr, 1)
        debug_image = bgr.copy()

        # MediaPipe expects RGB
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 33  # ≈30 FPS; any strictly increasing ms is fine

        # Draw
        if result and result.hand_landmarks:
            for lm in result.hand_landmarks:
                landmark_list = self.calc_landmark_list(debug_image, lm)
                debug_image = self.draw_landmarks(debug_image, landmark_list)

        # Encode back to JPEG
        ok, enc = cv.imencode(".jpg", debug_image, [cv.IMWRITE_JPEG_QUALITY, 90])
        return enc.tobytes() if ok else frame_bytes

    # ---------- helpers (adapted from your functions) ----------

    @staticmethod
    def calc_bounding_rect(image, landmarks) -> List[int]:
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for landmark in landmarks:
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            landmark_array = np.append(landmark_array, [[x, y]], axis=0)
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    @staticmethod
    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        pts = []
        for lm in landmarks:
            x = min(int(lm.x * image_width), image_width - 1)
            y = min(int(lm.y * image_height), image_height - 1)
            pts.append([x, y])
        return pts

    @staticmethod
    def pre_process_landmark(landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0]
        for i, (x, y) in enumerate(temp):
            temp[i][0] = x - base_x
            temp[i][1] = y - base_y
        temp = list(itertools.chain.from_iterable(temp))
        max_value = max(map(abs, temp)) or 1.0
        return [v / max_value for v in temp]

    @staticmethod
    def draw_landmarks(image, landmarks, color=(102, 25, 179)):
        # Bones
        if len(landmarks) > 0:
            # Thumb
            cv.line(image, tuple(landmarks[2]), tuple(landmarks[3]), (0, 0, 0), 6)
            cv.line(image, tuple(landmarks[2]), tuple(landmarks[3]), (255, 255, 255), 2)
            cv.line(image, tuple(landmarks[3]), tuple(landmarks[4]), (0, 0, 0), 6)
            cv.line(image, tuple(landmarks[3]), tuple(landmarks[4]), (255, 255, 255), 2)
            # Index
            for a, b in [(5, 6), (6, 7), (7, 8)]:
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (0, 0, 0), 6)
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (255, 255, 255), 2)
            # Middle
            for a, b in [(9, 10), (10, 11), (11, 12)]:
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (0, 0, 0), 6)
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (255, 255, 255), 2)
            # Ring
            for a, b in [(13, 14), (14, 15), (15, 16)]:
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (0, 0, 0), 6)
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (255, 255, 255), 2)
            # Pinky
            for a, b in [(17, 18), (18, 19), (19, 20)]:
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (0, 0, 0), 6)
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (255, 255, 255), 2)
            # Palm
            for a, b in [(0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]:
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (0, 0, 0), 6)
                cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]), (255, 255, 255), 2)

        # Joints
        for i, (x, y) in enumerate(landmarks):
            if i in (4, 8, 12, 16, 20):
                cv.circle(image, (x, y), 8, (84, 157, 138), -1)
                cv.circle(image, (x, y), 8, (0, 0, 0), 1)
            else:
                cv.circle(image, (x, y), 5, color, -1)
                cv.circle(image, (x, y), 5, (0, 0, 0), 1)
        return image