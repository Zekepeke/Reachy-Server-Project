# models.py
from typing import Union, Optional, Tuple, List
import numpy as np
import cv2 as cv
import mediapipe as mp

# MP Tasks
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class FaceLandmarksModel:
    """
    Generalized face-landmark annotator.

    Accepts frames as bytes (encoded), np.ndarray (BGR/RGB), or mp.Image.
    Returns JPEG bytes by default OR BGR/RGB ndarray.
    Can also return per-face pixel landmarks.

    Example:
        model = FaceLandmarksModel("face_landmarker.task")
        bgr = model(frame_ndarray, out_format="bgr", return_landmarks=False)
    """

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        *,
        num_faces: int = 2,
        running_mode: mp_vision.RunningMode = mp_vision.RunningMode.VIDEO,
    ) -> None:
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            num_faces=num_faces,
            running_mode=running_mode,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._timestamp_ms = 0  # must strictly increase in VIDEO mode

    # ---------- public API ----------
    def __call__(
        self,
        frame: Union[bytes, np.ndarray, "mp.Image"],
        *,
        mirror: bool = True,
        draw: bool = True,
        jpeg_quality: int = 90,
        assume_bgr: bool = True,
        out_format: str = "jpeg_bytes",  # "jpeg_bytes" | "bgr" | "rgb"
        return_landmarks: bool = False,
        timestamp_ms: Optional[int] = None,
    ) -> Union[bytes, np.ndarray, Tuple[Union[bytes, np.ndarray], List[List[Tuple[int, int]]]]]:
        """
        Runs face landmarking and returns the annotated frame (and optionally landmarks).

        return_landmarks → [[(x,y), ...] per face]
        """
        # 1) Normalize input to BGR
        bgr = self._to_bgr(frame, assume_bgr=assume_bgr)
        if bgr is None:
            if isinstance(frame, (bytes, bytearray)) and out_format == "jpeg_bytes":
                # pass-through on undecodable bytes if caller wanted bytes
                return bytes(frame)
            raise ValueError("Failed to decode/normalize input frame")

        if mirror:
            bgr = cv.flip(bgr, 1)

        debug = bgr.copy()

        # 2) Run MediaPipe (expects RGB)
        mp_image = self._to_mp_image(bgr)
        ts = self._timestamp_ms if timestamp_ms is None else int(timestamp_ms)
        result = self.landmarker.detect_for_video(mp_image, ts)
        # keep time monotonic
        self._timestamp_ms = max(ts + 1, self._timestamp_ms + 1)

        # 3) Draw + collect landmarks (pixel coords)
        all_faces_px: List[List[Tuple[int, int]]] = []
        if result and result.face_landmarks:
            for face in result.face_landmarks:
                lm_px = self._landmarks_to_pixels(debug, face)
                all_faces_px.append(lm_px)
                if draw:
                    self._draw_face(debug, lm_px)

        # 4) Format output
        if out_format == "jpeg_bytes":
            ok, enc = cv.imencode(".jpg", debug, [cv.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            output = enc.tobytes() if ok else b""
        elif out_format == "bgr":
            output = debug
        elif out_format == "rgb":
            output = cv.cvtColor(debug, cv.COLOR_BGR2RGB)
        else:
            raise ValueError("out_format must be 'jpeg_bytes', 'bgr', or 'rgb'")

        return (output, all_faces_px) if return_landmarks else output

    # ---------- internal helpers ----------
    @staticmethod
    def _to_bgr(frame: Union[bytes, np.ndarray, "mp.Image"], assume_bgr: bool = True) -> Optional[np.ndarray]:
        # bytes → decode
        if isinstance(frame, (bytes, bytearray)):
            buf = np.frombuffer(frame, dtype=np.uint8)
            return cv.imdecode(buf, cv.IMREAD_COLOR)

        # ndarray
        if isinstance(frame, np.ndarray):
            arr = frame
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    scale = 255.0 if arr.max() <= 1.0 else 1.0
                    arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                return cv.cvtColor(arr, cv.COLOR_GRAY2BGR)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return arr if assume_bgr else cv.cvtColor(arr, cv.COLOR_RGB2BGR)
            return None

        # mp.Image
        try:
            from mediapipe import Image as MPImage  # type: ignore
        except Exception:
            MPImage = None
        if MPImage is not None and isinstance(frame, MPImage):
            rgb = frame.numpy_view()
            if rgb.ndim == 3 and rgb.shape[2] == 3:
                return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            if rgb.ndim == 2:
                return cv.cvtColor(rgb, cv.COLOR_GRAY2BGR)
            return None

        return None

    @staticmethod
    def _to_mp_image(bgr: np.ndarray) -> "mp.Image":
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    @staticmethod
    def _landmarks_to_pixels(image: np.ndarray, landmarks) -> List[Tuple[int, int]]:
        h, w = image.shape[:2]
        pts: List[Tuple[int, int]] = []
        for lm in landmarks:
            x = min(max(int(lm.x * w), 0), w - 1)
            y = min(max(int(lm.y * h), 0), h - 1)
            pts.append((x, y))
        return pts

    @staticmethod
    def _draw_face(image: np.ndarray, pts: List[Tuple[int, int]]) -> None:
        """
        Lightweight overlay: dots + bounding box. (Fast on Pi 5)
        If you want fancier meshes, connect subsets or use convex hulls.
        """
        if not pts:
            return

        # points
        for (x, y) in pts:
            cv.circle(image, (x, y), 1, (255, 255, 255), -1)   # white dot
            cv.circle(image, (x, y), 1, (0, 0, 0), 1)          # outline

        # bounding box
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        cv.rectangle(image, (x0, y0), (x1, y1), (84, 157, 138), 2)