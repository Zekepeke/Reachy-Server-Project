from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from threading import Condition
import io, time

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

from model.models import FaceLandmarksModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

pipe = FaceLandmarksModel()

class MJPEGHub(io.BufferedIOBase):
    """
    Broadcast hub:
      - Annotates exactly once per incoming frame.
      - Stores latest annotated JPEG bytes + seq.
      - Wakes all clients; each pulls independently.
    """
    def __init__(self):
        self._cv = Condition()
        self._frame: bytes | None = None
        self._seq: int = 0

    def write(self, buf: bytes) -> int:
        # Copy encoder buffer to immutable bytes
        cam_jpeg = bytes(buf)
        # Annotate ONCE here (not per client)
        try:
            annotated = pipe(cam_jpeg, out_format="jpeg_bytes", jpeg_quality=80)
        except Exception:
            # Fallback: pass through raw camera frame (don’t flip raw bytes)
            annotated = cam_jpeg

        with self._cv:
            self._frame = annotated
            self._seq += 1
            self._cv.notify_all()
        return len(buf)

    def snapshot_new(self, last_seq: int, timeout: float = 1.0):
        """Wait for a newer frame than last_seq and return (seq, bytes), or None on timeout."""
        with self._cv:
            # Fast path
            if self._frame is not None and self._seq != last_seq:
                return self._seq, self._frame
            # Wait (spurious wakeups are fine; we re-check)
            if not self._cv.wait(timeout=timeout):
                return None
            if self._frame is None or self._seq == last_seq:
                return None
            return self._seq, self._frame

hub = MJPEGHub()
picam2 = Picamera2()

@app.on_event("startup")
def startup():
    cfg = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},  # 720p = more headroom for multiple clients
        buffer_count=2,
    )
    picam2.configure(cfg)
    picam2.set_controls({"AwbEnable": True, "AeEnable": True})
    picam2.start_recording(MJPEGEncoder(), FileOutput(hub))
    time.sleep(0.3)

@app.on_event("shutdown")
def shutdown():
    try:
        picam2.stop_recording()
    except Exception:
        pass

BOUNDARY = b"--frame"

@app.get("/mjpeg")
def mjpeg(request: Request):
    def gen():
        last_seq = -1
        try:
            while True:
                snap = hub.snapshot_new(last_seq, timeout=1.0)
                if snap is None:
                    # no new frame this second; keep connection alive
                    continue

                last_seq, jpeg_out = snap
                yield (
                    BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n"
                    b"Content-Length: " + str(len(jpeg_out)).encode() + b"\r\n\r\n" +
                    jpeg_out + b"\r\n"
                )
        except GeneratorExit:
            # client disconnected; just exit cleanly
            return

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )