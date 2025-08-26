# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from threading import Condition
import io, time

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import libcamera  # for Transform if you want flips/rotates

from server.pipeline import Pipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

pipe = Pipeline(model_path="hand_landmarker.task")

class MJPEGBuffer(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.cv = Condition()
    def write(self, buf):
        with self.cv:
            self.frame = buf
            self.cv.notify_all()

picam2 = Picamera2()
output = MJPEGBuffer()

@app.on_event("startup")
def startup():
    cfg = picam2.create_video_configuration(
        main={"size": (1920, 1080), "format": "RGB888"},
        buffer_count=2,
    )
    picam2.configure(cfg)
    picam2.set_controls({"AwbEnable": True, "AeEnable": True})
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))
    time.sleep(0.5)  # let AE/AWB settle

@app.on_event("shutdown")
def shutdown():
    try:
        picam2.stop_recording()
    except Exception:
        pass

@app.get("/mjpeg")
def mjpeg():
    boundary = b"--frame"
    def gen():
        while True:
            with output.cv:
                output.cv.wait()
                frame = output.frame
                frame = pipe(frame)
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                   frame + b"\r\n")
    return StreamingResponse(gen(),
        media_type="multipart/x-mixed-replace; boundary=frame")