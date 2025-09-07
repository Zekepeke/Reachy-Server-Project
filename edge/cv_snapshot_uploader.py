import io, time, yaml, requests, subprocess
from PIL import Image

with open("edge/config.yaml","r") as f:
    CFG = yaml.safe_load(f)
RTSP_URL = CFG["cloud"]["rtsp_url"]

def capture_jpeg_from_camera():
    # fast single frame using libcamera-still
    # requires: sudo apt install libcamera-apps
    out = subprocess.check_output([
        "libcamera-still","-n","-e","jpg","-o","-","--width","640","--height","480","--quality","80"
    ])
    return out

def upload(img_bytes):
    # replace with your gateway’s snapshot endpoint if you expose one
    url = f"http://{CFG['cloud']['host']}:7000/snapshot"  # example
    try:
        requests.post(url, files={"image": ("snap.jpg", img_bytes, "image/jpeg")}, timeout=3)
    except Exception as e:
        print("[snapshot] upload failed:", e)

if __name__ == "__main__":
    jpg = capture_jpeg_from_camera()
    upload(jpg)
    print("[snapshot] sent")