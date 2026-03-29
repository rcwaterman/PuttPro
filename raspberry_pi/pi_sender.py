import time

import cv2
import requests

# ---------------------------------------------------------------------------
# Configuration — edit these before running on the Pi
# ---------------------------------------------------------------------------
SERVER_URL    = 'https://<PC_IP>:5000/upload_frame'  # Use HTTPS
UPLOAD_TOKEN  = '<token>'           # Copy from PC startup output or .upload_token
CAPTURE_SOURCE = 0                  # 0 = first camera; GStreamer string for libcamera
TARGET_FPS    = 20
JPEG_QUALITY  = 80                  # 0-100; lower = smaller payload
# ---------------------------------------------------------------------------

# The Pi will see a self-signed cert warning; disable verification for the
# local network. For production, install the cert on the Pi instead.
VERIFY_SSL = False
if not VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    'Content-Type': 'image/jpeg',
    'X-Upload-Token': UPLOAD_TOKEN,
}

cap = cv2.VideoCapture(CAPTURE_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f'Could not open capture source: {CAPTURE_SOURCE}')

frame_interval = 1.0 / TARGET_FPS
session = requests.Session()
print(f'Sending to {SERVER_URL} at {TARGET_FPS} FPS')

while True:
    t0 = time.monotonic()

    success, frame = cap.read()
    if not success:
        print('Camera read failed, retrying...')
        time.sleep(0.5)
        continue

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    try:
        session.post(
            SERVER_URL,
            data=buffer.tobytes(),
            headers=HEADERS,
            timeout=2,
            verify=VERIFY_SSL,
        )
    except requests.exceptions.RequestException as e:
        print(f'Send error: {e}')

    elapsed = time.monotonic() - t0
    wait = frame_interval - elapsed
    if wait > 0:
        time.sleep(wait)
