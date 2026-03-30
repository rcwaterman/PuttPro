import queue
import threading

import cv2
import numpy as np

# Outgoing MJPEG stream encode quality (0-100).
_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 85]

_DEFAULT_IMGSZ = 640
_DEFAULT_CONF  = 0.4


def _load_yolo(model_path: str, conf: float, imgsz: int):
    """
    Load a YOLO model on the best available device.
    - CUDA GPU: full float16 (half-precision) for maximum throughput on RTX cards.
    - CPU fallback: float32.
    Warm-up inference eliminates the first-frame latency spike.
    """
    from ultralytics import YOLO
    import torch

    model = YOLO(model_path)

    if torch.cuda.is_available():
        model.to(0)
        #model.model.half()   # FP16 — halves VRAM usage, ~2x throughput on RTX
        dtype  = torch.float16
        device = f'CUDA — {torch.cuda.get_device_name(0)}'
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.float16)
    else:
        dtype  = torch.float32
        device = 'CPU'
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.float32)

    print(f'YOLO "{model_path}" loaded on {device}.')
    model(dummy, imgsz=imgsz, conf=conf, verbose=False)
    print(f'YOLO warm-up complete.')

    return model


class _BaseCamera:
    def __init__(
        self,
        yolo_model: str | None = None,
        yolo_conf: float = _DEFAULT_CONF,
        yolo_imgsz: int = _DEFAULT_IMGSZ,
    ):
        self._current_frame: bytes | None = None
        self._frame_lock = threading.Lock()
        self.frame_event = threading.Event()

        self._yolo       = _load_yolo(yolo_model, yolo_conf, yolo_imgsz) if yolo_model else None
        self._yolo_conf  = yolo_conf
        self._yolo_imgsz = yolo_imgsz

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        if self._yolo is None:
            return frame
        results = self._yolo(
            frame,
            imgsz=self._yolo_imgsz,
            conf=self._yolo_conf,
            verbose=False,
        )
        return results[0].plot()

    def _store(self, frame: np.ndarray):
        _, buf = cv2.imencode('.jpg', frame, _ENCODE_PARAMS)
        with self._frame_lock:
            self._current_frame = buf.tobytes()
        self.frame_event.set()

    def get_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._current_frame


class LocalCamera(_BaseCamera):
    def __init__(
        self,
        source: int = 0,
        yolo_model: str | None = None,
        yolo_conf: float = _DEFAULT_CONF,
        yolo_imgsz: int = _DEFAULT_IMGSZ,
    ):
        super().__init__(yolo_model, yolo_conf, yolo_imgsz)
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def _capture_loop(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                self._store(self._annotate(frame))


class RemoteCamera(_BaseCamera):
    """
    Receives JPEG frames pushed from a remote device (Pi, phone, native app).
    receive_frame() enqueues raw bytes; a background thread decodes and runs
    YOLO.  Queue is bounded (maxsize=4) — frames drop rather than building lag.
    """

    def __init__(
        self,
        yolo_model: str | None = None,
        yolo_conf: float = _DEFAULT_CONF,
        yolo_imgsz: int = _DEFAULT_IMGSZ,
    ):
        super().__init__(yolo_model, yolo_conf, yolo_imgsz)
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=4)
        threading.Thread(target=self._process_loop, daemon=True).start()

    def receive_frame(self, jpeg_bytes: bytes):
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(jpeg_bytes)

    def _process_loop(self):
        while True:
            raw = self._queue.get()
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._store(self._annotate(frame))


def generate_frames(camera: _BaseCamera):
    while True:
        camera.frame_event.wait(timeout=1.0)
        camera.frame_event.clear()
        frame = camera.get_frame()
        if frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
