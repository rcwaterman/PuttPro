"""
processor.py — YOLO video processing pipeline for PuttPro video analysis.

Reads an uploaded video frame-by-frame, runs YOLOv8 inference on each frame,
annotates only disc-golf-relevant detections, and writes the result to a new
video file. Optionally converts the output to H.264 via ffmpeg for browser
playback. Also analyzes disc flight trajectory to classify putts as made/missed.
"""

import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


# ── Class filtering ───────────────────────────────────────────────────────────
# Labels kept from any model (COCO + future custom model).
# All other detected classes are silently discarded.

_DISC_LABELS = {
    'frisbee',           # COCO class 29 — proxy until custom model
    'disc',
    'disc golf disc',
    'disc_golf_disc',
    'flying disc',
}

_BASKET_LABELS = {
    'disc golf basket',
    'disc_golf_basket',
    'basket',
    'pole hole',
}

_CHAINS_LABELS = {
    'disc golf chains',
    'disc_golf_chains',
    'chains',
}

TRACKED_LABELS = _DISC_LABELS | _BASKET_LABELS | _CHAINS_LABELS


def _is_disc(label: str)   -> bool: return label.lower() in _DISC_LABELS
def _is_basket(label: str) -> bool: return label.lower() in _BASKET_LABELS


def _fmt_mmss(seconds: float | int | None) -> str:
    if seconds is None:
        return ''
    s = max(0, int(round(float(seconds))))
    return f'{s // 60}:{s % 60:02d}'


# ── Model loading ─────────────────────────────────────────────────────────────

_model_cache: dict = {}
_depth_cache: dict = {}
DEPTH_ENABLED = os.environ.get('DEPTH_ENABLED', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
DEPTH_MODEL_NAME = os.environ.get('DEPTH_MODEL', 'depth-anything/Depth-Anything-V2-Small-hf')
DEPTH_FRAME_SKIP = max(1, int(os.environ.get('DEPTH_FRAME_SKIP', '2')))
DISC_DIAMETER_IN = float(os.environ.get('DISC_DIAMETER_IN', '8.0'))
CAMERA_HFOV_DEG = float(os.environ.get('CAMERA_HFOV_DEG', '69.0'))

def _resolve_model_path(model_name: str) -> str:
    """Find the model file: check ../app/ first, then local, then let
    ultralytics auto-download it."""
    candidates = [
        Path(__file__).parent.parent / 'app' / model_name,  # shared with main app
        Path(__file__).parent / model_name,                  # local copy
        Path(model_name),                                     # absolute / CWD
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # ultralytics will download to CWD on first use
    return model_name


def load_model(model_name: str = 'yolov8n.pt', conf: float = 0.4):
    """Load (or return cached) YOLO model with GPU if available."""
    cache_key = (model_name, conf)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from ultralytics import YOLO
    import torch

    path = _resolve_model_path(model_name)
    model = YOLO(path)
    model.conf = conf

    if torch.cuda.is_available():
        model.to(0)
        device_name = torch.cuda.get_device_name(0)
        print(f'[processor] YOLO loaded on CUDA — {device_name}')
    else:
        print('[processor] YOLO loaded on CPU')

    _model_cache[cache_key] = model
    return model


class DepthEstimator:
    """Depth estimator backed strictly by a Hugging Face depth model."""
    def __init__(self, model_name: str = DEPTH_MODEL_NAME):
        self.backend = 'uninitialized'
        self._processor = None
        self._model = None
        self._torch = None
        self._device = 'cpu'

        if not DEPTH_ENABLED:
            raise RuntimeError('Depth pipeline is disabled (DEPTH_ENABLED=0).')

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForDepthEstimation.from_pretrained(model_name)
            model.to(device).eval()

            self._torch = torch
            self._processor = processor
            self._model = model
            self._device = device
            self.backend = f'hf:{model_name}'
            print(f'[processor] Depth model loaded ({self.backend}) on {device}')
        except Exception as e:
            raise RuntimeError(f'Depth model load failed for "{model_name}": {e}') from e

    def _to_closeness_u8(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert raw predicted depth to an 8-bit closeness map:
        brighter = closer, darker = farther.
        """
        depth_f = depth_map.astype(np.float32)
        lo, hi = np.percentile(depth_f, (2.0, 98.0))
        if float(hi - lo) < 1e-6:
            norm_u8 = np.zeros(depth_f.shape, dtype=np.uint8)
        else:
            clipped = np.clip((depth_f - lo) / (hi - lo), 0.0, 1.0)
            norm_u8 = (clipped * 255.0).astype(np.uint8)

        # Heuristic orientation check: lower image region is usually closer.
        h = norm_u8.shape[0]
        top_med = float(np.median(norm_u8[: max(1, h // 4), :]))
        bot_med = float(np.median(norm_u8[max(0, (3 * h) // 4):, :]))
        if bot_med < top_med:
            norm_u8 = 255 - norm_u8

        # Smooth to resemble dense continuous depth shading.
        norm_u8 = cv2.bilateralFilter(norm_u8, d=7, sigmaColor=28, sigmaSpace=22)
        norm_u8 = cv2.GaussianBlur(norm_u8, (0, 0), 2.0)
        return norm_u8

    def _to_grayscale_vis(self, closeness_u8: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(closeness_u8, cv2.COLOR_GRAY2BGR)

    def render(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError('Depth model is not initialized.')

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = self._processor(images=rgb, return_tensors='pt')
            if hasattr(input_batch, 'to'):
                input_batch = input_batch.to(self._device)
            else:
                input_batch = {
                    k: (v.to(self._device) if hasattr(v, 'to') else v)
                    for k, v in input_batch.items()
                }
            with self._torch.no_grad():
                pred = self._model(**input_batch).predicted_depth
                pred = self._torch.nn.functional.interpolate(
                    pred.unsqueeze(1),  # Bx1xHxW
                    size=frame_bgr.shape[:2],
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()
            depth = pred.detach().cpu().numpy()
            closeness_u8 = self._to_closeness_u8(depth)
            return self._to_grayscale_vis(closeness_u8), closeness_u8
        except Exception as e:
            raise RuntimeError(f'Depth inference failed: {e}') from e


def _estimate_disc_distance_ft_from_disc_size(
    bbox: list[float],
    frame_width: int,
) -> float | None:
    """Approximate distance (feet) from apparent disc size (8-inch diameter)."""
    w = max(float(frame_width), 1.0)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    disc_px = max(float(x2 - x1), float(y2 - y1))
    if disc_px <= 1.0:
        return None

    hfov_deg = float(np.clip(CAMERA_HFOV_DEG, 20.0, 140.0))
    focal_px = w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    disc_ft = max(0.1, DISC_DIAMETER_IN / 12.0)
    distance_ft = (disc_ft * focal_px) / disc_px
    distance_ft = float(np.clip(distance_ft, 1.0, 180.0))
    return round(distance_ft, 1)


def load_depth_estimator(model_name: str = DEPTH_MODEL_NAME) -> DepthEstimator:
    cache_key = model_name
    if cache_key in _depth_cache:
        return _depth_cache[cache_key]
    est = DepthEstimator(model_name)
    _depth_cache[cache_key] = est
    return est


def prime_models(
    model_name: str = 'yolov8n.pt',
    conf: float = 0.4,
    imgsz: int = 320,
) -> None:
    """Load YOLO and depth models into cache and run a warm-up forward pass.

    Call once at startup (in a background thread) so the first real job gets
    full GPU throughput without paying model-load or CUDA kernel compilation
    costs at inference time.
    """
    t_total = time.time()

    # ── YOLO ──────────────────────────────────────────────────────────────────
    try:
        t0 = time.time()
        model = load_model(model_name, conf)
        dummy_yolo = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model(dummy_yolo, imgsz=imgsz, verbose=False)
        print(f'[processor] YOLO primed in {time.time() - t0:.1f}s')
    except Exception as e:
        print(f'[processor] YOLO prime failed: {e}')

    # ── Depth ─────────────────────────────────────────────────────────────────
    if DEPTH_ENABLED:
        try:
            t0 = time.time()
            est = load_depth_estimator(DEPTH_MODEL_NAME)
            dummy_depth = np.zeros((64, 64, 3), dtype=np.uint8)
            est.render(dummy_depth)
            print(f'[processor] Depth model primed in {time.time() - t0:.1f}s')
        except Exception as e:
            print(f'[processor] Depth prime failed: {e}')

    print(f'[processor] Models ready in {time.time() - t_total:.1f}s total')


# ── Video processing ──────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    info = {
        'width':       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height':      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps':         cap.get(cv2.CAP_PROP_FPS) or 30.0,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_s':  0.0,
    }
    info['duration_s'] = info['frame_count'] / info['fps']
    cap.release()
    return info


def _try_h264_writer(path: str, fps: float, w: int, h: int):
    """Try to open a VideoWriter with H.264; fall back to mp4v."""
    for fourcc_str, ext in [('avc1', '.mp4'), ('H264', '.mp4'), ('mp4v', '.mp4')]:
        try:
            out_path = str(Path(path).with_suffix(ext))
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if writer.isOpened():
                return writer, out_path
            writer.release()
        except Exception:
            pass
    raise RuntimeError('No suitable video codec found (tried avc1, H264, mp4v)')


def _ffmpeg_to_h264(src: str, dst: str) -> bool:
    """Convert src to H.264 MP4 at dst using ffmpeg. Returns True on success."""
    try:
        subprocess.run(
            [
                'ffmpeg', '-y',
                '-i', src,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '22',
                '-movflags', '+faststart',
                '-an',
                dst,
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def process_video(
    input_path: str,
    output_dir: str,
    job_id: str,
    jobs: dict,
    model_name: str = 'yolov8n.pt',
    conf: float = 0.4,
    imgsz: int = 640,
    frame_skip: int = 1,
    on_complete=None,   # optional callable(job_id) invoked when done or error
) -> None:
    """
    Process a video file with YOLO and write annotated output.

    Updates jobs[job_id] with progress, status, and results.
    Only detections whose label is in TRACKED_LABELS are retained.
    Analyzes disc trajectory and classifies the putt as made/missed.

    Args:
        input_path:  Path to uploaded video.
        output_dir:  Directory to write output files.
        job_id:      Unique job identifier.
        jobs:        Shared dict updated in-place for SSE progress.
        model_name:  YOLO model filename.
        conf:        Detection confidence threshold.
        imgsz:       YOLO inference image size.
        frame_skip:  Process every Nth frame (1 = every frame).
        on_complete: Optional callback invoked when job finishes or errors.
    """
    jobs[job_id].update({
        'status': 'loading_model',
        'progress': 0.0,
        'pipeline_message': 'Loading YOLO detector',
    })

    def _fail(msg: str) -> None:
        jobs[job_id].update({
            'status': 'error',
            'error': msg,
            'pipeline_message': 'Processing failed',
        })
        if on_complete:
            on_complete(job_id)

    try:
        model = load_model(model_name, conf)
    except Exception as e:
        _fail(f'Model load failed: {e}')
        return

    # ── Open source video ────────────────────────────────────────────────────
    jobs[job_id].update({
        'status': 'processing',
        'progress': 0.0,
        'pipeline_message': 'Opening source video',
    })

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        _fail('Cannot open uploaded video')
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    jobs[job_id].update({
        'total_frames': total,
        'fps':          fps,
        'width':        width,
        'height':       height,
    })

    # ── Prepare output writer ────────────────────────────────────────────────
    raw_out   = os.path.join(output_dir, f'{job_id}_raw.mp4')
    final_out = os.path.join(output_dir, f'{job_id}.mp4')

    jobs[job_id].update({'pipeline_message': 'Preparing output writers'})
    try:
        writer, raw_out = _try_h264_writer(raw_out, fps, width, height)
    except Exception as e:
        cap.release()
        _fail(f'Video writer init failed: {e}')
        return
    depth_writer = None
    depth_raw_out = os.path.join(output_dir, f'{job_id}_depth_raw.mp4')
    depth_final_out = os.path.join(output_dir, f'{job_id}_depth.mp4')
    depth_output_path = None
    depth_backend = 'required'
    depth_estimator = None
    depth_skip = DEPTH_FRAME_SKIP
    last_depth_frame = None

    if not DEPTH_ENABLED:
        cap.release()
        writer.release()
        _fail('Depth pipeline is disabled. Set DEPTH_ENABLED=1.')
        return

    jobs[job_id].update({'pipeline_message': 'Loading depth model'})
    try:
        depth_estimator = load_depth_estimator(DEPTH_MODEL_NAME)
        depth_backend = depth_estimator.backend
        depth_writer, depth_raw_out = _try_h264_writer(depth_raw_out, fps, width, height)
    except Exception as e:
        cap.release()
        writer.release()
        if depth_writer is not None:
            depth_writer.release()
        _fail(f'Depth pipeline failed: {e}')
        return

    # ── Per-frame inference ──────────────────────────────────────────────────
    frame_detections: list[dict] = []
    frame_idx = 0
    processed = 0
    t_start   = time.time()
    jobs[job_id].update({'pipeline_message': 'Analyzing frames'})

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = frame_idx / fps
        depth_frame_for_write = None

        if depth_writer is not None and depth_estimator is not None:
            if frame_idx % depth_skip == 0 or last_depth_frame is None:
                try:
                    last_depth_frame, _ = depth_estimator.render(frame)
                except Exception as e:
                    cap.release()
                    writer.release()
                    depth_writer.release()
                    _fail(f'Depth inference failed: {e}')
                    return
            depth_frame_for_write = last_depth_frame.copy()

        if frame_idx % frame_skip == 0:
            try:
                results = model(frame, imgsz=imgsz, verbose=False)
            except Exception as e:
                cap.release()
                writer.release()
                if depth_writer is not None:
                    depth_writer.release()
                _fail(f'Inference failed at frame {frame_idx}: {e}')
                return
            annotated = frame.copy()

            # Collect detections — filter to tracked labels only
            dets = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                if label.lower() not in TRACKED_LABELS:
                    continue
                conf_v = float(box.conf[0])
                xyxy   = box.xyxy[0].tolist()
                dets.append({
                    'class_id':   cls_id,
                    'label':      label,
                    'confidence': round(conf_v, 4),
                    'bbox':       [round(v, 1) for v in xyxy],
                })
                if _is_disc(label):
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    dist_ft = _estimate_disc_distance_ft_from_disc_size(dets[-1]['bbox'], width)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 170, 255), 2)
                    dist_txt = f'  est {dist_ft:.1f} ft' if dist_ft is not None else ''
                    cv2.putText(
                        annotated,
                        f'{label} {conf_v:.2f}{dist_txt}',
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 170, 255),
                        2,
                        cv2.LINE_AA,
                    )

            frame_detections.append({
                'frame_idx':   frame_idx,
                'timestamp_s': round(timestamp_s, 4),
                'detections':  dets,
            })

            writer.write(annotated)
            processed += 1
        else:
            writer.write(frame)

        if depth_writer is not None and depth_frame_for_write is not None:
            depth_writer.write(depth_frame_for_write)

        frame_idx += 1

        if frame_idx % 10 == 0:
            progress = frame_idx / max(total, 1)
            elapsed  = time.time() - t_start
            fps_proc = processed / max(elapsed, 0.001)
            jobs[job_id].update({
                'progress':      round(progress, 4),
                'frames_done':   frame_idx,
                'inference_fps': round(fps_proc, 1),
            })

    cap.release()
    writer.release()
    if depth_writer is not None:
        depth_writer.release()

    jobs[job_id].update({
        'status': 'encoding',
        'progress': 1.0,
        'pipeline_message': 'Encoding output videos',
    })

    # ── Convert to browser-compatible H.264 via ffmpeg ───────────────────────
    try:
        converted = _ffmpeg_to_h264(raw_out, final_out)
        if converted:
            os.remove(raw_out)
            output_path = final_out
        else:
            os.rename(raw_out, final_out)
            output_path = final_out

        if depth_writer is not None and os.path.exists(depth_raw_out):
            depth_converted = _ffmpeg_to_h264(depth_raw_out, depth_final_out)
            if depth_converted:
                os.remove(depth_raw_out)
                depth_output_path = depth_final_out
            else:
                os.rename(depth_raw_out, depth_final_out)
                depth_output_path = depth_final_out
    except Exception as e:
        _fail(f'Encoding failed: {e}')
        return

    # ── Analyze trajectory & classify putt ───────────────────────────────────
    flight    = _analyze_trajectory(frame_detections, fps, width, height)
    summary   = _build_summary(frame_detections, flight)

    if depth_estimator is not None:
        depth_backend = depth_estimator.backend

    elapsed = time.time() - t_start
    jobs[job_id].update({
        'status':       'done',
        'progress':     1.0,
        'pipeline_message': 'Completed',
        'output_path':  output_path,
        'depth_output_path': depth_output_path,
        'has_depth':    bool(depth_output_path),
        'depth_backend': depth_backend,
        'elapsed_s':    round(elapsed, 2),
        'summary':      summary,
        'flight':       flight,
        'putt_result':  flight.get('putt_result', 'unknown'),
    })
    if on_complete:
        on_complete(job_id)


# ── Trajectory analysis ───────────────────────────────────────────────────────

def _analyze_trajectory(
    frame_detections: list[dict],
    fps: float,
    width: int,
    height: int,
) -> dict:
    """
    Analyze disc flight from per-frame detections.

    Returns a dict with:
        disc_detected    bool
        disc_frames      int     - number of frames the disc was seen
        flight_duration_s float
        speed_px_s       float   - avg speed in pixels/second
        angle_deg        float   - flight direction angle (0=right, 90=up)
        putt_result      str     - 'made' | 'missed' | 'unknown'
        basket_detected  bool
        trajectory       list    - downsampled [{cx, cy, t}]
    """
    disc_track      = []
    basket_sightings = []

    for fd in frame_detections:
        for det in fd['detections']:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if _is_disc(det['label']):
                disc_track.append({
                    'frame': fd['frame_idx'],
                    'cx': cx, 'cy': cy,
                    't': fd['timestamp_s'],
                })
            elif _is_basket(det['label']):
                basket_sightings.append({'cx': cx, 'cy': cy})

    if not disc_track:
        return {
            'disc_detected':   False,
            'disc_frames':     0,
            'basket_detected': bool(basket_sightings),
            'putt_result':     'unknown',
            'putt_events':     [],
            'trajectory':      [],
        }

    segments = []
    cur = [disc_track[0]]
    for p in disc_track[1:]:
        if p['t'] - cur[-1]['t'] > 0.65:
            segments.append(cur)
            cur = [p]
        else:
            cur.append(p)
    if cur:
        segments.append(cur)

    putt_events = []
    for seg in segments:
        if len(seg) < 4:
            continue

        first = seg[0]
        last  = seg[-1]
        dt = max(last['t'] - first['t'], 1 / fps)
        dx = last['cx'] - first['cx']
        dy = last['cy'] - first['cy']
        dist_px = math.hypot(dx, dy)
        if dist_px < max(width, height) * 0.07:
            continue

        speed_px_s = dist_px / dt
        angle_deg = math.degrees(math.atan2(-dy, dx))

        ys = [p['cy'] for p in seg]
        peak_idx = min(range(len(seg)), key=lambda i: ys[i])
        pre_peak = seg[:peak_idx + 1]
        post_peak = seg[peak_idx:]
        rise_px = pre_peak[0]['cy'] - pre_peak[-1]['cy']
        drop_px = post_peak[-1]['cy'] - post_peak[0]['cy']
        has_arc = rise_px > height * 0.08 and drop_px > height * 0.06 and 1 < peak_idx < len(seg) - 2

        peak_x_frac = seg[peak_idx]['cx'] / width
        peak_y_frac = seg[peak_idx]['cy'] / height
        end_x_frac = last['cx'] / width
        end_y_frac = last['cy'] / height
        center_lane = 0.22 <= peak_x_frac <= 0.78 or 0.22 <= end_x_frac <= 0.78

        if has_arc and peak_y_frac < 0.50 and center_lane and end_y_frac < 0.85:
            result = 'made'
        elif rise_px > height * 0.08:
            result = 'missed'
        else:
            result = 'unknown'

        putt_events.append({
            'timestamp_s': round(first['t'], 2),
            'timestamp_label': _fmt_mmss(first['t']),
            'result': result,
            'angle_deg': round(angle_deg, 1),
            'speed_px_s': round(speed_px_s, 1),
            'disc_frames': len(seg),
            'duration_s': round(dt, 3),
            'peak_y_frac': round(peak_y_frac, 3),
            'end_x_frac': round(end_x_frac, 3),
            'end_y_frac': round(end_y_frac, 3),
        })

    classified = [p for p in putt_events if p['result'] in ('made', 'missed')]
    if classified:
        made = sum(1 for p in classified if p['result'] == 'made')
        missed = len(classified) - made
        putt_result = 'made' if made >= missed else 'missed'
    else:
        putt_result = 'unknown'

    step = max(1, len(disc_track) // 60)
    trajectory = [
        {'cx': round(p['cx'], 1), 'cy': round(p['cy'], 1), 't': p['t']}
        for p in disc_track[::step]
    ]

    result = {
        'disc_detected':    True,
        'disc_frames':      len(disc_track),
        'flight_duration_s': round(max(disc_track[-1]['t'] - disc_track[0]['t'], 1 / fps), 3),
        'speed_px_s':        putt_events[-1]['speed_px_s'] if putt_events else 0.0,
        'angle_deg':         putt_events[-1]['angle_deg'] if putt_events else 0.0,
        'final_x_frac':      round(disc_track[-1]['cx'] / width, 3),
        'final_y_frac':      round(disc_track[-1]['cy'] / height, 3),
        'basket_detected':   bool(basket_sightings),
        'putt_result':       putt_result,
        'putt_events':       putt_events,
        'trajectory':        trajectory,
    }
    return result

def process_chunk(
    chunk_path: str,
    session_id: str,
    sessions: dict,
    chunk_idx: int,
    model_name: str = 'yolov8n.pt',
    conf: float = 0.4,
    imgsz: int = 320,
    frame_skip: int = 3,
) -> dict:
    """
    Process a single streaming video chunk with YOLO.

    Designed for real-time use: smaller imgsz and higher frame_skip
    than the full upload pipeline. Updates sessions[session_id] in-place
    and returns a compact event dict suitable for SSE.
    """
    try:
        model = load_model(model_name, conf)
    except Exception as e:
        return {'type': 'chunk_error', 'chunk_idx': chunk_idx, 'error': str(e)}

    cap = cv2.VideoCapture(chunk_path)
    if not cap.isOpened():
        return {'type': 'chunk_error', 'chunk_idx': chunk_idx, 'error': 'Cannot open chunk'}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    chunk_detections: list[dict] = []
    frame_idx = 0
    disc_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, imgsz=imgsz, verbose=False)
            dets = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                if label.lower() not in TRACKED_LABELS:
                    continue
                conf_v = float(box.conf[0])
                xyxy   = box.xyxy[0].tolist()
                dets.append({
                    'class_id':   cls_id,
                    'label':      label,
                    'confidence': round(conf_v, 4),
                    'bbox':       [round(v, 1) for v in xyxy],
                })
                if _is_disc(label):
                    disc_frames += 1

            chunk_detections.append({
                'frame_idx':   frame_idx,
                'timestamp_s': round(frame_idx / fps, 4),
                'detections':  dets,
            })

        frame_idx += 1

    cap.release()

    # Trajectory analysis for this chunk
    flight = _analyze_trajectory(chunk_detections, fps, width, height)
    putt_result = flight.get('putt_result', 'unknown')
    chunk_events = flight.get('putt_events', [])

    # Accumulate into session
    session = sessions.get(session_id)
    if session is not None:
        session['frames_processed'] = session.get('frames_processed', 0) + frame_idx

        if chunk_events:
            for ev in chunk_events:
                if ev.get('result') not in ('made', 'missed'):
                    continue
                session.setdefault('putts', []).append({
                    'chunk_idx':     chunk_idx,
                    'result':        ev.get('result'),
                    'angle_deg':     ev.get('angle_deg'),
                    'speed_px_s':    ev.get('speed_px_s'),
                    'disc_frames':   ev.get('disc_frames', 0),
                    'timestamp_s':   ev.get('timestamp_s', 0.0),
                    'timestamp_label': ev.get('timestamp_label', ''),
                })
        elif putt_result in ('made', 'missed'):
            session.setdefault('putts', []).append({
                'chunk_idx':   chunk_idx,
                'result':      putt_result,
                'angle_deg':   flight.get('angle_deg'),
                'speed_px_s':  flight.get('speed_px_s'),
                'disc_frames': flight.get('disc_frames', 0),
            })

        session['chunks_received'] = chunk_idx + 1
        putts = session.get('putts', [])
        made  = sum(1 for p in putts if p['result'] == 'made')

        session['live_summary'] = {
            'total_putts': len(putts),
            'made':        made,
            'missed':      len(putts) - made,
            'make_pct':    round(made / len(putts) * 100, 1) if putts else 0,
        }

    return {
        'type':        'chunk_done',
        'chunk_idx':   chunk_idx,
        'frames':      frame_idx,
        'disc_frames': disc_frames,
        'putt_result': putt_result,
        'angle_deg':   flight.get('angle_deg'),
        'speed_px_s':  flight.get('speed_px_s'),
        'live_summary': session.get('live_summary') if session else None,
    }


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(frame_detections: list[dict], flight: dict) -> dict:
    """Aggregate per-frame detections into a top-level summary."""
    label_counts: dict[str, int] = {}
    label_conf:   dict[str, list[float]] = {}
    total_detections = 0

    for fd in frame_detections:
        for det in fd['detections']:
            lbl = det['label']
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            label_conf.setdefault(lbl, []).append(det['confidence'])
            total_detections += 1

    labels_summary = {}
    for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        confs = label_conf[lbl]
        labels_summary[lbl] = {
            'count':    count,
            'avg_conf': round(sum(confs) / len(confs), 3),
            'max_conf': round(max(confs), 3),
        }

    return {
        'total_detections': total_detections,
        'analyzed_frames':  len(frame_detections),
        'labels':           labels_summary,
        'putt_result':      flight.get('putt_result', 'unknown'),
        'putt_count':       len(flight.get('putt_events', [])),
        'putt_events':      flight.get('putt_events', []),
        'disc_detected':    flight.get('disc_detected', False),
        'basket_detected':  flight.get('basket_detected', False),
    }
