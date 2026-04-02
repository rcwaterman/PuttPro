"""
app.py — PuttPro Video Analysis Service

Routes:
    GET  /                       Desktop upload page
    GET  /mobile                 Mobile PWA (record + upload + history)
    POST /upload                 Accept video, enqueue processing job
    GET  /status/<job_id>        SSE stream: real-time progress events
    GET  /result/<job_id>        Stream annotated video for in-browser playback
    GET  /result_depth/<job_id>  Stream depth-map video for in-browser playback
    GET  /clip/<job_id>          Stream ±N second clip around a putt timestamp
    GET  /download/<job_id>      Download annotated video as attachment
    GET  /history                All completed jobs (for mobile history screen)
    GET  /jobs                   All jobs including in-progress (debug)
    GET  /healthz                Health endpoint for deployment platforms

Run:
    python app.py
"""

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, stream_with_context
from werkzeug.middleware.proxy_fix import ProxyFix

from processor import (
    ALLOWED_EXTENSIONS,
    CAMERA_HFOV_DEG,
    DEPTH_ENABLED,
    DEPTH_FRAME_SKIP,
    DEPTH_MODEL_NAME,
    DISC_DIAMETER_IN,
    process_video,
    process_chunk,
)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB  = 500
YOLO_MODEL     = os.environ.get('YOLO_MODEL',  'yolov8n.pt')
YOLO_CONF      = float(os.environ.get('YOLO_CONF',  '0.35'))
YOLO_IMGSZ     = int(os.environ.get('YOLO_IMGSZ', '640'))
FRAME_SKIP     = int(os.environ.get('FRAME_SKIP',  '1'))
PORT           = int(os.environ.get('PORT', '5050'))

DB_PATH           = OUTPUT_DIR / 'jobs_db.json'
STATS_ARCHIVE_PATH = OUTPUT_DIR / 'stats_archive.json'

# ── App ───────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()

# ── Persistence ───────────────────────────────────────────────────────────────

def _load_jobs_db() -> None:
    """Load persisted jobs from disk on startup. Skips jobs whose output
    files no longer exist on disk (cleaned up manually)."""
    if not DB_PATH.exists():
        return
    try:
        with open(DB_PATH) as f:
            saved = json.load(f)
        with jobs_lock:
            for job in saved:
                jid = job.get('job_id')
                if not jid:
                    continue
                # Legacy cleanup: detection JSON artifacts are no longer part
                # of the product surface.
                job.pop('json_path', None)
                # Mark any previously-running job as failed (server was restarted)
                if job.get('status') not in ('done', 'error'):
                    job['status'] = 'error'
                    job['error']  = 'Server restarted during processing'
                # Drop if output file is gone
                if job.get('status') == 'done':
                    out = job.get('output_path', '')
                    if out and not os.path.exists(out):
                        continue
                    depth_out = job.get('depth_output_path', '')
                    if depth_out and os.path.exists(depth_out):
                        job['has_depth'] = True
                    else:
                        job['has_depth'] = False
                        job.pop('depth_output_path', None)
                jobs[jid] = job
    except Exception as e:
        print(f'[app] Warning: could not load jobs_db: {e}')


def _save_jobs_db() -> None:
    """Persist all jobs to disk (called when a job completes or errors)."""
    try:
        with jobs_lock:
            snapshot = list(jobs.values())
        with open(DB_PATH, 'w') as f:
            json.dump(snapshot, f, indent=2)
    except Exception as e:
        print(f'[app] Warning: could not save jobs_db: {e}')


def _on_job_complete(job_id: str) -> None:
    """Callback invoked by the processor thread when a job finishes."""
    _save_jobs_db()


# ── Stats archive (survives job deletion) ─────────────────────────────────────

def _job_to_series_entries(job: dict) -> list[dict]:
    """Convert a completed job into the list of series entries used by /stats."""
    entries = []
    created_at = job.get('created_at', 0)
    events = job.get('flight', {}).get('putt_events') or []
    if events:
        for idx, ev in enumerate(events, start=1):
            t_s = ev.get('timestamp_s', 0.0) or 0.0
            entries.append({
                'job_id':             job['job_id'],
                'filename':           job.get('filename', ''),
                'created_at':         created_at + t_s,
                'session_created_at': created_at,
                'putt_result':        ev.get('result', 'unknown'),
                'angle_deg':          ev.get('angle_deg'),
                'speed_px_s':         ev.get('speed_px_s'),
                'disc_frames':        ev.get('disc_frames', 0),
                'timestamp_s':        round(t_s, 2),
                'timestamp_label':    ev.get('timestamp_label', ''),
                'event_idx':          idx,
            })
    else:
        entries.append({
            'job_id':             job['job_id'],
            'filename':           job.get('filename', ''),
            'created_at':         created_at,
            'session_created_at': created_at,
            'putt_result':        job.get('putt_result', 'unknown'),
            'angle_deg':          job.get('flight', {}).get('angle_deg'),
            'speed_px_s':         job.get('flight', {}).get('speed_px_s'),
            'disc_frames':        job.get('flight', {}).get('disc_frames', 0),
            'timestamp_s':        None,
            'timestamp_label':    '',
            'event_idx':          1,
        })
    return entries


def _load_stats_archive() -> list[dict]:
    if not STATS_ARCHIVE_PATH.exists():
        return []
    try:
        with open(STATS_ARCHIVE_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _append_to_stats_archive(entries: list[dict]) -> None:
    archive = _load_stats_archive()
    archive.extend(entries)
    try:
        with open(STATS_ARCHIVE_PATH, 'w') as f:
            json.dump(archive, f)
    except Exception as e:
        print(f'[app] Warning: could not update stats_archive: {e}')


def _session_chunk_paths(session_id: str) -> list[Path]:
    return sorted(UPLOAD_DIR.glob(f'{session_id}_chunk_*'))


def _merge_session_chunks(chunk_paths: list[Path], out_path: Path) -> bool:
    """Decode chunk files and stitch them into one MP4 for full analysis."""
    writer = None
    target_wh = None
    target_fps = 30.0
    frames_written = 0

    try:
        for chunk in chunk_paths:
            cap = cv2.VideoCapture(str(chunk))
            if not cap.isOpened():
                continue

            if writer is None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if w <= 0 or h <= 0:
                    cap.release()
                    continue
                target_wh = (w, h)
                target_fps = max(fps, 1.0)
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    target_fps,
                    target_wh,
                )
                if not writer.isOpened():
                    cap.release()
                    return False

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if target_wh and (frame.shape[1], frame.shape[0]) != target_wh:
                    frame = cv2.resize(frame, target_wh)
                writer.write(frame)
                frames_written += 1

            cap.release()
    finally:
        if writer is not None:
            writer.release()

    return frames_written > 0 and out_path.exists()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _job_public(job: dict) -> dict:
    """Strip internal filesystem paths before sending to the client."""
    return {
        k: v for k, v in job.items()
        if k not in ('output_path', 'depth_output_path')
    }


# ── Routes ────────────────────────────────────────────────────────────────────

_load_jobs_db()


@app.route('/')
def index():
    ua = (request.headers.get('User-Agent') or '').lower()
    is_mobile = any(k in ua for k in ('iphone', 'ipad', 'android', 'mobile'))
    if is_mobile or request.args.get('mobile') == '1':
        return redirect('/mobile', code=302)
    return render_template('index.html')


@app.route('/mobile')
def mobile():
    # Ensure testers always receive the latest mobile frontend (avoid stale PWA cache).
    resp = Response(render_template('mobile.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video field in request'}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400
    if not _allowed(file.filename):
        exts = ', '.join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({'error': f'Unsupported file type. Allowed: {exts}'}), 400

    job_id  = uuid.uuid4().hex
    suffix  = Path(file.filename).suffix.lower() or '.mp4'
    in_path = str(UPLOAD_DIR / f'{job_id}{suffix}')
    file.save(in_path)

    conf_override  = request.form.get('conf',       type=float, default=YOLO_CONF)
    imgsz_override = request.form.get('imgsz',      type=int,   default=YOLO_IMGSZ)
    skip_override  = request.form.get('frame_skip', type=int,   default=FRAME_SKIP)
    model_override = request.form.get('model',      default=YOLO_MODEL)

    with jobs_lock:
        jobs[job_id] = {
            'job_id':     job_id,
            'status':     'queued',
            'progress':   0.0,
            'pipeline_message': 'Queued for processing',
            'filename':   file.filename,
            'created_at': time.time(),
        }

    thread = threading.Thread(
        target=process_video,
        args=(in_path, str(OUTPUT_DIR), job_id, jobs),
        kwargs={
            'model_name':  model_override,
            'conf':        conf_override,
            'imgsz':       imgsz_override,
            'frame_skip':  skip_override,
            'on_complete': _on_job_complete,
        },
        daemon=True,
    )
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status_sse(job_id: str):
    """SSE stream — client receives updates until job is done or errors."""
    def generate():
        last_sent     = None
        timeout_at    = time.time() + 3600

        while time.time() < timeout_at:
            with jobs_lock:
                job = jobs.get(job_id)

            if job is None:
                yield f'data: {json.dumps({"error": "Job not found"})}\n\n'
                return

            pub = _job_public(job)
            sig = (
                pub.get('status'),
                pub.get('progress', 0),
                pub.get('pipeline_message', ''),
                pub.get('frames_done', 0),
                pub.get('inference_fps', 0),
            )

            if sig != last_sent:
                last_sent = sig
                yield f'data: {json.dumps(pub)}\n\n'

            if pub['status'] in ('done', 'error'):
                return

            time.sleep(0.4)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/result/<job_id>')
def result_video(job_id: str):
    """Stream annotated MP4 — supports HTTP Range for mobile scrubbing."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or job.get('status') != 'done':
        return jsonify({'error': 'Result not ready'}), 404

    path = job.get('output_path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Output file missing'}), 404

    return send_file(path, mimetype='video/mp4', conditional=True)


@app.route('/result_depth/<job_id>')
def result_depth_video(job_id: str):
    """Stream generated depth-map MP4 for a completed job."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or job.get('status') != 'done':
        return jsonify({'error': 'Result not ready'}), 404

    path = job.get('depth_output_path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Depth output unavailable'}), 404

    return send_file(path, mimetype='video/mp4', conditional=True)


@app.route('/clip/<job_id>')
def clip_video(job_id: str):
    """Return a short clip around a putt timestamp.
    Query params:
      t   float seconds (required)
      pad float seconds of padding before/after (default 2)
    """
    t = request.args.get('t', type=float)
    if t is None:
        return jsonify({'error': 'Missing query param: t'}), 400
    pad = max(0.1, request.args.get('pad', type=float, default=2.0))

    with jobs_lock:
        job = jobs.get(job_id)

    if not job or job.get('status') != 'done':
        return jsonify({'error': 'Result not ready'}), 404

    src = job.get('output_path')
    if not src or not os.path.exists(src):
        return jsonify({'error': 'Output file missing'}), 404

    fps = float(job.get('fps') or 0.0)
    total = int(job.get('total_frames') or 0)
    duration = (total / fps) if fps > 0 and total > 0 else None

    start_s = max(0.0, t - pad)
    end_s = t + pad
    if duration is not None:
        end_s = min(end_s, duration)
    if end_s <= start_s + 0.15:
        end_s = start_s + 0.5

    clips_dir = OUTPUT_DIR / 'clips'
    clips_dir.mkdir(exist_ok=True)
    clip_name = f'{job_id}_{int(start_s * 1000)}_{int(end_s * 1000)}.mp4'
    clip_path = clips_dir / clip_name

    if not clip_path.exists():
        try:
            subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-ss', f'{start_s:.3f}',
                    '-to', f'{end_s:.3f}',
                    '-i', src,
                    '-an',
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',
                    '-movflags', '+faststart',
                    str(clip_path),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback: return full video if ffmpeg is unavailable.
            return send_file(src, mimetype='video/mp4', conditional=True)

    return send_file(str(clip_path), mimetype='video/mp4', conditional=True)


@app.route('/download/<job_id>')
def download_video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or job.get('status') != 'done':
        return jsonify({'error': 'Result not ready'}), 404

    path = job.get('output_path')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Output file missing'}), 404

    stem = Path(job.get('filename', 'video')).stem
    return send_file(path, as_attachment=True, download_name=f'{stem}_analyzed.mp4')


@app.route('/history/<job_id>', methods=['DELETE'])
def delete_job(job_id: str):
    """Delete a completed job: remove video files and history entry.
    Aggregated putting stats are preserved in the stats archive."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.get('status') not in ('done', 'error'):
        return jsonify({'error': 'Cannot delete a job that is still running'}), 409

    # Preserve stats before deleting
    if job.get('status') == 'done':
        _append_to_stats_archive(_job_to_series_entries(job))

    # Delete video / output files
    for key in ('output_path', 'depth_output_path'):
        path = job.get(key)
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    # Delete uploaded input (regular upload or session-assembled input)
    for pattern in (f'{job_id}.*', f'{job_id}_session_input.mp4'):
        for p in UPLOAD_DIR.glob(pattern):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    # Delete any cached clips
    clips_dir = OUTPUT_DIR / 'clips'
    if clips_dir.exists():
        for p in clips_dir.glob(f'{job_id}_*.mp4'):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    # Remove from in-memory jobs and persist
    with jobs_lock:
        jobs.pop(job_id, None)
    _save_jobs_db()

    return jsonify({'status': 'deleted', 'job_id': job_id})


@app.route('/history')
def history():
    """Return all completed jobs, newest first. Used by mobile history screen."""
    with jobs_lock:
        all_jobs = list(jobs.values())

    visible = [
        _job_public(j) for j in all_jobs
        if j.get('status') in ('done', 'error', 'processing', 'queued', 'encoding', 'loading_model')
    ]
    visible.sort(key=lambda j: j.get('created_at', 0), reverse=True)
    return jsonify(visible)


@app.route('/jobs')
def list_jobs():
    with jobs_lock:
        return jsonify([_job_public(j) for j in jobs.values()])


@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok'})


@app.route('/stats')
def putt_stats():
    """Aggregate putt analytics across all completed jobs and the stats archive."""
    with jobs_lock:
        done_jobs = [j for j in jobs.values() if j.get('status') == 'done']

    # Live jobs
    series = []
    for j in done_jobs:
        series.extend(_job_to_series_entries(j))

    # Archived (deleted) jobs — merge and deduplicate by job_id+event_idx
    live_keys = {(e['job_id'], e.get('event_idx', 1)) for e in series}
    for entry in _load_stats_archive():
        key = (entry.get('job_id'), entry.get('event_idx', 1))
        if key not in live_keys:
            series.append(entry)

    series.sort(key=lambda x: x['created_at'])

    total  = len(series)
    made   = sum(1 for p in series if p.get('putt_result') == 'made')
    missed = sum(1 for p in series if p.get('putt_result') == 'missed')

    # Rolling made% (window = 10 putts)
    rolling = []
    for i, pt in enumerate(series):
        window = series[max(0, i - 9): i + 1]
        w_made = sum(1 for p in window if p['putt_result'] == 'made')
        rolling.append(round(w_made / len(window) * 100, 1))

    return jsonify({
        'total_putts':  total,
        'made':         made,
        'missed':       missed,
        'unknown':      total - made - missed,
        'make_pct':     round(made / total * 100, 1) if total else 0,
        'series':       series,
        'rolling_pct':  rolling,
    })


# ── Streaming routes ──────────────────────────────────────────────────────────

@app.route('/stream/start', methods=['POST'])
def stream_start():
    """Begin a live streaming session. Returns session_id."""
    data      = request.get_json(silent=True) or {}
    conf      = float(data.get('conf',  YOLO_CONF))
    imgsz     = int(data.get('imgsz',  320))       # lower default for real-time
    model_name = data.get('model',     YOLO_MODEL)

    session_id = uuid.uuid4().hex
    with sessions_lock:
        sessions[session_id] = {
            'session_id':      session_id,
            'status':          'streaming',
            'created_at':      time.time(),
            'conf':            conf,
            'imgsz':           imgsz,
            'model':           model_name,
            'chunks_received': 0,
            'frames_processed': 0,
            'putts':           [],
            'event_queue':     [],
        }
    return jsonify({'session_id': session_id})


@app.route('/stream/chunk/<session_id>', methods=['POST'])
def stream_chunk(session_id: str):
    """Receive a video chunk blob and queue it for GPU processing."""
    with sessions_lock:
        session = sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if 'chunk' not in request.files:
        return jsonify({'error': 'No chunk field'}), 400

    chunk_idx = request.form.get('chunk_idx', type=int, default=0)
    mime      = request.form.get('mime', 'video/webm')
    ext       = '.mp4' if 'mp4' in mime else '.webm'
    chunk_path = str(UPLOAD_DIR / f'{session_id}_chunk_{chunk_idx:04d}{ext}')
    request.files['chunk'].save(chunk_path)

    # Process in background thread; push SSE event when done
    conf       = session['conf']
    imgsz      = session['imgsz']
    model_name = session['model']

    def _process():
        result = process_chunk(
            chunk_path, session_id, sessions, chunk_idx,
            model_name=model_name, conf=conf, imgsz=imgsz,
        )
        with sessions_lock:
            if session_id in sessions:
                sessions[session_id]['event_queue'].append(result)

    threading.Thread(target=_process, daemon=True).start()
    return jsonify({'status': 'queued', 'chunk_idx': chunk_idx}), 202


@app.route('/stream/status/<session_id>')
def stream_status(session_id: str):
    """SSE stream: emits an event for each processed chunk until session ends."""
    def generate():
        sent   = 0
        end_at = time.time() + 7200  # 2-hour hard cap

        while time.time() < end_at:
            with sessions_lock:
                session = sessions.get(session_id)

            if session is None:
                yield f'data: {json.dumps({"error": "Session not found"})}\n\n'
                return

            queue = session.get('event_queue', [])
            while sent < len(queue):
                yield f'data: {json.dumps(queue[sent])}\n\n'
                sent += 1

            if session.get('status') == 'done':
                yield f'data: {json.dumps({"type": "session_done", "job_id": session.get("job_id"), **session.get("live_summary", {})})}\n\n'
                return

            time.sleep(0.3)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/stream/stop/<session_id>', methods=['POST'])
def stream_stop(session_id: str):
    """Signal end of a streaming session; compute summary and persist as a job."""
    with sessions_lock:
        session = sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.get('status') in ('finalizing', 'done'):
        return jsonify({'status': session.get('status'), 'job_id': session.get('job_id')})

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0.0,
            'pipeline_message': 'Queued for processing',
            'filename': f'session_{int(time.time())}.mp4',
            'created_at': time.time(),
            'source': 'live_session',
        }

    with sessions_lock:
        session['status'] = 'finalizing'
        session['job_id'] = job_id
        model_for_job = session.get('model', YOLO_MODEL)
        conf_for_job = session.get('conf', YOLO_CONF)
        imgsz_for_job = session.get('imgsz', YOLO_IMGSZ)

    def _finalize():
        # Brief wait for any in-flight chunk processing
        time.sleep(1.5)
        with sessions_lock:
            s = sessions.get(session_id)
            if not s:
                return
            putts  = s.get('putts', [])
            made   = sum(1 for p in putts if p['result'] == 'made')
            missed = len(putts) - made
            s['live_summary'] = {
                'total_putts': len(putts),
                'made':        made,
                'missed':      missed,
                'make_pct':    round(made / len(putts) * 100, 1) if putts else 0,
            }
            local_job_id = s.get('job_id')

        chunk_paths = _session_chunk_paths(session_id)
        input_path = UPLOAD_DIR / f'{local_job_id}_session_input.mp4'
        with jobs_lock:
            if local_job_id in jobs:
                jobs[local_job_id].update({
                    'status': 'processing',
                    'progress': 0.0,
                    'pipeline_message': 'Assembling session video',
                })
        merged_ok = _merge_session_chunks(chunk_paths, input_path)
        if not merged_ok:
            with jobs_lock:
                jobs[local_job_id].update({
                    'status': 'error',
                    'error': 'Could not assemble live session video',
                })
            with sessions_lock:
                if session_id in sessions:
                    sessions[session_id]['status'] = 'done'
            _on_job_complete(local_job_id)
            return

        for p in chunk_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

        thread = threading.Thread(
            target=process_video,
            args=(str(input_path), str(OUTPUT_DIR), local_job_id, jobs),
            kwargs={
                'model_name':  model_for_job,
                'conf':        conf_for_job,
                'imgsz':       imgsz_for_job,
                'frame_skip':  FRAME_SKIP,
                'on_complete': _on_job_complete,
            },
            daemon=True,
        )
        thread.start()

        with sessions_lock:
            if session_id in sessions:
                sessions[session_id]['status'] = 'done'

    threading.Thread(target=_finalize, daemon=True).start()
    return jsonify({'status': 'stopping', 'job_id': job_id})


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': f'File too large. Maximum is {MAX_UPLOAD_MB} MB.'}), 413


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    scheme = 'http'
    print(f'\n  PuttPro Video Analysis')
    print(f'  Model:       {YOLO_MODEL}  conf={YOLO_CONF}  imgsz={YOLO_IMGSZ}')
    print(f'  Depth:       {"on" if DEPTH_ENABLED else "off"}  model={DEPTH_MODEL_NAME}  skip={DEPTH_FRAME_SKIP}')
    print(f'  Distance:    disc={DISC_DIAMETER_IN:.1f}in  camera_hfov={CAMERA_HFOV_DEG:.1f}deg')
    print(f'  Frame skip:  {FRAME_SKIP}')
    print(f'  Max upload:  {MAX_UPLOAD_MB} MB')
    print(f'  Listening:   {scheme}://localhost:{PORT}')
    print(f'  Mobile:      {scheme}://localhost:{PORT}/mobile')
    print(f'  Dashboard:   {scheme}://localhost:{PORT}/dashboard')
    print(f'  Streaming:   /stream/start  /stream/chunk/<id>  /stream/stop/<id>\n')
    print('  Dev note (Chrome mobile + HTTP):')
    print('  chrome://flags/#unsafely-treat-insecure-origin-as-secure')
    print(f'  add origin:  http://<YOUR_PC_IP>:{PORT}\n')
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=False,
        threaded=True,
    )
