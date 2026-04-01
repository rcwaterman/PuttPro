# PuttPro Video Analysis

Upload a disc golf putting video; get it back as an annotated video with putting-focused session stats.

## Setup

```bash
# From the project root, using the existing venv:
pip install -r video_analysis/requirements.txt
# GPU (already done if main app is set up):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Run

```bash
cd video_analysis
python app.py
# → http://localhost:5050
```

## Mobile web recording and streaming

Open `http://localhost:5050/mobile` and use the split mobile flows:

- `Start Session` (default landing page) is dedicated to live camera streaming:
  - Tap the center record control to start/stop live chunked streaming to:
  - `POST /stream/start`
  - `POST /stream/chunk/<session_id>`
  - `POST /stream/stop/<session_id>`
- `Upload` is a separate page for selecting a local phone video (`Choose Video`) and analyzing it.

When a live session is stopped, the backend now assembles and processes the session into a normal analysis job, so the analyzed output still appears in History.
Live HUD stats at the top update during active sessions; chunk upload status text is hidden from the session UI.

### Notes for frontend testing

- Test against the `video_analysis` service URL (`http://localhost:5050/mobile`), not the root app service.
- `/mobile` is returned with no-cache headers to reduce stale UI during iterative testing.

## Environment variables (all optional)

| Variable     | Default      | Description                                  |
|--------------|--------------|----------------------------------------------|
| `YOLO_MODEL` | `yolov8n.pt` | Model file — checks `../app/` then downloads |
| `YOLO_CONF`  | `0.35`       | Default confidence threshold                 |
| `YOLO_IMGSZ` | `640`        | Default inference image size                 |
| `FRAME_SKIP` | `1`          | Analyze every Nth frame (1 = every frame)    |
| `DEPTH_ENABLED` | `1`       | Required for processing; if set to `0` jobs fail with a depth pipeline error |
| `DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | Required Hugging Face depth model (no fallback mode) |
| `DEPTH_FRAME_SKIP` | `2`    | Recompute depth every N frames (reuses last map between) |
| `DISC_DIAMETER_IN` | `8.0`  | Disc diameter used for distance estimation in overlays |
| `CAMERA_HFOV_DEG` | `69.0`  | Camera horizontal FOV used by the size-based distance formula |
| `PORT`       | `5050`       | Listening port                               |

## Output

For each uploaded video the service produces:

- **Annotated MP4** — frisbee-only bounding boxes drawn on analyzed frames, H.264 encoded (requires `ffmpeg` on PATH for best browser compatibility; falls back to `mp4v` if not available)
- **Depth MP4** — grayscale depth-map visualization aligned to the same timeline (`/result_depth/<job_id>`)
- **Putt analytics** — putt count, made/missed totals, make %, and per-putt timestamps used by History and Dashboard clip links

Dashboard stats now expose per-putt event timestamps (`timestamp_s`, `timestamp_label`) so session timelines can show where putts occurred within a recording.
Mobile History playback now includes a tap-to-seek putt timeline under the result video.
Mobile and desktop result playback include a `Depth Mode` toggle to switch between analyzed video and depth-map video.
Frisbee overlays include estimated distance (`est X.X ft`) from apparent disc size (8-inch diameter) and camera FOV.
If depth model load or inference fails, the job fails explicitly instead of falling back to synthetic depth output.
Dashboard `View` links now open a putt-specific clip (`/clip/<job_id>?t=<timestamp>&pad=2`) when timestamp data is available.

## Deploying for off-LAN mobile testing

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Cloudflare Tunnel quick setup (fastest path to public HTTPS testing)
- Hosted deployment setup using included `Dockerfile`, `Procfile`, and `wsgi.py`

## ffmpeg (recommended)

Install ffmpeg for H.264 output that plays in all browsers without extra codecs:

```
winget install Gyan.FFmpeg
```

Without ffmpeg the service still works but browser playback may require downloading the file first.

## Performance

On an RTX 4080 SUPER with `yolov8n.pt` at `imgsz=640`:
- ~60–80 frames/sec inference throughput
- A 30-second clip at 60fps (1800 frames) processes in ~25 seconds
- Use `imgsz=320` for ~3× faster processing at the cost of small-object accuracy
