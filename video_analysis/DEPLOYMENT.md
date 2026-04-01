# PuttPro Deployment Guide

This app can be deployed so phone testing works off-LAN and over HTTPS.

## Quickest Option: Cloudflare Tunnel (no code deploy)

Use this when you want internet access to your local dev server right away.

1. Run the app locally:
```bash
cd video_analysis
python app.py
```
2. Install `cloudflared` and log in:
```bash
cloudflared tunnel login
```
3. Start a quick tunnel to local port 5050:
```bash
cloudflared tunnel --url http://localhost:5050
```
4. Open the generated `https://...trycloudflare.com/mobile` URL on your phone.

Notes:
- This gives HTTPS, so mobile camera permissions work normally.
- URL changes each run unless you configure a named tunnel.

## Hosted Option: Render/Railway/Fly.io

This repo now includes:
- `video_analysis/Dockerfile`
- `video_analysis/Procfile`
- `video_analysis/wsgi.py`

### Render (Docker)
1. Create a new Web Service from this repo.
2. Set root directory to `video_analysis`.
3. Use Docker deploy.
4. Set environment variables:
   - `PORT=5050`
   - `YOLO_MODEL=yolov8n.pt`
   - `YOLO_CONF=0.35`
   - `YOLO_IMGSZ=640`
   - `FRAME_SKIP=1`
   - Depth pipeline (required):
     - `DEPTH_ENABLED=1`
     - `DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf`
     - `DEPTH_FRAME_SKIP=2`
     - `DISC_DIAMETER_IN=8.0`
     - `CAMERA_HFOV_DEG=69.0`
5. Deploy and use the generated HTTPS URL.

### Railway (Procfile or Docker)
1. Create project from repo.
2. Set root to `video_analysis`.
3. Add same environment variables as above.
4. Deploy and test at `https://<app-domain>/mobile`.

## Production Notes

- Use object storage (S3/R2) for `uploads/` and `outputs/` when scaling.
- Keep one worker per GPU process for YOLO-heavy workloads.
- For persistent analytics history across restarts, move from local JSON DB to a real database.
