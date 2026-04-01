# PuttPro Video Analysis — Production Scaling Guide

## What needs to scale

Each analysis job is CPU/GPU-bound (YOLO inference over every frame) and can take 5–60 seconds depending on clip length and hardware. The bottleneck is the worker, not the web server.

---

## Option 1 — Single GPU VPS (recommended for early users, ≤ 20 concurrent jobs/day)

**Providers**: RunPod, Lambda Labs, Vast.ai, AWS g4dn.xlarge

**Stack**: Same Flask app, run behind `gunicorn` with a single worker (no threads needed — GPU is the bottleneck). A background thread pool of 2–4 workers handles job queue.

```
gunicorn -w 1 --threads 4 -b 0.0.0.0:5050 app:app
```

**Cost**: ~$0.40–0.80/hr on RunPod/Lambda (RTX 3090/4090)
**Pros**: Zero infrastructure change, instant deploy
**Cons**: No auto-scale; long queue during traffic bursts

---

## Option 2 — Modal.com (recommended for serverless MVP, pay-per-second)

Modal runs GPU containers on demand — you pay only during actual inference. Zero idle cost.

### Architecture

```
Browser → Flask (Modal web endpoint) → Modal Queue → GPU worker function
                                                     ↓
                                              outputs/  (Modal Volume)
```

### Key changes

1. Move `process_video()` into a `@modal.function(gpu="T4")` decorated function
2. Use a `modal.Queue` instead of in-process `jobs` dict
3. Use a `modal.Volume` or S3 for output storage (workers are ephemeral)
4. Flask becomes a lightweight `@modal.asgi_app()` that enqueues jobs and polls status

```python
import modal

app_modal = modal.App("puttproanalysis")
vol       = modal.Volume.from_name("puttpro-outputs", create_if_missing=True)

@app_modal.function(gpu="T4", volumes={"/outputs": vol}, timeout=600)
def run_inference(input_bytes: bytes, job_id: str, conf: float, imgsz: int) -> dict:
    # write input_bytes to temp file, run process_video(), return summary
    ...
```

**Cost**: ~$0.000225/sec on T4; a 30-second clip = ~$0.007
**Pros**: True zero-idle, auto-scales to many parallel jobs
**Cons**: ~5s cold start per new container; slightly more complex code

---

## Option 3 — AWS S3 + SQS + EC2 Auto Scaling (production at scale)

For high traffic (hundreds of jobs/day):

```
Browser → ALB → Flask (ECS Fargate, no GPU) → S3 upload → SQS → EC2 Auto Scaling Group (GPU)
                                                                   ↓
                                                              S3 output bucket
```

### Components

| Component | Purpose |
|---|---|
| S3 (input bucket) | Stores uploaded videos — pre-signed URLs for direct browser upload |
| SQS FIFO queue | Job queue; each message = `{job_id, s3_key, conf, imgsz, frame_skip}` |
| EC2 g4dn.xlarge ASG | GPU workers; scale 0→N based on queue depth |
| S3 (output bucket) | Annotated MP4 + depth MP4 + putt analytics metadata; pre-signed URLs for playback |
| DynamoDB | Job status table (replaces in-memory `jobs` dict) |
| CloudFront | CDN for output video delivery |

### Scale trigger

```
Target metric: SQS ApproximateNumberOfMessagesVisible
Scale out: > 3 messages → add 1 instance
Scale in: 0 messages for 5 min → terminate instances
```

**Cost (at 500 jobs/day)**: ~$200–400/month depending on clip length
**Pros**: Handles thousands of concurrent jobs; zero-downtime deploys; built-in durability
**Cons**: Significant infrastructure work; overkill for < 100 jobs/day

---

## Option 4 — Celery + Redis (self-hosted, mid-scale)

Drop-in upgrade to the current Flask app if you want a job queue without cloud migration:

```
pip install celery[redis] redis
```

Replace `threading.Thread` with `celery.send_task()`. Run multiple workers:

```bash
celery -A tasks worker --concurrency=2 -Q video --loglevel=info
```

**When to use**: You already have a VPS and want queue durability + retries without moving to cloud. Works with Redis on the same machine.

---

## Recommendation path

| Stage | Users | Recommendation |
|---|---|---|
| Development / beta | < 10/day | Current Flask + threads on RTX 4080 |
| Early public | 10–100/day | Modal.com serverless GPU |
| Growth | 100–1000/day | Celery + Redis on 2× GPU VPS |
| Scale | 1000+/day | AWS SQS + Auto Scaling Group |

---

## Fine-tuned model (disc golf dataset)

The current COCO `frisbee` class is a rough proxy. For reliable disc, basket, and chains detection:

1. Collect 500–2000 labelled frames (disc in flight, basket, chains) using [Roboflow](https://roboflow.com)
2. Fine-tune YOLOv8n: `yolo train model=yolov8n.pt data=discgolf.yaml epochs=100 imgsz=640`
3. Export to ONNX for mobile standalone: `yolo export model=best.pt format=onnx opset=12 imgsz=320`
4. Drop `best.pt` into `video_analysis/` and set `YOLO_MODEL=best.pt`

With a custom model the basket proximity classifier in `processor.py` will activate, replacing the frame-position heuristic with accurate made/missed detection.
