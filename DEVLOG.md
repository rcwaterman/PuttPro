# PuttPro Development Log

---

## Features

| Feature | Status | Notes |
|---|---|---|
| Flask HTTPS server | ✅ | mkcert preferred; self-signed fallback |
| MJPEG video stream (`/video_feed`) | ✅ | Event-driven, no polling timer |
| Local webcam source | ✅ | `LocalCamera`, background capture thread |
| Remote frame upload (`POST /upload_frame`) | ✅ | Token auth, rate limit, JPEG validation |
| Raspberry Pi sender | ✅ | `pi_sender.py` |
| Mobile browser sender (`/mobile`) | ✅ | `getUserMedia`, OffscreenCanvas Worker, explicit `Record` start/stop button |
| Native mobile app (Android/iOS) | ✅ | Capacitor 6 — `mobile/` directory |
| **Video upload analysis service** | ✅ | `video_analysis/` — upload → YOLO → annotated MP4 + putting analytics |
| Standalone native app (on-device YOLO) | ✅ | React Native + ONNX Runtime — `mobile_standalone/` |
| YOLO detection (test model) | ✅ | YOLOv8n COCO; includes `frisbee` class 29 |
| YOLO fine-tuned model (disc/basket) | 🔲 | Planned — swap model file in both apps |
| Class filtering (disc-golf only) | ✅ | Only frisbee / disc golf basket / chains retained |
| Disc trajectory analysis | ✅ | Track bounding box centers, compute speed/angle, classify made/missed |
| Putt analytics dashboard | ✅ | `/dashboard` — make %, trend chart, flight data table |
| Mobile PWA (record + upload + stats) | ✅ | `video_analysis/templates/mobile.html` — 3-tab PWA with explicit `Enable Camera` / `Record` / `Start Live` action button |
| GPU inference (YOLO, server) | ✅ | CUDA FP32 — RTX 4080 SUPER; FP16 breaks YOLO inference, do not use |
| GPU inference (YOLO, mobile) | ✅ | CoreML (iOS) / NNAPI (Android) via ONNX Runtime |
| MediaPipe hand landmark detection | ❌ | Removed 2026-03-29 — not suited to disc flight |
| Token auth (persistent) | ✅ | `.upload_token` survives restarts |
| Rate limiting | ✅ | 70 req/s per IP, sliding window |
| QR code in startup output | ✅ | Mobile URL + CA cert URL |
| mkcert trusted cert (PC browser) | ⚠️ | Requires `mkcert` installed — see HTTPS section |
| mkcert CA cert for mobile trust | ⚠️ | Plain HTTP server on port 5001, requires `mkcert` |

---

## HTTPS / "Not Secure" Status

### Root cause
`mkcert` is **not installed**. The server falls back to a self-signed certificate.
Browsers flag self-signed certs regardless of domain or IP.

### Fix (one-time, per machine)
```
winget install FiloSottile.mkcert
```
Restart the server. It automatically runs `mkcert -install` (installs the local
CA into Windows + browser trust stores) and regenerates `cert.pem`/`key.pem`.

### Mobile devices / native app ("Not Secure")
Even after mkcert on the PC, mobile devices maintain their own trust store.

**Workflow (after mkcert is installed on PC):**
1. Start the server — it prints a second QR code for `http://<IP>:5001/`
2. Scan that QR on the mobile device to download `PuttPro-CA.crt`
3. Install the cert:
   - **Android**: tap downloaded file → install as *CA certificate*
   - **iOS**: Settings → General → VPN & Device Management → trust
4. Restart the browser / reopen the native app

**Native app (Android) additional step:**
After `npx cap add android`, run `mobile/android_patch/apply.bat`, then add
`android:networkSecurityConfig="@xml/network_security_config"` to `<application>`
in `AndroidManifest.xml`. This tells the WebView to trust user-installed CAs.

---

## Standalone Native App (on-device inference)

**Stack**: React Native (bare TypeScript) + react-native-vision-camera v4 + ONNX Runtime
**Location**: `mobile_standalone/`
**Architecture**: Camera frames captured on a native Worklet thread → `toArrayBuffer()` →
passed to JS thread → preprocess (NV12/BGRA → Float32 NCHW 320×320) → ONNX inference →
NMS → bounding box overlay. No server required.

| Setting | Value |
|---|---|
| Model | `yolov8n_320.onnx` (exported at imgsz=320, opset 12) |
| Inference FPS target | 15 (configurable in `CameraScreen.tsx`) |
| iOS execution provider | CoreML → CPU fallback |
| Android execution provider | NNAPI → CPU fallback |
| Confidence threshold | 0.4 |
| IoU threshold (NMS) | 0.45 |

### Setup
```bash
# 1. Export model (from project root)
python mobile_standalone/export_model.py

# 2. Initialize + build
cd mobile_standalone
setup.bat          # Windows
bash setup.sh      # Mac/Linux

# 3. Run
cd PuttProApp
npx react-native run-android   # or run-ios (macOS only)
```

---

## Server-Connected Native App

**Stack**: Capacitor 6 wrapping vanilla HTML/JS. No bundler required.
**Location**: `mobile/`
**Architecture**:
- **Setup screen** — user pastes the server mobile URL (or scans with built-in jsQR scanner). Token + server base URL are parsed from the URL and saved to `localStorage`.
- **Streaming screen** — identical frame capture/upload pipeline to `mobile.html`. Uses `requestVideoFrameCallback`, OffscreenCanvas Worker, 2 concurrent uploads.
- On relaunch, config is loaded from `localStorage` and streaming starts immediately.

### Build steps (one-time)
```
winget install OpenJS.NodeJS.LTS   # install Node.js if not present
cd mobile
npm install                        # installs Capacitor + jsqr
npm run prepare                    # copies jsQR.js into src/
npx cap add android                # generates android/ project
npx cap add ios                    # macOS + Xcode required
cd android_patch && apply.bat      # apply Android cert trust patch
npx cap sync                       # copies web assets + syncs plugins
```

### Open in IDE
```
npx cap open android   # opens Android Studio
npx cap open ios       # opens Xcode (macOS only)
```

### Iterating
After changing `mobile/src/index.html`:
```
npx cap sync && npx cap open android
```

---

## YOLO

| Setting | Value | Notes |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | Nano COCO; auto-downloads ~6 MB on first run |
| `YOLO_CONF` | `0.4` | Confidence threshold |
| `YOLO_IMGSZ` | `320` | Inference input size — smallest for CPU speed |
| Device | CPU | CUDA not detected; upgrade to GPU host for higher throughput |

**Swapping to a custom model**: set `YOLO_MODEL = '/path/to/custom.pt'` in `app.py`.
YOLO runs in `RemoteCamera._process_loop` (background thread) — the upload handler is never blocked.

---

## Performance

### Frame rate target: 60 fps

#### Server side
| Bottleneck | Fix applied |
|---|---|
| `sleep(0.033)` polling in `generate_frames` | `threading.Event` wait — wakes on every new frame |
| Flask request thread blocked by inference | `RemoteCamera` bounded queue + background worker thread |
| Flask reloader double-initialises threads | `use_reloader=False` |
| YOLO on CPU (~20fps throughput) | Queue drops frames; browser still receives at display rate with stale annotations |

#### Client side (mobile / native app)
| Bottleneck | Fix applied |
|---|---|
| `canvas.toBlob()` blocks main thread | OffscreenCanvas Worker (inline blob URL in native app) |
| One upload at a time | `MAX_IN_FLIGHT = 2` pipelines encoding latency with RTT |
| Timer drift | `requestVideoFrameCallback`; `requestAnimationFrame` fallback |
| getUserMedia capped at 30fps | `frameRate: { ideal: 60 }` — actual rate is device-dependent |

### User feedback log
| Date | Observation | Action taken |
|---|---|---|
| 2026-03-29 | "limitation in frame rate is coming from the client side" | Worker encoder, parallel uploads, requestVideoFrameCallback |
| 2026-03-29 | "not secure warning still showing on both client and server" | CA HTTP server port 5001; mkcert instructions in startup |
| 2026-03-29 | Hand tracking not suited to first-person disc flight view | Removed MediaPipe; switched to YOLO pipeline |
| 2026-03-29 | Wants deployable Android/iOS app + test YOLO model | Capacitor 6 native app in `mobile/`; YOLOv8n COCO test model added |
| 2026-03-30 | Wants fully standalone app — model and all on device | React Native standalone app in `mobile_standalone/`; ONNX Runtime with CoreML/NNAPI |
| 2026-03-31 | Standalone app build issues on Windows; pivoted to web upload service | `video_analysis/` — Flask + YOLO, annotated MP4 output, SSE progress, detection JSON |
| 2026-03-31 | Wants disc-golf-only class tracking, putt made/missed classification, analytics dashboard | Class filter in processor.py; trajectory analysis; `/dashboard` with trend chart + table; mobile stats tab |
| 2026-03-31 | Mobile web flow needed a visible control to start camera and live streaming | Added labeled record action control in `video_analysis/templates/mobile.html` to drive camera enable + upload/live start/stop |
| 2026-03-31 | Client `/mobile` page still lacked a visible record control | Added `Record` / `Stop` button in `app/templates/mobile.html` to initiate camera + frame streaming on demand |
| 2026-03-31 | Testing needed to target `video_analysis/app.py` frontend only | Updated `video_analysis/templates/mobile.html` to one-tap record/start-live and added no-cache behavior (`video_analysis/app.py`, `video_analysis/static/sw.js`) |
| 2026-03-31 | Needed explicit camera permission UX on mobile web | Added in-app camera permission popup (`Allow Camera` / `Not now`) in `video_analysis/templates/mobile.html` and wired it into record/live actions |
| 2026-03-31 | Wanted separate mobile flows: Start Session vs local Upload, with session results in History | Split `video_analysis` mobile UI into distinct Session and Upload pages (Session as default landing), and changed `/stream/stop` to persist a real analysis job for History |
| 2026-03-31 | Mobile camera permission blocked on LAN URL | Added secure-context guidance + `Open Secure URL` action in `video_analysis/templates/mobile.html`; enabled HTTPS-by-default (`USE_HTTPS`) in `video_analysis/app.py` |
| 2026-03-31 | Requested simple HTTP-only dev flow (no cert/ssl_utils in `video_analysis`) | Removed cert/`ssl_utils` coupling from `video_analysis/app.py`; mobile page now uses browser-native permission flow and shows Chrome HTTP dev-flag guidance |
| 2026-03-31 | Needed session putt timestamps + cleaner frisbee-only overlays + better fallback putt heuristics | Added per-putt event timestamps in `/stats` + dashboard table, restricted annotated boxes to frisbee labels, and switched trajectory classification to segmented event-based heuristic |
| 2026-03-31 | Wanted cleaner live session UI while keeping background analysis active | Removed Session-page action buttons and chunk status text, kept top live stats HUD, and reduced chunk interval for more continuous live analysis updates |
| 2026-03-31 | Wanted putt timestamps directly in History playback and less noisy metrics | Added tap-to-seek putt timeline under mobile result video; removed angle/disc-detection emphasis in favor of putt outcome stats (putts, made, make %) |
| 2026-03-31 | Requested unified mobile-like dashboard styling + per-putt clip playback from dashboard rows | Updated dashboard palette to match mobile theme and routed `View` links to `/clip/<job_id>?t=<timestamp>&pad=2` for timestamped putt clips |
| 2026-03-31 | Requested no JSON detection output and a stronger mobile session banner | Removed JSON detection download/output paths from `video_analysis` backend + templates, and redesigned the mobile Start Session banner with richer visual styling and state-aware messaging |
| 2026-03-31 | Requested cleaner top header UX, no in-feed session banner, depth mode playback toggle, and web deployment path | Removed camera-feed session banner, changed mobile header to fixed `PuttPro` + page context indicator, added depth-map processing/output route + playback toggle in mobile/desktop analyzers, and added deployment assets/docs (`video_analysis/DEPLOYMENT.md`, `Dockerfile`, `Procfile`, `wsgi.py`) |
| 2026-03-31 | Requested replacing depth model with a well-rated performant Hugging Face option | Swapped depth backend from torch.hub MiDaS loading to Hugging Face `depth-anything/Depth-Anything-V2-Small-hf`; updated deployment and README defaults accordingly |
| 2026-03-31 | Requested grayscale depth view and depth-based frisbee distance labels | Updated depth output to grayscale map for `View Depth` playback and overlaid `est X.X ft` frisbee distance labels using sampled depth-map values in processed frames |
| 2026-03-31 | Requested immediate save feedback on stop + more realistic distance estimation + smoother grayscale depth style | Made mobile stop-session hide live HUD immediately and show `Saving video...`; switched distance math to disc-size (8in) + camera FOV pinhole estimate; updated grayscale depth rendering to smoother, image-like shading |
| 2026-03-31 | Reported UI text encoding issues and depth-map correctness concerns | Repaired corrupted mobile UI glyph strings in `video_analysis/templates/mobile.html`; removed depth fallback paths in `video_analysis/processor.py` so jobs now require successful Hugging Face depth model load + inference (fail-fast on depth errors) |
| 2026-03-31 | Reported jobs stuck at `Analyzing frames - 0%` and requested clearer in-app processing visibility | Added explicit pipeline stage messages (`pipeline_message`) across backend job lifecycle, extended SSE signature to emit stage-only updates, hardened processor failure paths so jobs flip to `error` instead of stalling at 0%, and added a dedicated pipeline status bar on the mobile uploading/analyzing screen |
| 2026-03-31 | Requested next-day handoff: delete videos from History while preserving persistent stats | Plan for next implementation: add a History delete action that removes video artifacts and history entries, but keeps aggregated putting stats from those sessions in persistent data so longitudinal analytics remain intact. Also clean up the mobile dashboard layout, specifically the malformed formatting in the recent putt data section at the bottom of the page |
| 2026-04-01 | Enabled CUDA FP16, added History delete with persistent stats, fixed mobile dashboard formatting | `model.model.half()` enabled in `processor.py`; `DELETE /history/<job_id>` archives putt events to `stats_archive.json` before removing files so `/stats` retains longitudinal data; mobile history cards have tap-twice-to-confirm delete with fade-out; stats putt list uses `rgba()` backgrounds, capitalized labels, no double-border, friendly session filenames |

---

## Known Issues / Open Questions

- **YOLO fine-tuning needed**: YOLOv8n COCO `frisbee` class is a rough proxy. Disc golf discs and baskets require a labelled dataset and fine-tuning for reliable detection.
- **YOLO fine-tuning needed**: YOLOv8n COCO `frisbee` class is a rough proxy. Disc golf discs and baskets require a labelled dataset and fine-tuning for reliable detection.
- **Mobile camera FPS**: many phones cap `getUserMedia` rear camera at 30fps regardless of constraints. True 60fps capture requires native camera APIs (not available in browser/WebView).
- **Single MJPEG consumer**: `frame_event` is a `threading.Event`; only one browser tab gets efficient delivery. Multiple viewers need a broadcast `Condition` variable.
- **mkcert cert rotation**: delete `cert.pem` to force regeneration. Token rotation: delete `.upload_token`.
