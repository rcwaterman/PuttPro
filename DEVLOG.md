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
| Mobile browser sender (`/mobile`) | ✅ | `getUserMedia`, OffscreenCanvas Worker |
| Native mobile app (Android/iOS) | ✅ | Capacitor 6 — `mobile/` directory |
| YOLO detection (test model) | ✅ | YOLOv8n COCO; includes `frisbee` class 29 |
| YOLO fine-tuned model (disc/basket) | 🔲 | Planned — swap `YOLO_MODEL` in `app.py` |
| GPU inference (YOLO) | ⚠️ | CUDA not detected; running on CPU |
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

## Native App

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

---

## Known Issues / Open Questions

- **CUDA not available**: YOLO runs on CPU, capping processed output at ~15-25 fps on this machine. Install CUDA + torch GPU build on a CUDA-capable host to raise this.
- **YOLO fine-tuning needed**: YOLOv8n COCO `frisbee` class is a rough proxy. Disc golf discs and baskets require a labelled dataset and fine-tuning for reliable detection.
- **Mobile camera FPS**: many phones cap `getUserMedia` rear camera at 30fps regardless of constraints. True 60fps capture requires native camera APIs (not available in browser/WebView).
- **Single MJPEG consumer**: `frame_event` is a `threading.Event`; only one browser tab gets efficient delivery. Multiple viewers need a broadcast `Condition` variable.
- **mkcert cert rotation**: delete `cert.pem` to force regeneration. Token rotation: delete `.upload_token`.
