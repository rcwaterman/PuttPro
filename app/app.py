import http.server
import os
import secrets
import socket
import socketserver
import threading
import time
from collections import defaultdict

import qrcode
from flask import Flask, Response, abort, jsonify, render_template, request
from camera import LocalCamera, RemoteCamera, generate_frames
from ssl_utils import ensure_ssl_cert, get_ca_cert_path, get_local_ip

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 'local'  — webcam attached to this machine
# 'remote' — frames pushed by a remote device (Pi, phone, native app)
CAMERA_MODE = 'remote'

# YOLO inference
# Set to None to disable; set to a file path for a fine-tuned model.
# 'yolov8n.pt' downloads automatically on first run (~6 MB, COCO 80 classes).
# Includes 'frisbee' (class 29) for disc detection testing.
YOLO_MODEL = 'yolov8n.pt'
YOLO_CONF  = 0.4    # detection confidence threshold
YOLO_IMGSZ = 640    # inference input size; 640 is YOLO default — fine on GPU

PORT         = 5000
CA_HTTP_PORT = 5001   # plain HTTP, serves only the mkcert CA cert
MDNS_PORT    = PORT   # mDNS announces the same port as Flask
MAX_FRAME_BYTES = 5 * 1024 * 1024
RATE_LIMIT = 70       # max frame uploads per second per source IP
JPEG_MAGIC = b'\xff\xd8\xff'

# ---------------------------------------------------------------------------
# Persistent upload token
# ---------------------------------------------------------------------------
_TOKEN_FILE = os.path.join(os.path.dirname(__file__), '.upload_token')


def _load_or_create_token() -> str:
    if os.path.exists(_TOKEN_FILE):
        token = open(_TOKEN_FILE).read().strip()
        if len(token) >= 32:
            return token
    token = secrets.token_urlsafe(32)
    with open(_TOKEN_FILE, 'w') as f:
        f.write(token)
    return token


UPLOAD_TOKEN = _load_or_create_token()

# ---------------------------------------------------------------------------
# Rate limiter (per source IP, sliding window)
# ---------------------------------------------------------------------------
_buckets: dict[str, list[float]] = defaultdict(list)
_bucket_lock = threading.Lock()


def _rate_allowed(ip: str) -> bool:
    now = time.monotonic()
    with _bucket_lock:
        _buckets[ip] = [t for t in _buckets[ip] if now - t < 1.0]
        if len(_buckets[ip]) >= RATE_LIMIT:
            return False
        _buckets[ip].append(now)
        return True

# ---------------------------------------------------------------------------
# mDNS — announce server as 'puttpro.local' on the LAN.
# iOS resolves .local natively. Android support varies by version.
# Falls back silently if zeroconf is unavailable.
# ---------------------------------------------------------------------------

def _start_mdns(local_ip: str) -> None:
    try:
        from zeroconf import ServiceInfo, Zeroconf
        info = ServiceInfo(
            '_puttpro._tcp.local.',
            'PuttPro._puttpro._tcp.local.',
            addresses=[socket.inet_aton(local_ip)],
            port=MDNS_PORT,
            properties={b'token': UPLOAD_TOKEN.encode()},
        )
        zc = Zeroconf()
        zc.register_service(info)
        # zeroconf uses daemon threads internally; keep the instance alive.
        threading.Thread(target=lambda: time.sleep(1e9), daemon=True).start()
        print(f'mDNS: server announced as puttpro.local:{MDNS_PORT}')
    except Exception as e:
        print(f'mDNS unavailable ({e}) — LAN auto-discovery disabled.')

# ---------------------------------------------------------------------------
# CA cert plain-HTTP server (bootstrapping: lets mobile devices download the
# CA cert before they trust the HTTPS endpoint)
# ---------------------------------------------------------------------------

def _start_ca_http_server(ca_cert_path: str):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            with open(ca_cert_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-x509-ca-cert')
            self.send_header('Content-Disposition', 'attachment; filename="PuttPro-CA.crt"')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = socketserver.TCPServer(('', CA_HTTP_PORT), _Handler)
    server.allow_reuse_address = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FRAME_BYTES

if CAMERA_MODE == 'local':
    camera = LocalCamera(source=0, yolo_model=YOLO_MODEL, yolo_conf=YOLO_CONF, yolo_imgsz=YOLO_IMGSZ)
else:
    camera = RemoteCamera(yolo_model=YOLO_MODEL, yolo_conf=YOLO_CONF, yolo_imgsz=YOLO_IMGSZ)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/mobile')
def mobile():
    if request.args.get('token') != UPLOAD_TOKEN:
        abort(403)
    return render_template('mobile.html', token=UPLOAD_TOKEN)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if request.headers.get('X-Upload-Token') != UPLOAD_TOKEN:
        abort(401)
    if not _rate_allowed(request.remote_addr):
        abort(429)
    data = request.data
    if not data or not data.startswith(JPEG_MAGIC):
        abort(400)
    if not isinstance(camera, RemoteCamera):
        abort(400)
    camera.receive_frame(data)
    return '', 204


@app.route('/api/info')
def api_info():
    """
    LAN-only pairing endpoint.  Returns the upload token so the native app
    can auto-connect after mDNS resolves 'puttpro.local'.

    NOTE: This endpoint is unauthenticated by design — it is intended only
    for initial pairing on a trusted local network.  Remove or gate this
    behind auth before exposing the server to the public internet.
    """
    return jsonify({'token': UPLOAD_TOKEN, 'port': PORT})

# ---------------------------------------------------------------------------
# Startup output
# ---------------------------------------------------------------------------

def _qr(url: str) -> qrcode.QRCode:
    q = qrcode.QRCode(border=1)
    q.add_data(url)
    q.make(fit=True)
    return q


def _print_startup(host_ip: str, ca_cert_path: str | None) -> None:
    mobile_url = f'https://{host_ip}:{PORT}/mobile?token={UPLOAD_TOKEN}'
    sep = '─' * 64

    print(f'\n{sep}')
    print(f'  PuttPro  │  https://{host_ip}:{PORT}')
    print(f'  Mobile   │  {mobile_url}')
    print(f'  mDNS     │  puttpro.local:{PORT}  (iOS / modern Android)')
    print(f'{sep}')
    print('  Scan to open mobile camera:')
    print()
    _qr(mobile_url).print_ascii(invert=True)

    if ca_cert_path:
        ca_url = f'http://{host_ip}:{CA_HTTP_PORT}/'
        print(f'{sep}')
        print('  To remove "Not Secure" on mobile devices:')
        print('    Step 1 — scan QR below to download the CA cert onto your device')
        print('    Step 2 — install it:')
        print('      Android : tap downloaded file → install as "CA certificate"')
        print('      iOS     : Settings → General → VPN & Device Management → trust')
        print('    Step 3 — reopen app')
        print(f'  CA cert  │  {ca_url}')
        print()
        _qr(ca_url).print_ascii(invert=True)
    else:
        print(f'{sep}')
        print('  "Not Secure" warning active (self-signed cert).')
        print('  Fix: winget install FiloSottile.mkcert  then restart server.')

    print(f'{sep}\n')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    cert_file, key_file = ensure_ssl_cert()
    local_ip = get_local_ip()
    ca_cert_path = get_ca_cert_path()

    _start_mdns(local_ip)
    if ca_cert_path:
        _start_ca_http_server(ca_cert_path)
    _print_startup(local_ip, ca_cert_path)

    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=True,
        use_reloader=False,
        threaded=True,
        ssl_context=(cert_file, key_file),
    )
