"""
TLS certificate management for PuttPro.

Priority order:
  1. mkcert — generates a browser-trusted cert via a local CA (no security warnings).
              Install: winget install FiloSottile.mkcert
              The server runs 'mkcert -install' automatically on first start.
  2. Self-signed fallback — used when mkcert is not in PATH. All browsers will warn.
"""
import datetime
import ipaddress
import os
import shutil
import socket
import subprocess

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

_DIR = os.path.dirname(__file__)
CERT_FILE = os.path.join(_DIR, 'cert.pem')
KEY_FILE  = os.path.join(_DIR, 'key.pem')


def get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]


def get_ca_cert_path() -> str | None:
    """Return the path to the mkcert root CA cert, or None if mkcert is not installed."""
    mkcert_bin = shutil.which('mkcert')
    if not mkcert_bin:
        return None
    try:
        ca_root = subprocess.check_output(
            [mkcert_bin, '-CAROOT'], text=True, stderr=subprocess.DEVNULL
        ).strip()
        path = os.path.join(ca_root, 'rootCA.pem')
        return path if os.path.exists(path) else None
    except Exception:
        return None


def _cert_covers_ip(ip: str) -> bool:
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        return False
    try:
        with open(CERT_FILE, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        return ipaddress.IPv4Address(ip) in san.value.get_values_for_type(x509.IPAddress)
    except Exception:
        return False


def _cert_is_mkcert() -> bool:
    if not os.path.exists(CERT_FILE):
        return False
    try:
        with open(CERT_FILE, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read())
        attrs = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)
        return any('mkcert' in a.value.lower() for a in attrs)
    except Exception:
        return False


def _gen_mkcert(local_ip: str, mkcert_bin: str) -> tuple[str, str]:
    print(f'Generating mkcert certificate for {local_ip} ...')
    subprocess.run([mkcert_bin, '-install'], check=True)
    subprocess.run([
        mkcert_bin,
        '-cert-file', CERT_FILE,
        '-key-file',  KEY_FILE,
        local_ip, 'localhost', '127.0.0.1',
    ], check=True)
    print('mkcert certificate ready.')
    return CERT_FILE, KEY_FILE


def _gen_self_signed(local_ip: str) -> tuple[str, str]:
    print(f'Generating self-signed certificate for {local_ip} ...')
    key  = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'PuttPro Local')])
    now  = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName('localhost'),
                x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
                x509.IPAddress(ipaddress.IPv4Address(local_ip)),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    with open(KEY_FILE, 'wb') as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(CERT_FILE, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return CERT_FILE, KEY_FILE


def ensure_ssl_cert() -> tuple[str, str]:
    """
    Return (cert_path, key_path).

    Priority:
      1. PUTTPRO_CERT_FILE / PUTTPRO_KEY_FILE env vars — used in Docker/cloud
         where a real cert is mounted into the container.
      2. mkcert — generates a browser-trusted cert (dev, no warnings).
      3. Self-signed fallback — browser will warn.
    """
    env_cert = os.environ.get('PUTTPRO_CERT_FILE')
    env_key  = os.environ.get('PUTTPRO_KEY_FILE')
    if env_cert and env_key:
        if os.path.exists(env_cert) and os.path.exists(env_key):
            print(f'Using cert from environment: {env_cert}')
            return env_cert, env_key
        raise FileNotFoundError(
            f'PUTTPRO_CERT_FILE/KEY set but files not found: {env_cert}, {env_key}'
        )

    local_ip   = get_local_ip()
    mkcert_bin = shutil.which('mkcert')

    if mkcert_bin:
        if not _cert_covers_ip(local_ip) or not _cert_is_mkcert():
            return _gen_mkcert(local_ip, mkcert_bin)
        return CERT_FILE, KEY_FILE

    if not _cert_covers_ip(local_ip):
        return _gen_self_signed(local_ip)
    return CERT_FILE, KEY_FILE
