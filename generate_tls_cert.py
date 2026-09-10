"""
=============================================================================
GENERATE TLS CERT — self-signed certificate for citra_ui_server.py
=============================================================================
WHY THIS EXISTS: browsers only allow microphone access (getUserMedia) from
a "secure context" — an https:// origin, or http://localhost specifically.
The dashboard is reached from other devices at this machine's LAN IP
(e.g. http://192.168.0.9:8765), which is neither, so iOS Safari (and every
other modern browser) refuses microphone access outright on the current
plain-HTTP server, no matter what the page's own JavaScript does — this
is a platform-level gate the app can't route around.

A self-signed certificate genuinely satisfies "secure context": once the
browser's one-time security warning is accepted for this origin, the
connection really is HTTPS and getUserMedia works normally. The warning
exists because this cert isn't signed by a CA anyone's browser already
trusts (that would need a real domain name and a service like Let's
Encrypt, neither of which applies to a LAN IP address) — it does NOT mean
the connection itself is insecure once accepted; it's genuinely encrypted
TLS, just not chain-of-trust-verified TLS.

Run this once. It writes citra_cert.pem + citra_key.pem into this
directory; citra_ui_server.py loads them at startup if present, generating
nothing itself (kept as a separate, explicit step rather than silent
auto-generation, so it's obvious when/why a new cert appears — a
regenerated cert invalidates every device's previous "trust this
certificate" decision, which is worth being deliberate about rather than
having happen implicitly on some unrelated code change).

Valid for 10 years (a LAN-only dev cert has no real reason to expire
sooner) and covers this machine's current LAN IP plus localhost/127.0.0.1
— if this machine's LAN IP changes later (new router, static reservation
changes), rerun this script.
=============================================================================
"""
import datetime
import ipaddress
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_PATH = "citra_cert.pem"
KEY_PATH = "citra_key.pem"


def _get_lan_ip() -> str:
    """Best-effort LAN IP discovery: opens a UDP socket toward a public IP
    (nothing is actually sent — UDP is connectionless — this just asks the
    OS routing table which local interface/IP would be used) rather than
    parsing `ipconfig` output, which is fragile across Windows locales and
    adapter naming."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    lan_ip = _get_lan_ip()
    print(f"Generating self-signed certificate for LAN IP {lan_ip} (+ localhost)...")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "citra.local"),
    ])

    san_entries = [
        x509.DNSName("localhost"),
        x509.DNSName("citra.local"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        san_entries.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
    except ValueError:
        pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(CERT_PATH, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Wrote {CERT_PATH} and {KEY_PATH}.")
    print(f"Covers: localhost, 127.0.0.1, citra.local, {lan_ip}")
    print("Valid for 10 years. Rerun this script if the LAN IP ever changes.")


if __name__ == "__main__":
    main()
