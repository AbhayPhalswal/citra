"""
=============================================================================
CITRA UI BRIDGE
=============================================================================
Thin, fire-and-forget notifier from jarvis_voice_assistant.py's main
process to citra_ui_server.py's local web UI process.

WHY TWO SEPARATE PROCESSES, TALKING OVER LOCALHOST HTTP:
jarvis_voice_assistant.py already runs a real-time audio pipeline across
several threads (reader thread, consumer/state-machine loop, TTS playback
thread) with careful single-writer state discipline (see AssistantState's
docstring). Bolting an asyncio-based web/WebSocket server directly into
that would mean running an asyncio event loop alongside all that existing
threading code — a second concurrency model layered on the first, with
its own footguns. Keeping the UI server as its own process and crossing
the boundary with small, one-way, fire-and-forget HTTP POSTs is simpler
and safer: the UI is a SUPPLEMENT to voice control, never load-bearing
for it, so if the UI server isn't running (or is slow, or crashes), the
voice assistant must never notice or care. That's the whole reason
every function here swallows its own exceptions.

WHY FIRE-AND-FORGET (background thread, not a blocking call):
Notifying the UI happens on the same hot paths as speech (every state
transition, every spoken sentence) — adding even a few milliseconds of
synchronous HTTP round-trip latency to those paths would work directly
against the project's low-latency goals. Each notify_* function spawns
a short-lived daemon thread and returns immediately.
=============================================================================
"""
import logging
import os
import threading

import requests
import urllib3

logger = logging.getLogger("citra_ui_bridge")

# citra_ui_server.py serves HTTPS the moment citra_cert.pem/citra_key.pem
# exist (see generate_tls_cert.py), plain HTTP otherwise -- and a plain
# HTTP request to a TLS-only listener fails at the handshake, not with a
# clean 4xx. That failure looked IDENTICAL to "UI server isn't running"
# (both raise requests.exceptions.RequestException, silently swallowed by
# _post() below) until traced directly: captions/state/transcript pushes
# were failing on every single call once TLS was enabled, because this
# was still hardcoded to http://. Checking for the same two files
# citra_ui_server.py itself checks -- rather than hardcoding a scheme, or
# trying to share state across the process boundary -- keeps both files
# deriving the scheme from the same one source of truth without needing
# any actual coordination between the two processes.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_TLS_CERT_PATH = os.path.join(_PROJECT_ROOT, "citra_cert.pem")
_TLS_KEY_PATH = os.path.join(_PROJECT_ROOT, "citra_key.pem")
_UI_SERVER_IS_HTTPS = os.path.exists(_TLS_CERT_PATH) and os.path.exists(_TLS_KEY_PATH)
UI_SERVER_BASE_URL = f"{'https' if _UI_SERVER_IS_HTTPS else 'http'}://localhost:8765"
UI_NOTIFY_TIMEOUT_SECONDS = 1.0  # short on purpose -- if the UI server
                                   # isn't responsive, fail fast and drop
                                   # the notification rather than let a
                                   # background thread hang around.

if _UI_SERVER_IS_HTTPS:
    # Self-signed cert, loopback-only traffic to a process on this same
    # machine -- there's no real MITM risk to guard against here the way
    # there would be over a real network, so skip verification rather
    # than bundling citra_cert.pem as a trusted CA for one internal call.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _post(path: str, payload: dict) -> None:
    try:
        requests.post(
            f"{UI_SERVER_BASE_URL}{path}",
            json=payload,
            timeout=UI_NOTIFY_TIMEOUT_SECONDS,
            verify=not _UI_SERVER_IS_HTTPS,
        )
    except requests.exceptions.RequestException:
        # The UI server being offline is the common case (most runs of
        # the voice assistant won't have it open at all) -- not worth
        # logging at error/warning level every single time.
        pass


def notify_caption(text: str) -> None:
    """Tell the UI what Citra is about to speak, for live on-screen captions."""
    threading.Thread(target=_post, args=("/caption", {"text": text}), daemon=True).start()


def notify_state(state_name: str) -> None:
    """Tell the UI which AssistantState Citra just entered (IDLE/LISTENING/etc.)."""
    threading.Thread(target=_post, args=("/state", {"state": state_name}), daemon=True).start()


def notify_transcript(text: str) -> None:
    """Tell the UI what Citra heard the user say (distinct from her own captions)."""
    threading.Thread(target=_post, args=("/transcript", {"text": text}), daemon=True).start()
