"""
=============================================================================
CITRA CONTROL PANEL
=============================================================================
A tiny local web app for starting, stopping, and restarting Citra's two
services (jarvis_voice_assistant.py and citra_ui_server.py) by hand — the
manual equivalent of double-clicking start_citra.ps1, plus a stop/restart
path that script never had. Also surfaces a direct link to Citra's own web
UI (citra_ui_server.py's page) so both live behind one desktop icon.

Kept as its OWN process rather than added to citra_ui_server.py: that server
already has a job (serving the fullscreen room-control UI and relaying
hardware state) and it can be stopped/restarted by this panel — a control
tool that stopped working the moment the thing it controls was stopped would
be useless. Same reasoning as citra_ui_server.py's own separation from
jarvis_voice_assistant.py (see citra_ui_bridge.py).

PROCESS MODEL — mirrors start_citra.ps1's own launch shape (see that file):
each service actually runs as watchdog.py wrapping the real script, so
"stop" has to reach both layers or watchdog just relaunches whatever this
panel kills. _matching_processes() finds every python/pythonw process whose
command line mentions the target script; stop_service() kills the
watchdog-wrapping ones FIRST (so nothing is left alive to respawn a child),
then the app processes themselves.

RUN THIS DIRECTLY (not via watchdog.py): unlike the two services it
controls, an unattended crash-and-restart isn't the concern here — this is
a tool you open, click, and close. It binds 127.0.0.1:CONTROL_PANEL_PORT;
if that's already taken (an existing instance is running), it just opens a
browser tab at the existing one instead of erroring, so double-clicking the
desktop icon twice is harmless the same way start_citra.ps1 promises to be.
=============================================================================
"""
import asyncio
import datetime
import logging
import os
import socket
import subprocess
import sys
import time
import webbrowser

import psutil
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("citra_control_panel")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHONW = os.path.join(PROJECT_ROOT, "jarvis_venv", "Scripts", "pythonw.exe")
WATCHDOG = os.path.join(PROJECT_ROOT, "watchdog.py")
STATIC_DIR = os.path.join(PROJECT_ROOT, "citra_control_ui")

CONTROL_PANEL_PORT = 8764
UI_SERVER_PORT = 8765

# Same cert-presence check citra_ui_server.py uses to decide http vs https —
# duplicated rather than imported so this panel has zero import-time
# dependency on that module (it needs to work even if that file has a bug).
_cert = os.path.join(PROJECT_ROOT, "citra_cert.pem")
_key = os.path.join(PROJECT_ROOT, "citra_key.pem")
UI_SERVER_URL = (
    f"https://localhost:{UI_SERVER_PORT}/"
    if os.path.exists(_cert) and os.path.exists(_key)
    else f"http://localhost:{UI_SERVER_PORT}/"
)

# =============================================================================
# READINESS — why "the process exists" is NOT the status worth showing
# =============================================================================
# The voice assistant logs "Citra is ready. Listening for the wake word..."
# about 2 SECONDS after launch, but its Whisper model, semantic router and
# Silero VAD all load in BACKGROUND threads that finished ~49s later in a
# real measured startup on this machine. A wake word spoken in that window
# is heard and then goes nowhere — which is exactly the "I started it but
# it isn't replying to Hey Citra" confusion this panel exists to end. A
# green "Running" dot driven purely by process existence actively hides
# that gap, so each service reports a STAGE parsed from its own log
# instead, and only the true end-of-startup marker counts as ready.
#
# `stages` are ordered substrings matched against the log; the LAST one
# that appears is the current stage (they're logged in this order). Order
# here must match the order the app actually logs them, since that's what
# makes "furthest match wins" correct — not alphabetical or arbitrary.
SERVICES = {
    "voice_assistant": {
        "script": "jarvis_voice_assistant.py",
        "log": "jarvis_voice_assistant.log",
        "label": "Voice Assistant",
        # Its audio-ingest listener (jarvis_voice_assistant.py's
        # _AudioIngestHandler) — the same port citra_ui_server.py forwards
        # browser-mic audio to. Listening here proves the process got far
        # enough to serve, independent of anything in the log.
        "port": 8766,
        "stages": [
            ("Initializing wake word engine", "Starting wake word engine"),
            ("Initializing microphone", "Opening the microphone"),
            ("Loading Whisper model", "Loading speech recognition"),
            ("Citra is ready. Listening for the wake word", "Wake word live — still loading the rest"),
            ("Loading Piper voice", "Loading her voice"),
            ("Piper voice loaded", "Voice loaded"),
            ("Loading semantic router", "Loading the semantic router"),
            ("Semantic router ready", "Semantic router ready"),
            ("Silero VAD loaded", "Voice detection ready"),
        ],
        # The one line that means she can actually hold a conversation.
        "ready_marker": "full pipeline ready",
        "ready_label": "Ready — say “Hey Citra”",
    },
    "ui_server": {
        "script": "citra_ui_server.py",
        "log": "citra_ui_server.log",
        "label": "UI Server",
        "port": 8765,
        "stages": [
            ("Citra UI server starting on", "Binding the port"),
        ],
        # aiohttp is started with print=None here (see citra_ui_server.py's
        # run_app call), so there is no "now serving" line to match — the
        # port probe below is this service's real readiness signal.
        "ready_marker": None,
        "ready_label": "Ready — serving the UI",
    },
}

# Past this many seconds of uptime, a process that is listening on its port
# is treated as ready even if the startup markers have scrolled out of the
# log tail we read. Without this, a service running happily for hours would
# report "starting" forever, since its ready line is long gone from the
# window read below.
READY_ASSUME_SECONDS = 180.0

LOG_TAIL_BYTES = 96 * 1024   # enough to cover a full startup sequence
LOG_TAIL_LINES = 200         # what the panel's log view shows

# Matches the leading timestamp of a log line ("2026-08-20 23:16:21,796").
# Used to keep only lines from the CURRENT run: the log is appended across
# restarts, so a PREVIOUS run's "full pipeline ready" is sitting right
# there in the same file and would otherwise make a still-loading process
# look finished.
_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"


def _parse_log_timestamp(line: str) -> "float | None":
    try:
        return datetime.datetime.strptime(line[:23], _LOG_TIMESTAMP_FORMAT).timestamp()
    except (ValueError, IndexError):
        return None  # continuation lines, tqdm bars, banner art


def _read_log_tail(log_path: str) -> list:
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - LOG_TAIL_BYTES))
            raw = f.read()
    except OSError:
        return []
    # These logs are a MIX of encodings: watchdog.py opens the file as
    # utf-8, but the supervised child writes to that handle through its own
    # Windows stdout, which is cp1252 — so an em-dash in a log message
    # ("finished in 10.3s — full pipeline ready") lands as a cp1252 byte
    # inside an otherwise-utf-8 file and renders as a replacement char.
    # Falling back to cp1252 only when utf-8 actually produced garbage
    # keeps genuinely-utf-8 lines intact instead of mangling them the
    # other way round.
    text = raw.decode("utf-8", errors="replace")
    if "�" in text:
        text = raw.decode("cp1252", errors="replace")
    return text.splitlines()


def _is_noise(line: str) -> bool:
    """tqdm progress bars rewrite one line thousands of times and land in
    the log as unreadable 'Batches: 50%|#####' spam — useless in a status
    view and enough of it to push the real lines off screen."""
    stripped = line.strip()
    if not stripped:
        return True
    return "it/s]" in stripped or stripped.startswith(("Batches:", "Loading weights:"))


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.35)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _matching_processes(script_name: str) -> list:
    """Every python/pythonw process whose command line names this script AND
    lives under this project — matches both the watchdog wrapping it (whose
    cmdline embeds the inner command as one string) and the app itself."""
    matches = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmdline = p.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        cmdline_str = " ".join(cmdline)
        if script_name in cmdline_str and PROJECT_ROOT in cmdline_str:
            matches.append(p)
    return matches


def is_running(service_key: str) -> bool:
    return len(_matching_processes(SERVICES[service_key]["script"])) > 0


def service_status(service_key: str) -> dict:
    """Full status for one service: not just up/down, but HOW FAR into
    startup it is — see the SERVICES table's comment for why that
    distinction is the whole point of this panel."""
    svc = SERVICES[service_key]
    procs = _matching_processes(svc["script"])

    if not procs:
        return {
            "label": svc["label"],
            "running": False,
            "state": "stopped",
            "stage": "Stopped",
            "progress": 0,
            "uptime": None,
            "port_up": False,
        }

    # Start time comes from the APP process, not its watchdog: the watchdog
    # keeps its own start time across a crash-restart of the child, so
    # using it would report a stale uptime and mask the fact that the app
    # underneath just restarted and is loading models all over again.
    app_procs = [
        p for p in procs
        if "watchdog.py" not in " ".join(p.info.get("cmdline") or [])
    ]
    start_time = None
    for p in (app_procs or procs):
        try:
            created = p.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        start_time = created if start_time is None else min(start_time, created)

    uptime = (time.time() - start_time) if start_time else None
    port_up = _port_listening(svc["port"])

    # Only lines from THIS run count — see _LOG_TIMESTAMP_FORMAT's comment.
    cutoff = (start_time - 2.0) if start_time else 0.0
    stage_label, stage_index, ready_seen, last_error = None, -1, False, None
    for line in _read_log_tail(os.path.join(PROJECT_ROOT, svc["log"])):
        ts = _parse_log_timestamp(line)
        if ts is not None and ts < cutoff:
            continue
        for i, (needle, label) in enumerate(svc["stages"]):
            if needle in line and i > stage_index:
                stage_index, stage_label = i, label
        if svc["ready_marker"] and svc["ready_marker"] in line:
            ready_seen = True
        if "[ERROR]" in line or "Traceback (most recent call last)" in line:
            last_error = line.strip()[:300]

    total_stages = len(svc["stages"]) + 1  # +1 for the ready marker itself

    if ready_seen or (svc["ready_marker"] is None and port_up) or (
        uptime is not None and uptime > READY_ASSUME_SECONDS and port_up
    ):
        state, stage, progress = "ready", svc["ready_label"], 100
    else:
        state = "starting"
        stage = stage_label or "Starting up"
        progress = int(max(1, stage_index + 1) / total_stages * 100)

    return {
        "label": svc["label"],
        "running": True,
        "state": state,
        "stage": stage,
        "progress": progress,
        "uptime": round(uptime) if uptime is not None else None,
        "port_up": port_up,
        "port": svc["port"],
        "error": last_error,
    }


def stop_service(service_key: str) -> bool:
    procs = _matching_processes(SERVICES[service_key]["script"])
    if not procs:
        return False

    watchdog_procs, app_procs = [], []
    for p in procs:
        cmdline_str = " ".join(p.info.get("cmdline") or [])
        (watchdog_procs if "watchdog.py" in cmdline_str else app_procs).append(p)

    # Watchdogs first: killing the app before its watchdog just triggers a
    # respawn of the very thing we're trying to stop.
    ordered = watchdog_procs + app_procs
    for p in ordered:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(ordered, timeout=5)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return True


def start_service(service_key: str) -> bool:
    if is_running(service_key):
        return False
    svc = SERVICES[service_key]
    inner_command = f'"{PYTHONW}" "{os.path.join(PROJECT_ROOT, svc["script"])}"'
    log_path = os.path.join(PROJECT_ROOT, svc["log"])
    subprocess.Popen(
        [PYTHONW, WATCHDOG, inner_command, log_path],
        cwd=PROJECT_ROOT,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    return True


# =============================================================================
# HTTP HANDLERS
# =============================================================================
async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def status_handler(request: web.Request) -> web.Response:
    # run_in_executor: service_status() does blocking work (process scan,
    # a socket probe with a timeout, a file read) that would otherwise
    # stall this server's event loop on every 2s poll from the page.
    loop = asyncio.get_event_loop()
    result = {}
    for key in SERVICES:
        result[key] = await loop.run_in_executor(None, service_status, key)
    result["ui_url"] = UI_SERVER_URL
    return web.json_response(result)


async def logs_handler(request: web.Request) -> web.Response:
    service = request.query.get("service")
    if service not in SERVICES:
        return web.json_response({"ok": False, "message": "Unknown service."}, status=400)

    loop = asyncio.get_event_loop()
    log_path = os.path.join(PROJECT_ROOT, SERVICES[service]["log"])
    lines = await loop.run_in_executor(None, _read_log_tail, log_path)
    clean = [ln for ln in lines if not _is_noise(ln)][-LOG_TAIL_LINES:]
    return web.json_response({"ok": True, "lines": clean})


async def action_handler(request: web.Request) -> web.Response:
    data = await request.json()
    service = data.get("service")
    action = data.get("action")

    if service not in SERVICES:
        return web.json_response({"ok": False, "message": "Unknown service."}, status=400)

    label = SERVICES[service]["label"]
    loop = asyncio.get_event_loop()

    if action == "start":
        started = await loop.run_in_executor(None, start_service, service)
        message = f"{label} started." if started else f"{label} was already running."
    elif action == "stop":
        stopped = await loop.run_in_executor(None, stop_service, service)
        message = f"{label} stopped." if stopped else f"{label} wasn't running."
    elif action == "restart":
        await loop.run_in_executor(None, stop_service, service)
        await asyncio.sleep(1)
        await loop.run_in_executor(None, start_service, service)
        message = f"{label} restarted."
    else:
        return web.json_response({"ok": False, "message": "Unknown action."}, status=400)

    return web.json_response({"ok": True, "message": message})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/api/status", status_handler)
    app.router.add_get("/api/logs", logs_handler)
    app.router.add_post("/api/action", action_handler)
    app.router.add_static("/", STATIC_DIR, show_index=False)
    return app


async def main() -> None:
    runner = web.AppRunner(create_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", CONTROL_PANEL_PORT)

    try:
        await site.start()
    except OSError:
        # Already running elsewhere -- open a tab at the existing instance
        # instead of erroring, same "safe to click twice" promise as
        # start_citra.ps1.
        logger.info("Control panel already running -- opening a tab at the existing instance.")
        webbrowser.open(f"http://localhost:{CONTROL_PANEL_PORT}/")
        return

    logger.info("Citra Control Panel running at http://localhost:%d/", CONTROL_PANEL_PORT)
    webbrowser.open(f"http://localhost:{CONTROL_PANEL_PORT}/")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
