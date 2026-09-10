"""
CITRA DOCTOR - one command that tells you what is actually wrong.

Run it any time:   python citra_doctor.py

WHY THIS EXISTS: this system is now eleven relay boards, three AC units,
two AI backends, four model files, three network services and a config
file, spread over sixteen modules. When something does not work the
question is never "is Citra broken" - it is "WHICH of those twenty-odd
things is broken", and answering that by hand takes twenty minutes of
curl and guesswork. It took most of one evening to discover that a board
had gone missing because the laptop had been handed its IP address.

Designed to be FAST (a few seconds, everything network-bound runs
concurrently) and to be run by somebody standing in a stranger's flat
with a phone in one hand. Every failure line says what to do about it,
not just that something is red.

Exit code 0 if nothing is broken, 1 if anything is.
"""

import ast
import concurrent.futures
import glob
import io
import os
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

# Short on purpose. This is a health check, not a retry loop - a board
# that needs more than a second and a half on a LAN is a board with a
# problem worth reporting.
NET_TIMEOUT = 1.5
MAX_PARALLEL = 12

OK, WARN, BAD, INFO = "ok", "warn", "BAD", "--"
_results = []


def line(status, label, detail="", fix=""):
    _results.append((status, label, detail, fix))
    mark = {OK: "  [ ok ]", WARN: "  [warn]", BAD: "  [FAIL]", INFO: "  [    ]"}[status]
    print(f"{mark} {label}" + (f"   {detail}" if detail else ""))
    if fix:
        print(f"         -> {fix}")


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


# =============================================================================
def check_python():
    section("Python")
    v = sys.version_info
    in_venv = "jarvis_venv" in sys.executable
    line(OK if v >= (3, 9) else BAD, "version", f"{v.major}.{v.minor}.{v.micro}")
    line(OK if in_venv else WARN, "interpreter", sys.executable,
         "" if in_venv else "not the project venv - imports may differ from what Citra runs")


def check_syntax():
    section("Source files")
    bad = []
    files = sorted(glob.glob(os.path.join(ROOT, "*.py")))
    for path in files:
        try:
            ast.parse(io.open(path, encoding="utf-8").read())
        except SyntaxError as exc:
            bad.append(f"{os.path.basename(path)}:{exc.lineno}")
    if bad:
        line(BAD, "syntax", f"{len(bad)} file(s) will not parse: {', '.join(bad)}")
    else:
        line(OK, "syntax", f"all {len(files)} modules parse")


def check_imports():
    section("Imports")
    # Deliberately NOT importing jarvis_voice_assistant or jarvis_vision:
    # they load Whisper, Silero and sentence-transformers on import, which
    # is tens of seconds. Their syntax is covered above, and the models
    # they need are checked separately below.
    light = ["citra_devices", "jarvis_hardware_api", "jarvis_presence",
             "jarvis_reminders", "jarvis_router", "citra_ui_bridge"]
    sys.path.insert(0, ROOT)
    for name in light:
        try:
            __import__(name)
            line(OK, name)
        except Exception as exc:
            line(BAD, name, f"{type(exc).__name__}: {exc}")
    line(INFO, "jarvis_voice_assistant", "not imported (loads models; syntax checked above)")


def _local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if ":" not in addr:
                ips.add(addr)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips


def check_devices():
    section("Device config")
    sys.path.insert(0, ROOT)
    try:
        from citra_devices import DeviceRegistry, DeviceConfigError
    except Exception as exc:
        line(BAD, "citra_devices", str(exc))
        return None
    try:
        reg = DeviceRegistry.load()
    except Exception as exc:
        line(BAD, "citra_devices.json", str(exc),
             "copy citra_devices.example.json and fill in your boards")
        return None

    line(OK, "config", f"{len(reg.boards)} boards, {len(reg.rooms)} rooms, "
                       f"{len(reg.switches)} switches, {len(reg.air_conditioners)} ACs")

    # THE CHECK THAT COST AN EVENING. A board's static IP is invisible to
    # the router unless it is a DHCP reservation, so the router will
    # cheerfully hand that same address to a laptop. Everything then goes
    # intermittent in a way that looks exactly like bad wifi or a dying
    # board, and you will blame the hardware.
    mine = _local_ips()
    clashes = []
    for board in reg.boards:
        host = reg.host_for_board(board)
        if host in mine:
            clashes.append(f"{board} ({host})")
    for ac in reg.air_conditioners:
        if ac.host in mine:
            clashes.append(f"{ac.room} AC ({ac.host})")
    if clashes:
        line(BAD, "IP conflict", f"this machine is using: {', '.join(clashes)}",
             "that address belongs to a board. Set DHCP reservations in the router "
             "so it stops handing board addresses to other devices.")
    else:
        line(OK, "no IP conflict", f"this machine is {', '.join(sorted(mine)) or 'unknown'}")
    return reg


def _http_ok(url, timeout=NET_TIMEOUT):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, (time.perf_counter() - t0) * 1000
    except Exception:
        return None, (time.perf_counter() - t0) * 1000


def check_boards(reg):
    section("Boards on the network")
    if reg is None:
        line(INFO, "skipped", "no valid device config")
        return
    targets = [(b, reg.host_for_board(b)) for b in reg.boards]
    targets += [(f"{ac.room} AC", ac.host) for ac in reg.air_conditioners]

    # Concurrent is safe HERE because every target is a DIFFERENT board -
    # the one-request-at-a-time rule is about not stampeding a single
    # ESP8266, not about the network as a whole.
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_http_ok, f"http://{host}/"): (name, host)
                   for name, host in targets}
        for fut in concurrent.futures.as_completed(futures):
            name, host = futures[fut]
            results[name] = (host,) + fut.result()

    up = [n for n, (h, s, ms) in results.items() if s is not None]
    for name, host in targets:
        h, status, ms = results[name]
        if status is not None:
            line(OK, name, f"{host}  {ms:.0f}ms")
        else:
            line(WARN, name, f"{host}  no answer")
    if not up:
        line(BAD, "no boards reachable", f"0 of {len(targets)}",
             "check they are powered, on the wifi, and that the IPs above match reality")
    else:
        line(OK if len(up) == len(targets) else WARN, "summary",
             f"{len(up)} of {len(targets)} boards answering")


def check_models():
    section("Voice models")
    checks = [
        ("Piper voice", os.path.join(ROOT, "en_US-amy-medium.onnx"),
         "download the Piper voice into the project folder"),
        ("Piper config", os.path.join(ROOT, "en_US-amy-medium.onnx.json"), ""),
        ("wake word", os.path.join(ROOT, "hey_citra.onnx"),
         "the openWakeWord model - Citra cannot be woken without it"),
    ]
    for label, path, fix in checks:
        if os.path.exists(path):
            line(OK, label, f"{os.path.getsize(path)/1e6:.1f} MB")
        else:
            line(BAD, label, "missing", fix)

    hf = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hf):
        models = [d for d in os.listdir(hf) if d.startswith("models--")]
        line(OK, "HuggingFace cache", f"{len(models)} model(s) cached")
    else:
        line(WARN, "HuggingFace cache", "empty",
             "first run will download Whisper - slow, and needs internet")


def check_ai_backends():
    section("AI backends")
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        line(OK, "GEMINI_API_KEY", f"set ({len(key)} chars)")
        status, ms = _http_ok(
            "https://generativelanguage.googleapis.com/v1beta/models?key=" + key, timeout=6)
        if status == 200:
            line(OK, "Gemini reachable", f"{ms:.0f}ms")
        else:
            line(WARN, "Gemini", f"no valid response ({ms:.0f}ms)",
                 "key may be wrong, or there is no internet")
    else:
        line(WARN, "GEMINI_API_KEY", "not set",
             'setx GEMINI_API_KEY "..." then open a NEW terminal')

    status, ms = _http_ok("http://localhost:1234/v1/models", timeout=3)
    if status == 200:
        line(OK, "LM Studio", f"serving on :1234  {ms:.0f}ms")
    else:
        line(WARN, "LM Studio", "not serving on :1234",
             "start it if you want the local backend; Citra falls back to Gemini otherwise")

    if not key and status != 200:
        line(BAD, "no AI backend", "neither Gemini nor LM Studio is available",
             "Citra can still switch lights, but cannot answer anything")


def check_services():
    section("Services and ports")
    for port, what in [(8765, "UI server"), (8766, "voice ingest")]:
        s = socket.socket()
        s.settimeout(0.4)
        busy = s.connect_ex(("127.0.0.1", port)) == 0
        s.close()
        line(OK if busy else INFO, f"port {port} ({what})",
             "in use" if busy else "free (not running)")

    cert = os.path.join(ROOT, "citra_cert.pem")
    if os.path.exists(cert):
        line(WARN, "TLS cert", "present - server will serve HTTPS",
             "iOS Shortcuts REJECTS self-signed certs. Rename the .pem files if you "
             "want phone shortcuts to work over Tailscale.")
    else:
        line(OK, "TLS cert", "absent - server will serve plain HTTP")

    token = os.path.join(ROOT, "citra_api_token.txt")
    line(OK if os.path.exists(token) else INFO, "phone API token",
         "present" if os.path.exists(token) else "not yet generated (created on first server start)")


def check_scheduled_tasks():
    section("Windows scheduled tasks")
    try:
        out = subprocess.run(["schtasks", "/query", "/fo", "list"],
                             capture_output=True, text=True, timeout=20).stdout or ""
    except Exception:
        line(INFO, "schtasks", "could not query")
        return
    found = [l.split(":", 1)[1].strip() for l in out.splitlines()
             if l.startswith("TaskName:") and "citra" in l.lower()]

    # A scheduled CALL is a feature, not something that crept in. Warning
    # about it - and offering to delete it - is how a 6am wake-up call
    # quietly disappears the day somebody tidies up after a doctor run.
    intentional = [t for t in found if "wake-up" in t.lower() or "call" in t.lower()]
    other = [t for t in found if t not in intentional]

    for task in intentional:
        detail = task
        try:
            info = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-ScheduledTaskInfo -TaskName '{task.lstrip(chr(92))}')"
                 f".NextRunTime"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            if info:
                detail = f"{task} - next run {info}"
        except Exception:
            pass
        line(OK, "scheduled call", detail,
             "runs on its own; needs the laptop awake and UNLOCKED")

    if other:
        line(WARN, "autostart tasks", ", ".join(other),
             'Unregister-ScheduledTask -TaskName "..." -Confirm:$false  (elevated PowerShell)')
    elif not found:
        line(OK, "autostart tasks", "none registered")


# =============================================================================
def check_calling():
    """
    Everything that has to be true before Citra can ring somebody.

    Worth its own section because the failures here are all SILENT ones:
    a missing audio cable means she talks into the room instead of the
    call, an empty phonebook means every request is refused, and quiet
    hours switched off means nothing stops a 3am dial. None of those
    raise; they just behave wrongly at the moment it matters.
    """
    section("Calling")

    # -- the audio path ------------------------------------------------
    try:
        import citra_live_call as lc

        cable = lc._find_cable_output()
        if cable:
            line(OK, "audio route", f"cable - {cable[1]}",
                 "her voice never touches the speakers or the microphone")
        else:
            line(WARN, "audio route", "speakers + microphone (no virtual cable)",
                 "install VB-CABLE so the call doesn't carry room noise")

        speaker = lc._cable_speaker()
        if cable and speaker is None:
            line(BAD, "cable endpoint", "found by index but not by name",
                 "PortAudio may be bound to a dead VB-CABLE instance")
    except Exception as exc:
        line(BAD, "audio route", f"couldn't check: {exc}")

    # -- the phonebook -------------------------------------------------
    try:
        import citra_contacts

        people = citra_contacts.book().all()
        if people:
            line(OK, "phonebook", f"{len(people)} saved: " +
                 ", ".join(c.name for c in people[:6]))
        else:
            line(WARN, "phonebook", "empty",
                 "add contacts at /calls, or Citra refuses every call by name")
    except Exception as exc:
        line(BAD, "phonebook", f"couldn't load: {exc}")

    # -- quiet hours ---------------------------------------------------
    try:
        import citra_quiet_hours

        start, end, enabled = citra_quiet_hours.load_config()
        window = f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
        verdict = citra_quiet_hours.check()
        if not enabled:
            line(WARN, "quiet hours", "switched OFF",
                 "nothing stops Citra ringing somebody at 3am")
        else:
            line(OK, "quiet hours", f"{window} - " +
                 ("calls allowed now" if verdict.allowed else "calls blocked now"))
    except Exception as exc:
        line(BAD, "quiet hours", f"couldn't check: {exc}")

    # -- where recordings go -------------------------------------------
    try:
        import citra_live_call as lc

        folder = lc.RECORDINGS_DIR
        exists = os.path.isdir(folder)
        writable = exists and os.access(folder, os.W_OK)
        count = len([f for f in os.listdir(folder)
                     if f.lower().endswith(".wav")]) if exists else 0
        if writable:
            line(OK, "recordings", f"{count} saved in {folder}")
        else:
            line(BAD, "recordings", f"{folder} is not writable",
                 "calls will happen but nothing will be kept")
    except Exception as exc:
        line(BAD, "recordings", f"couldn't check: {exc}")

    # -- is WhatsApp reachable at all ----------------------------------
    try:
        import citra_whatsapp as wa

        uia, UIA = wa._uia()
        window = wa._whatsapp_window(uia, UIA, timeout=4)
        if window is None:
            line(WARN, "WhatsApp", "not running",
                 "Citra will launch it, which adds ~25s to the first call")
        else:
            names = {(b.CurrentName or "").strip().lower()
                     for b in wa._walk(uia, UIA, window, wa.BUTTON, max_depth=28)}
            if "calls" in names:
                line(OK, "WhatsApp", "running, Calls tab reachable")
            else:
                line(BAD, "WhatsApp", "a window exists but the UI isn't there",
                     "it may be minimised to tray or still loading")
    except Exception as exc:
        line(WARN, "WhatsApp", f"couldn't inspect: {exc}")


def main():
    t0 = time.perf_counter()
    print("=" * 62)
    print("  CITRA DOCTOR")
    print("=" * 62)

    check_python()
    check_syntax()
    check_imports()
    reg = check_devices()
    check_boards(reg)
    check_models()
    check_ai_backends()
    check_services()
    check_calling()
    check_scheduled_tasks()

    bad = [r for r in _results if r[0] == BAD]
    warn = [r for r in _results if r[0] == WARN]
    print("\n" + "=" * 62)
    print(f"  {len(bad)} broken, {len(warn)} worth a look, "
          f"checked in {time.perf_counter() - t0:.1f}s")
    if bad:
        print("\n  BROKEN:")
        for _, label, detail, fix in bad:
            print(f"    - {label}: {detail}")
            if fix:
                print(f"      {fix}")
    print("=" * 62)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
