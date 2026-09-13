"""
=============================================================================
CITRA UI SERVER
=============================================================================
Local web server for the fullscreen Citra control UI (citra_ui/index.html).
Runs as its OWN process, separate from jarvis_voice_assistant.py — see
citra_ui_bridge.py's module docstring for why that separation exists.

RESPONSIBILITIES:
  1. Serve the static frontend (citra_ui/).
  2. Accept WebSocket connections from browser tabs and push them live
     updates: what Citra is currently saying (captions), what she just
     heard (transcript), her current AssistantState, and the last-known
     hardware state — all fed in from citra_ui_bridge's POSTs below.
  3. Accept button/slider actions FROM the browser over that same
     WebSocket and dispatch them to a SmartRoomController instance,
     broadcasting the resulting hardware state back to every connected
     tab (not just the one that clicked) so multiple open windows stay
     in sync.
  4. Expose /caption, /state, /transcript POST endpoints — this is the
     other half of citra_ui_bridge.py's fire-and-forget notifications
     from the main voice assistant process.

RUN THIS ALONGSIDE jarvis_voice_assistant.py, not instead of it:
    python citra_ui_server.py
Then open http://localhost:8765 in a browser. The voice assistant works
identically with or without this running — it's a supplementary control
surface, never a dependency (see citra_ui_bridge.py for the fire-and-
forget design that guarantees this).

LOCAL-ONLY BY DESIGN: this server binds 127.0.0.1 (loopback only) — not
reachable from your phone or any other device on the WiFi, only from this
machine. It was briefly made LAN-reachable (0.0.0.0 + a PIN login gate,
for controlling Citra from a phone) but that was dropped by request in
favor of keeping everything local for now; see git history (the commit
adding auth_middleware/CITRA_PIN, and the one after it removing them) if
that's ever revisited — the login page, session cookies, and brute-force
lockout were all working and tested, just not currently wired in.
=============================================================================
"""
import asyncio
import concurrent.futures
import hmac
import json
import logging
import os
import secrets
import ssl
import subprocess
import time

import aiohttp
from aiohttp import WSMsgType, web

import citra_mute
from jarvis_hardware_api import SmartRoomController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("citra_ui_server")

UI_SERVER_PORT = 8765
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citra_ui")
TLS_CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citra_cert.pem")
TLS_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citra_key.pem")

# jarvis_voice_assistant.py's own audio-ingest listener — a separate
# process from this one (see citra_ui_bridge.py's docstring for why), so
# even though both are loopback-only now, this server still has to reach
# it over HTTP rather than a direct function call. See
# jarvis_voice_assistant.py's _AudioIngestHandler for the other end of
# this.
VOICE_INGEST_URL = "http://127.0.0.1:8766/ingest"

AC_TEMP_MIN = 17
AC_TEMP_MAX = 30

# =============================================================================
# REMOTE CONTROL API - the /api/* routes, for iOS Shortcuts and Android
# =============================================================================
# WHY THESE EXIST SEPARATELY FROM THE WEBSOCKET the dashboard uses:
# Shortcuts (and Android's equivalents) speak one-shot HTTP. They cannot
# open a WebSocket, wait for a hello frame, send JSON and read a reply -
# and even if they could, that handshake is several round trips of
# latency on a mobile connection for what is meant to feel instant. These
# routes are deliberately plain GETs with the whole command in the path,
# so a shortcut is ONE "Get Contents of URL" action with nothing to
# configure but a header.
#
# GET rather than POST for a state-changing call is a real trade. It is
# made here because Shortcuts' GET is markedly fewer taps to set up, the
# token header means no drive-by request can trigger anything, and the
# clients involved (Shortcuts, curl) do not prefetch or preload URLs the
# way a browser or a chat-app link preview would. Do not put one of these
# URLs anywhere a link unfurler can see it.
BIND_HOST = os.environ.get("CITRA_BIND_HOST", "127.0.0.1")
API_TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citra_api_token.txt")

# A SECOND controller, tuned to fail fast, used only by the /api/* routes.
#
# The shared `controller` below is built for the dashboard: timeout=3.0
# with max_retries=2, so urllib3 makes three attempts with 0.5s and 1.0s
# of backoff between them. Against an offline board that is a measured
# 10.0 SECONDS before the call gives up - the retries multiply the
# timeout by more than three, which the "a short timeout means a dead
# board fails FAST" comment in jarvis_hardware_api.py does not account
# for.
#
# For a browser tab already sitting on the dashboard, waiting 10s and
# eventually succeeding is better than failing at 3s. For a phone on
# mobile data it is exactly backwards: the user tapped a shortcut and is
# standing in a dark room. Failing in 1.5s with a clear error beats
# succeeding in 10, so the phone routes get their own controller with
# retries off. Same board, same session pooling, different patience.
PHONE_TIMEOUT_SECONDS = 1.5

# Relays 1-4 are "the lights" as a group - the same set the voice
# assistant's whole-room lighting protocol drives when you do not name a
# specific light. Kept consistent deliberately: "Citra, turn on the
# lights" and the phone shortcut should do the identical thing.
ALL_RELAY_NUMBERS = (1, 2, 3, 4)

# One controller instance for the whole server's lifetime — same reasoning
# as JarvisRouter's own controller: reuse the requests.Session inside it
# rather than reconnecting to the NodeMCUs on every button press.
controller = SmartRoomController()
phone_controller = SmartRoomController(timeout=PHONE_TIMEOUT_SECONDS, max_retries=0)

# Connected browser tabs. A plain set is fine here — this server has a
# single event loop and no multi-worker concerns; every access happens
# from that one loop, so no lock is needed around it.
connected_clients: "set[web.WebSocketResponse]" = set()

# Last-known AC temperature, tracked locally so /ac/temp/up and
# /ac/temp/down (which the firmware has no dedicated endpoint for) can
# compute a clamped delta without a round-trip to the AC board first.
# Seeded from a real get_ac_status() call at server startup.
_last_known_ac_temp = 24


async def broadcast(message: dict) -> None:
    """Send a JSON message to every currently-connected browser tab."""
    if not connected_clients:
        return
    payload = json.dumps(message)
    dead_clients = []
    for ws in connected_clients:
        try:
            await ws.send_str(payload)
        except ConnectionResetError:
            dead_clients.append(ws)
    for ws in dead_clients:
        connected_clients.discard(ws)


# CIRCUIT BREAKER for boards that are switched off.
#
# Every relay board is polled over HTTP with a connect timeout. With the
# boards unpowered - which is most of the time while the flat is still
# being wired - each poll costs the full timeout, and there are fourteen
# of them. That sweep runs on every dashboard connect and after every
# button press, and it is what made a typed message take eleven seconds
# to come back while the routing itself took one.
#
# So after a sweep that reaches nothing, stop sweeping for a while. The
# dashboard shows the last known state instead of freezing, and the
# moment a board answers again the breaker closes on its own.
_BOARDS_DOWN_UNTIL = 0.0
BOARDS_DOWN_BACKOFF_SECONDS = 60.0


def _boards_are_down() -> bool:
    return time.monotonic() < _BOARDS_DOWN_UNTIL


def _note_board_sweep(reachable: bool) -> None:
    global _BOARDS_DOWN_UNTIL
    if reachable:
        if _BOARDS_DOWN_UNTIL:
            logger.info("a board answered again - resuming normal polling")
        _BOARDS_DOWN_UNTIL = 0.0
    else:
        _BOARDS_DOWN_UNTIL = time.monotonic() + BOARDS_DOWN_BACKOFF_SECONDS
        logger.info("no board answered - backing off polling for %.0fs",
                    BOARDS_DOWN_BACKOFF_SECONDS)


def _relay_status_dict() -> dict:
    if _boards_are_down():
        return {}
    result = controller.get_all_relay_status()
    if not result.success or not result.data:
        return {}
    return result.data


def _ac_status_dict() -> dict:
    if _boards_are_down():
        return {}
    result = controller.get_ac_status()
    if not result.success or not result.data:
        return {}
    return result.data


# Guards against stampeding the NodeMCU boards. An ESP8266 running the
# Arduino WebServer library handles essentially one request at a time on
# very limited RAM — it is not a server you can hit concurrently and
# expect to survive. Two things here were doing exactly that:
#   - every websocket_handler connect calls broadcast_hw_state(), and
#   - each call fans out to BOTH boards at once via asyncio.gather.
# So opening three tabs (or one page that reconnects a couple of times)
# fired ~6 near-simultaneous requests within a second, and the boards
# stopped responding — observed live as connect-timeouts and read-timeouts
# from both boards immediately after a third client connected, having been
# perfectly healthy a moment earlier. The boards were fine; we were
# flooding them.
#
# _hw_state_lock serializes polls so only one is ever in flight, and
# _hw_state_cache makes any poll within HW_STATE_CACHE_SECONDS reuse the
# result instead of issuing a fresh one. Together these turn N clients
# connecting at once into ONE round of hardware requests.
HW_STATE_CACHE_SECONDS = 2.0  # short enough that the UI still feels live
                              # after an action (and every action pushes
                              # its own fresh poll anyway), long enough to
                              # absorb a burst of reconnects.
_hw_state_lock = asyncio.Lock()
_hw_state_cache: dict | None = None
_hw_state_cache_time = 0.0


async def _fetch_hw_state(force: bool) -> dict:
    """Returns current hardware state, coalescing and caching to protect
    the boards. See _hw_state_lock's comment for why that matters."""
    global _hw_state_cache, _hw_state_cache_time, _last_known_ac_temp

    async with _hw_state_lock:
        now = time.monotonic()
        if (
            not force
            and _hw_state_cache is not None
            and now - _hw_state_cache_time < HW_STATE_CACHE_SECONDS
        ):
            return _hw_state_cache

        loop = asyncio.get_event_loop()
        # controller.* methods are synchronous (blocking) HTTP calls — run
        # them off the event loop thread so one slow/unreachable board
        # can't stall every other WebSocket message this server is trying
        # to process. Note these two still run concurrently with EACH
        # OTHER, which is fine: they're different boards. The lock above
        # is what prevents multiple *rounds* from overlapping.
        relays, ac = await asyncio.gather(
            loop.run_in_executor(None, _relay_status_dict),
            loop.run_in_executor(None, _ac_status_dict),
        )
        # If a whole sweep reached nothing, stop sweeping for a minute.
        if not _boards_are_down():
            _note_board_sweep(bool(relays) or bool(ac))
        if ac.get("temp") is not None:
            _last_known_ac_temp = ac["temp"]

        _hw_state_cache = {"type": "hw_state", "relays": relays, "ac": ac}
        _hw_state_cache_time = time.monotonic()
        return _hw_state_cache


async def broadcast_hw_state(force: bool = False) -> None:
    """Fetch hardware state and push it to every connected tab.

    `force=True` bypasses the cache — used after an action, where the
    whole point is to observe the change that action just made.
    """
    await broadcast(await _fetch_hw_state(force))


# =============================================================================
# ACTION DISPATCH — one entry per button/slider the frontend can press
# =============================================================================
async def handle_action(message: dict) -> None:
    """
    Dispatches one browser-side action to SmartRoomController and, if the
    underlying hardware call failed (unreachable board, or — as happened
    in practice — an endpoint the physical board's CURRENT firmware
    doesn't have yet because it hasn't been re-flashed since these
    routes were added), broadcasts an explicit error the UI can surface.
    Earlier versions of this handler let a failed call disappear
    silently: broadcast_hw_state() would just re-report the unchanged
    state, which looked indistinguishable from the button doing nothing
    at all. Every action now returns its result so failures are visible,
    not silently swallowed.
    """
    loop = asyncio.get_event_loop()
    action = message.get("action")

    def run(fn, *args):
        return loop.run_in_executor(None, fn, *args)

    result = None  # a HardwareResult, when the action dispatches one

    if action == "chat":
        # Logged on purpose. A successful chat used to leave no trace at
        # all, so when the box looked dead there was no way to tell
        # whether the message reached the server or never left the page -
        # two completely different bugs that look identical from the
        # chair. One line each way settles it.
        logger.info("CHAT IN : %r", str(message.get("text", ""))[:120])
        # Answered on a background task, NOT awaited here. handle_action
        # runs inside the socket's receive loop, so awaiting a reply that
        # can take a second or two froze every other control on the
        # dashboard for the duration - the lights stopped responding
        # while Citra was thinking about a question.
        text = str(message.get("text", ""))
        async def _reply() -> None:
            payload = await _answer(text)
            logger.info("CHAT OUT: ok=%s %r", payload.get("ok"),
                        str(payload.get("reply"))[:120])
            await broadcast({"type": "chat_reply", **payload})

        asyncio.create_task(_reply())
        return

    if action == "open_console":
        try:
            opened = _open_log_console(str(message.get("log", "assistant")))
            await broadcast({"type": "toast", "message": f"Opened {opened}"})
        except Exception as exc:
            await broadcast({"type": "toast", "message": f"Couldn't open console: {exc}"})
        return

    if action == "get_status":
        pass  # handled by the broadcast_hw_state() call below regardless

    elif action == "relay_set":
        relay = int(message["relay"])
        state = bool(message["state"])
        result = await run(controller.turn_on_relay if state else controller.turn_off_relay, relay)

    elif action == "brightness_up":
        result = await run(controller.increase_brightness)
    elif action == "brightness_down":
        result = await run(controller.decrease_brightness)
    elif action == "brightness_max":
        result = await run(controller.set_max_brightness)
    elif action == "brightness_min":
        result = await run(controller.set_min_brightness)

    elif action == "ac_power":
        result = await run(controller.set_ac_power, bool(message["state"]))

    elif action == "ac_temp_set":
        global _last_known_ac_temp
        value = max(AC_TEMP_MIN, min(AC_TEMP_MAX, int(message["value"])))
        result = await run(controller.set_ac_temperature, value)
        _last_known_ac_temp = value

    elif action == "ac_temp_up":
        _last_known_ac_temp = max(AC_TEMP_MIN, min(AC_TEMP_MAX, _last_known_ac_temp + 1))
        result = await run(controller.set_ac_temperature, _last_known_ac_temp)

    elif action == "ac_temp_down":
        _last_known_ac_temp = max(AC_TEMP_MIN, min(AC_TEMP_MAX, _last_known_ac_temp - 1))
        result = await run(controller.set_ac_temperature, _last_known_ac_temp)

    elif action == "ac_mode":
        result = await run(controller.set_ac_mode, str(message["value"]))

    elif action == "ac_fan_set":
        result = await run(controller.set_ac_fan_speed, str(message["value"]))
    elif action == "ac_fan_up":
        result = await run(controller.increase_ac_fan_speed)
    elif action == "ac_fan_down":
        result = await run(controller.decrease_ac_fan_speed)

    else:
        logger.warning("Unknown UI action: %r", action)
        return

    if result is not None and not result.success:
        await broadcast({"type": "error", "message": result.message})

    # force=True: an action just changed the hardware, so a cached
    # pre-action reading would show the UI the old state and look like the
    # button did nothing.
    #
    # Backgrounded for the same reason as the connect-time poll: this
    # runs inside the socket's receive loop, so awaiting a full board
    # sweep after every button press froze the whole dashboard - and
    # every press pays it, not just the first.
    asyncio.create_task(broadcast_hw_state(force=True))


# =============================================================================
# HTTP HANDLERS
# =============================================================================
async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    connected_clients.add(ws)
    logger.info("UI client connected (%d total)", len(connected_clients))

    try:
        # NOT awaited. This polls every relay board, and with the boards
        # powered off each one sits in a 2-second connect timeout - about
        # eleven seconds for fourteen of them. Awaiting it here meant the
        # socket accepted nothing until that finished, so the first thing
        # typed into the chat took eleven seconds to come back and the
        # box looked broken. The state arrives when it arrives; the
        # socket is usable immediately.
        asyncio.create_task(broadcast_hw_state())
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                try:
                    await handle_action(data)
                except Exception as exc:
                    logger.error("Action %r failed: %s", data, exc)
            elif msg.type == WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())
    finally:
        connected_clients.discard(ws)
        logger.info("UI client disconnected (%d remaining)", len(connected_clients))

    return ws


async def caption_handler(request: web.Request) -> web.Response:
    data = await request.json()
    await broadcast({"type": "caption", "text": data.get("text", "")})
    return web.json_response({"ok": True})


async def state_handler(request: web.Request) -> web.Response:
    data = await request.json()
    await broadcast({"type": "assistant_state", "state": data.get("state", "")})
    return web.json_response({"ok": True})


async def transcript_handler(request: web.Request) -> web.Response:
    data = await request.json()
    await broadcast({"type": "transcript", "text": data.get("text", "")})
    return web.json_response({"ok": True})


async def voice_handler(request: web.Request) -> web.Response:
    """
    Receives a recorded audio blob from a browser's microphone (see
    citra_ui/app.js's mic button) and forwards the raw bytes to
    jarvis_voice_assistant.py's own local ingest endpoint, which decodes
    and routes it through the EXACT SAME pipeline the physical microphone
    uses (transcription, language filtering, hallucination filtering,
    semantic routing, Smart Path fallback, spoken response) — see
    VOICE_INGEST_URL's comment for why that forwarding happens instead of
    this process owning any of that logic itself.

    This handler's own job is narrow on purpose: pass the bytes through,
    translate the ingest endpoint's response into something the browser
    can react to, and turn "the assistant isn't reachable at all" into a
    clear error rather than a mysterious hang — everything about what the
    audio actually MEANS is the voice assistant's business, not this
    server's.
    """
    content_type = request.headers.get("Content-Type", "application/octet-stream")
    audio_bytes = await request.read()

    if not audio_bytes:
        return web.json_response({"success": False, "message": "No audio received."}, status=400)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                VOICE_INGEST_URL,
                data=audio_bytes,
                headers={"Content-Type": content_type},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as ingest_response:
                body = await ingest_response.json()
                return web.json_response(body, status=ingest_response.status)
    except aiohttp.ClientConnectorError:
        logger.error("Could not reach the voice assistant's ingest endpoint at %s", VOICE_INGEST_URL)
        return web.json_response(
            {"success": False, "message": "Citra's voice assistant isn't running right now."},
            status=503,
        )
    except TimeoutError:
        return web.json_response(
            {"success": False, "message": "Timed out waiting for the voice assistant."},
            status=504,
        )


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "")


def _load_or_create_api_token() -> str:
    """
    Reads the shared secret from citra_api_token.txt, creating it with a
    fresh random value the first time.

    A file rather than an environment variable because the token has to
    survive reboots and be readable by a human copying it into Shortcuts
    on their phone. secrets.token_urlsafe(32) is 256 bits of entropy,
    URL-safe so it can also be passed as ?token= when a client cannot set
    headers.
    """
    if os.path.exists(API_TOKEN_PATH):
        token = open(API_TOKEN_PATH, encoding="utf-8").read().strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    with open(API_TOKEN_PATH, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    logger.info("Generated a new API token at %s", API_TOKEN_PATH)
    return token


API_TOKEN = _load_or_create_api_token()


@web.middleware
async def no_cache_middleware(request: web.Request, handler):
    """
    Stop the browser serving a stale dashboard.

    THE PROBLEM THIS SOLVES: app.js was fixed, the server was serving the
    fixed file, and the page in the browser kept running the old one -
    so a bug that was genuinely repaired still looked broken, and the
    only remedy was knowing to press Ctrl+Shift+R. That is not a thing a
    user should have to know, and it makes every future fix land
    unreliably.

    The dashboard is a handful of small files served from loopback, so
    there is nothing to gain from caching them and a lot to lose.
    """
    response = await handler(request)
    try:
        path = request.path.lower()
        if path.endswith((".js", ".css", ".html")) or path in ("/", "/calls"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
    except Exception:
        pass
    return response


@web.middleware
async def api_token_middleware(request: web.Request, handler):
    """
    Gates every /api/* route behind the shared secret.

    Only /api/* is gated. The dashboard itself, the WebSocket and the
    voice-ingest routes stay open because they are reachable only from
    loopback - see the bind check in __main__, which refuses to start at
    all if that stops being true. That check is what makes this narrow
    gate sufficient rather than negligent.

    hmac.compare_digest rather than ==: token comparison with a plain
    equality operator short-circuits on the first differing byte, which
    leaks the length of the matching prefix through timing. Over a LAN
    that is a real signal, and the fix costs nothing.
    """
    if not request.path.startswith("/api/"):
        return await handler(request)

    presented = request.headers.get("X-Citra-Token") or request.query.get("token") or ""
    if not hmac.compare_digest(presented, API_TOKEN):
        logger.warning("Rejected /api call from %s - bad or missing token", request.remote)
        return web.json_response({"ok": False, "error": "unauthorized"}, status=401)
    return await handler(request)


_api_refresh_pending = False


def _schedule_dashboard_refresh() -> None:
    """
    Refreshes the browser dashboard after a phone action - but only if
    there is a browser to refresh, and never more than one at a time.

    BOTH GUARDS ARE LOAD-BEARING, and this function exists because the
    naive version (asyncio.create_task(broadcast_hw_state(force=True))
    straight from every API call) took the server down.

    broadcast_hw_state serializes behind _hw_state_lock, on purpose, so
    the ESP8266s are never hit concurrently. Each poll reads both boards.
    With a board offline that poll runs for ~10 seconds holding the lock
    and two executor threads. Firing one per API request queues them:
    six shortcut taps became a minute-long backlog, and requests that
    touched no hardware at all - even GET / - started timing out behind
    it. Measured, not theorized.

    The `connected_clients` check is the one that matters in practice.
    When someone taps a shortcut from their phone there is usually no
    dashboard open anywhere, so the entire poll is work nobody will ever
    see. Skipping it is both the fix and the right behaviour.
    """
    global _api_refresh_pending
    if not connected_clients or _api_refresh_pending:
        return
    _api_refresh_pending = True

    async def _run():
        global _api_refresh_pending
        try:
            await broadcast_hw_state(force=True)
        finally:
            _api_refresh_pending = False

    asyncio.create_task(_run())


async def _run_hw(fn, *args):
    """
    Runs one blocking SmartRoomController call off the event loop, then
    schedules the dashboard refresh WITHOUT waiting for it.

    That ordering is the whole reason these routes feel instant. The
    WebSocket path calls broadcast_hw_state(force=True) inline, which
    re-reads relay AND ac state over HTTP before anything returns - two
    extra round trips to the board that the phone would otherwise sit
    through. Here the response goes back the moment the board has
    acknowledged the command, and the browser dashboard catches up a
    few tens of milliseconds later on its own.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, fn, *args)
    _schedule_dashboard_refresh()
    return result


def _ok(did: str, result=None) -> web.Response:
    if result is not None and not result.success:
        return web.json_response({"ok": False, "did": did, "error": result.message}, status=502)
    return web.json_response({"ok": True, "did": did})


async def api_lights(request: web.Request) -> web.Response:
    state = request.match_info["state"].lower()
    if state not in ("on", "off"):
        return web.json_response({"ok": False, "error": "state must be on or off"}, status=400)

    turn_on = state == "on"

    def _all():
        # Sequential, not asyncio.gather: SmartRoomController wraps a
        # single requests.Session, which is not thread-safe, and the
        # ESP8266's own web server handles one connection at a time
        # anyway. Four calls at ~30ms each is still well inside a blink.
        fn = phone_controller.turn_on_relay if turn_on else phone_controller.turn_off_relay
        results = [fn(n) for n in ALL_RELAY_NUMBERS]
        return next((r for r in results if not r.success), results[-1])

    result = await _run_hw(_all)
    return _ok("lights " + state, result)


async def api_light(request: web.Request) -> web.Response:
    try:
        number = int(request.match_info["number"])
    except ValueError:
        return web.json_response({"ok": False, "error": "bad light number"}, status=400)
    if number not in ALL_RELAY_NUMBERS:
        return web.json_response({"ok": False, "error": "light must be 1-4"}, status=400)

    state = request.match_info["state"].lower()
    if state not in ("on", "off"):
        return web.json_response({"ok": False, "error": "state must be on or off"}, status=400)

    fn = phone_controller.turn_on_relay if state == "on" else phone_controller.turn_off_relay
    result = await _run_hw(fn, number)
    return _ok("light %d %s" % (number, state), result)


async def api_ac(request: web.Request) -> web.Response:
    state = request.match_info["state"].lower()
    if state not in ("on", "off"):
        return web.json_response({"ok": False, "error": "state must be on or off"}, status=400)
    result = await _run_hw(phone_controller.set_ac_power, state == "on")
    return _ok("ac " + state, result)


async def api_ac_temp(request: web.Request) -> web.Response:
    global _last_known_ac_temp
    try:
        value = int(request.match_info["value"])
    except ValueError:
        return web.json_response({"ok": False, "error": "bad temperature"}, status=400)
    if not (AC_TEMP_MIN <= value <= AC_TEMP_MAX):
        return web.json_response(
            {"ok": False, "error": "temperature must be %d-%d" % (AC_TEMP_MIN, AC_TEMP_MAX)},
            status=400,
        )
    result = await _run_hw(phone_controller.set_ac_temperature, value)
    _last_known_ac_temp = value
    return _ok("ac %d degrees" % value, result)


async def api_everything_off(request: web.Request) -> web.Response:
    def _all_off():
        results = [phone_controller.turn_off_relay(n) for n in ALL_RELAY_NUMBERS]
        results.append(phone_controller.set_ac_power(False))
        return next((r for r in results if not r.success), results[-1])

    result = await _run_hw(_all_off)
    return _ok("everything off", result)


# =============================================================================
# TEXT CHAT
# =============================================================================
# Same brain, typed instead of spoken. This exists because voice is not
# always the right input: you are in a lecture, someone is asleep, the
# command has a word Whisper keeps mangling, or you simply want the answer
# on screen where you can re-read it.
#
# It routes through the SAME JarvisRouter the voice path uses, so a typed
# "turn on the bedroom lights" does exactly what the spoken one does -
# fast-path regex first, then Gemini with the full tool schema. Nothing
# here is a parallel implementation that could drift.
#
# Its own router instance, not the assistant's: this server runs in a
# separate process (see citra_ui_bridge's docstring), and reaching across
# would mean an IPC hop on every message. The practical consequence is a
# separate conversation memory for typing vs speaking, which is arguably
# what you want anyway - a typed thread and a spoken one are different
# conversations.
#
# Constructed LAZILY on first use. Building it eagerly would add the tool
# schema construction to this server's startup, and the dashboard should
# come up instantly whether or not anyone ever types anything.
_text_router = None
_text_router_lock = asyncio.Lock()


# Chat gets its OWN threads, separate from everything else.
#
# THE BUG THIS FIXES: _answer used to run on the default executor - the
# same pool every hardware call uses. With the relay boards powered off,
# each board poll sits in a 2-second connect timeout holding a thread,
# and there are fourteen boards. A typed message then queued behind them
# and took 11.3 seconds to come back, measured. From the chat box that
# is indistinguishable from broken: you type, you get "thinking...", and
# nothing happens for long enough that you give up.
#
# Two threads is enough - the router serialises on the model anyway - and
# the point is not throughput, it is that a dead board on the WiFi can
# never again make Citra look like she has stopped listening.
_CHAT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="citra-chat")


def _build_text_router():
    """Construct the router. BLOCKING - must not run on the event loop."""
    from jarvis_router import JarvisRouter

    logger.info("Building the text-chat router...")
    started = time.perf_counter()
    router = JarvisRouter(controller=phone_controller)
    logger.info("Text-chat router ready in %.1fs",
                time.perf_counter() - started)
    return router


async def _get_text_router():
    """
    The chat router, built OFF the event loop and pre-warmed at startup.

    MEASURED: the first typed message took 11.3 seconds end to end while
    the routing itself took 1.2 - ten seconds of it was constructing
    JarvisRouter, synchronously, inside the event loop. That froze the
    entire server: no other socket message, no dashboard update, nothing.
    From the chat box it looked exactly like Citra had stopped
    responding, which is precisely what it was reported as.

    Two changes: the build now happens in a thread so it can never block
    the loop again, and prewarm_text_router() runs it at startup so the
    first person to type has already had it done for them.
    """
    global _text_router
    async with _text_router_lock:
        if _text_router is None:
            loop = asyncio.get_event_loop()
            _text_router = await loop.run_in_executor(
                _CHAT_EXECUTOR, _build_text_router)
    return _text_router


async def prewarm_text_router(app=None) -> None:
    """Build the router at startup so nobody waits for it mid-conversation."""
    try:
        await _get_text_router()
    except Exception:
        logger.exception("couldn't pre-warm the chat router; the first "
                         "message will build it instead")


async def _unused_get_text_router():
    global _text_router
    async with _text_router_lock:
        if _text_router is None:
            from jarvis_router import JarvisRouter
            logger.info("Building the text-chat router (first message)...")
            # phone_controller, NOT controller: someone typing is waiting
            # on the answer. Measured with the boards offline, the patient
            # dashboard controller took 34.5 SECONDS to answer "turn on the
            # lights" (3s timeout x 3 attempts x 4 relays) before reporting
            # what it already knew after the first one. Fast-fail gets the
            # same answer in about six.
            _text_router = JarvisRouter(controller=phone_controller)
    return _text_router


del _unused_get_text_router





async def _answer(text: str) -> dict:
    """Route one typed message and return a reply payload."""
    text = (text or "").strip()
    if not text:
        return {"ok": False, "reply": "Say something and I'll answer."}
    router = await _get_text_router()
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    result = await loop.run_in_executor(_CHAT_EXECUTOR, router.route, text)
    elapsed = (time.perf_counter() - t0) * 1000
    # A failed route still has a message worth showing - it says WHY.
    return {"ok": bool(result.success), "reply": result.message,
            "path": result.path, "ms": round(elapsed)}


async def api_ask(request: web.Request) -> web.Response:
    """
    Ask by URL, for Shortcuts: /api/ask?q=is+the+geyser+on

    Token-gated like every other /api route, so this is safe to reach
    from the phone over the tailnet.
    """
    payload = await _answer(request.query.get("q", ""))
    return web.json_response(payload)


# =============================================================================
# CONSOLE
# =============================================================================
def _open_log_console(log_name: str) -> str:
    """
    Opens a real terminal window on THIS machine tailing one log file.

    Deliberately not a shell string with anything user-supplied in it:
    log_name is matched against a fixed whitelist below and the path is
    built here. A dashboard button that could put arbitrary text on a
    PowerShell command line would be a remote code execution hole on the
    machine that controls the flat's mains wiring.

    Reachable only over the dashboard WebSocket, which binds loopback -
    NOT exposed on the token API, because a terminal window opening on a
    laptop is of no use to someone holding a phone anyway.
    """
    allowed = {
        "assistant": "jarvis_voice_assistant.log",
        "server": "citra_ui_server.log",
    }
    filename = allowed.get(log_name)
    if filename is None:
        raise ValueError(f"unknown log {log_name!r}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} does not exist yet")

    # -NoExit keeps the window open after the tail is interrupted, so a
    # stray Ctrl-C does not make the window vanish along with whatever
    # you were reading. CREATE_NEW_CONSOLE is what makes it a VISIBLE
    # window rather than a hidden child of this pythonw process.
    subprocess.Popen(
        ["powershell", "-NoExit", "-Command",
         f"$Host.UI.RawUI.WindowTitle='Citra - {filename}'; "
         f"Get-Content -LiteralPath '{path}' -Wait -Tail 60"],
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    return filename


async def api_mute(request: web.Request) -> web.Response:
    """
    Go quiet for a while, from the phone. One tap before a lecture.

    Defaults to 60 minutes rather than forever - see citra_mute's module
    docstring on why an indefinite mute is a footgun. Pass ?minutes=0 to
    mute with no expiry when you actually mean it.
    """
    raw = request.query.get("minutes", "60")
    try:
        minutes = float(raw)
    except ValueError:
        return web.json_response({"ok": False, "error": "minutes must be a number"}, status=400)
    citra_mute.mute(minutes if minutes > 0 else None, reason="muted from the phone")
    return web.json_response({"ok": True, "did": citra_mute.describe()})


async def api_unmute(request: web.Request) -> web.Response:
    citra_mute.unmute()
    return web.json_response({"ok": True, "did": "Citra can talk again."})


async def api_mute_status(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "state": citra_mute.status(),
                              "summary": citra_mute.describe()})


async def api_status(request: web.Request) -> web.Response:
    # Sequential and on the phone controller, unlike the dashboard's own
    # poll: this is answering a phone that is waiting, so it must fail
    # fast, and the two boards must not be hit at once (see the comment
    # on _hw_state_lock).
    def _read():
        relays = phone_controller.get_all_relay_status()
        ac = phone_controller.get_ac_status()
        return (relays.data if relays.success and relays.data else {},
                ac.data if ac.success and ac.data else {})

    loop = asyncio.get_event_loop()
    relays, ac = await loop.run_in_executor(None, _read)
    return web.json_response({"ok": True, "relays": relays, "ac": ac})


# ----------------------------------------------------------------------
# PHONEBOOK, RECORDINGS AND QUIET HOURS
# ----------------------------------------------------------------------
# Everything here is behind the same /api/ token gate as the rest, and
# reachable only from loopback. The phonebook decides who Citra is
# willing to ring, so it is exactly as sensitive as the relay routes.

async def api_contacts_list(request: web.Request) -> web.Response:
    import citra_contacts

    people = [
        {"name": c.name, "number": c.number, "relation": c.relation,
         "aliases": c.aliases, "note": c.note}
        for c in citra_contacts.book().all()
    ]
    return web.json_response({"ok": True, "contacts": people})


async def api_contacts_add(request: web.Request) -> web.Response:
    import citra_contacts

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)

    aliases = body.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]

    try:
        contact = citra_contacts.book().add(
            name=str(body.get("name", "")),
            number=str(body.get("number", "")),
            relation=str(body.get("relation", "")),
            aliases=aliases,
            note=str(body.get("note", "")),
            overwrite=bool(body.get("overwrite", False)),
        )
    except citra_contacts.ContactError as error:
        # A refusal here is the feature working - duplicate names and
        # duplicate numbers are how the wrong person gets rung.
        return web.json_response({"ok": False, "error": str(error)}, status=409)
    except Exception as error:
        return web.json_response({"ok": False, "error": str(error)}, status=400)

    return web.json_response({"ok": True, "contact": {
        "name": contact.name, "number": contact.number,
        "relation": contact.relation, "aliases": contact.aliases}})


async def api_contacts_delete(request: web.Request) -> web.Response:
    import citra_contacts

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    removed = citra_contacts.book().remove(str(body.get("name", "")))
    return web.json_response({"ok": removed})


def _recordings_dir() -> str:
    """
    Where call recordings live. A plain path, not an import of the call
    module - the calling feature was removed and this page still lists
    whatever recordings already exist on disk.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "call_recordings")


async def api_recordings(request: web.Request) -> web.Response:
    """
    Every saved call, newest first, with its transcript if there is one.
    """
    import wave

    folder = _recordings_dir()
    items = []
    if os.path.isdir(folder):
        for entry in sorted(os.listdir(folder), reverse=True):
            if not entry.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, entry)
            try:
                stat = os.stat(path)
                with wave.open(path, "rb") as handle:
                    seconds = handle.getnframes() / float(handle.getframerate())
            except Exception:
                continue
            transcript = ""
            text_path = path[:-4] + ".txt"
            if os.path.exists(text_path):
                try:
                    with open(text_path, encoding="utf-8") as handle:
                        transcript = handle.read()
                except Exception:
                    pass
            items.append({
                "file": entry,
                "seconds": round(seconds, 1),
                "size_kb": round(stat.st_size / 1024.0),
                "when": time.strftime("%d %b %Y, %I:%M %p",
                                      time.localtime(stat.st_mtime)),
                "transcript": transcript,
            })
    return web.json_response({"ok": True, "recordings": items})


async def api_recording_play(request: web.Request):
    """
    Stream one recording back.

    The filename is taken as a BASENAME ONLY and re-joined to the
    recordings folder. Passing it through unchecked would let
    ?file=../../../Windows/win.ini walk straight out of the directory -
    the whole point of this route is that it can only ever serve calls.
    """
    name = os.path.basename(request.query.get("file", ""))
    if not name.lower().endswith(".wav"):
        return web.json_response({"ok": False, "error": "not a recording"},
                                 status=400)
    path = os.path.join(_recordings_dir(), name)
    if not os.path.isfile(path):
        return web.json_response({"ok": False, "error": "no such recording"},
                                 status=404)
    return web.FileResponse(path, headers={"Content-Type": "audio/wav"})


async def api_quiet_get(request: web.Request) -> web.Response:
    import citra_quiet_hours

    start, end, enabled = citra_quiet_hours.load_config()
    verdict = citra_quiet_hours.check()
    return web.json_response({
        "ok": True,
        "quiet_from": start.strftime("%H:%M"),
        "quiet_until": end.strftime("%H:%M"),
        "enabled": enabled,
        "calling_allowed_now": verdict.allowed,
        "reason": verdict.reason,
    })


async def api_quiet_set(request: web.Request) -> web.Response:
    import citra_quiet_hours

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    try:
        citra_quiet_hours.save_config(
            str(body.get("quiet_from", "22:00")),
            str(body.get("quiet_until", "07:00")),
            bool(body.get("enabled", True)))
    except Exception as error:
        return web.json_response({"ok": False, "error": str(error)}, status=400)
    return await api_quiet_get(request)


async def calls_page_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(os.path.join(STATIC_DIR, "calls.html"))


async def index_handler(request: web.Request) -> web.FileResponse:
    # aiohttp's add_static() below does NOT auto-serve index.html for the
    # bare directory root (it 403s instead, since show_index=False disables
    # directory listing entirely, root path included) -- an explicit route
    # for "/" is the standard fix, not a workaround.
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


def create_app() -> web.Application:
    app = web.Application(
        middlewares=[no_cache_middleware, api_token_middleware])

    # Remote-control API. Registered BEFORE add_static("/") below, which
    # is a catch-all and would otherwise swallow these paths.
    app.router.add_get("/api/lights/{state}", api_lights)
    app.router.add_get("/api/light/{number}/{state}", api_light)
    app.router.add_get("/api/ac/{state}", api_ac)
    app.router.add_get("/api/ac/temp/{value}", api_ac_temp)
    app.router.add_get("/api/everything/off", api_everything_off)
    app.router.add_get("/api/ask", api_ask)
    app.router.add_get("/api/mute", api_mute)
    app.router.add_get("/api/unmute", api_unmute)
    app.router.add_get("/api/mute/status", api_mute_status)
    app.router.add_get("/api/status", api_status)

    # --- phonebook, recordings, quiet hours -------------------------
    app.router.add_get("/api/contacts", api_contacts_list)
    app.router.add_post("/api/contacts", api_contacts_add)
    app.router.add_post("/api/contacts/delete", api_contacts_delete)
    app.router.add_get("/api/recordings", api_recordings)
    app.router.add_get("/api/recordings/play", api_recording_play)
    app.router.add_get("/api/quiet", api_quiet_get)
    app.router.add_post("/api/quiet", api_quiet_set)

    app.on_startup.append(prewarm_text_router)

    app.router.add_get("/calls", calls_page_handler)
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/caption", caption_handler)
    app.router.add_post("/state", state_handler)
    app.router.add_post("/transcript", transcript_handler)
    app.router.add_post("/voice", voice_handler)
    app.router.add_static("/", STATIC_DIR, show_index=False)
    return app


if __name__ == "__main__":
    # HTTPS if a cert is present, plain HTTP otherwise -- either works fine
    # now that this server is loopback-only: browsers treat "localhost"
    # itself as a secure context regardless of scheme, so even the
    # microphone feature (getUserMedia) doesn't actually need TLS here
    # anymore. Kept as an option rather than removed since it doesn't cost
    # anything to leave working.
    ssl_context: ssl.SSLContext | None = None
    if os.path.exists(TLS_CERT_PATH) and os.path.exists(TLS_KEY_PATH):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(TLS_CERT_PATH, TLS_KEY_PATH)
        scheme = "https"
    else:
        scheme = "http"

    # THE ONE CHECK THAT MATTERS. Binding anywhere other than loopback
    # makes every relay in this flat reachable by whoever can route a
    # packet here. The /api/* routes carry a token, but the dashboard,
    # the WebSocket and the voice-ingest route do NOT - they were built
    # on the assumption that only this machine could reach them. If that
    # assumption is broken, refusing to start is the only honest
    # behaviour: a server that came up anyway would be silently serving
    # unauthenticated control of somebody's home.
    #
    # The intended way to reach this from mobile data is a private mesh
    # VPN (Tailscale), where BIND_HOST is the tailnet address and nothing
    # is exposed to the public internet at all. Port-forwarding this to a
    # public IP is not supported and should not be done.
    if not _is_loopback(BIND_HOST) and os.environ.get("CITRA_ALLOW_REMOTE") != "1":
        logger.error("CITRA_BIND_HOST is %r, which is not loopback.", BIND_HOST)
        logger.error("The dashboard and WebSocket on this server have NO authentication.")
        logger.error("Binding them to a reachable address would expose control of every")
        logger.error("relay and the AC to anyone who can reach this machine.")
        logger.error("")
        logger.error("If this is a private tailnet address and you understand the above,")
        logger.error("set CITRA_ALLOW_REMOTE=1 to proceed. Do NOT do this on a public IP.")
        raise SystemExit(1)

    logger.info("Citra UI server starting on %s://%s:%d", scheme, BIND_HOST, UI_SERVER_PORT)
    if _is_loopback(BIND_HOST):
        logger.info("Local machine only -- not reachable from your phone or other")
        logger.info("devices on the WiFi (see the module docstring for why).")
    else:
        logger.warning("REACHABLE OFF THIS MACHINE at %s -- /api/* is token-gated,", BIND_HOST)
        logger.warning("everything else is NOT. Only do this on a private tailnet.")
    logger.info("Phone API token is in %s", API_TOKEN_PATH)
    logger.info("This is a supplementary control surface -- the voice assistant")
    logger.info("works fine without it running.")
    web.run_app(create_app(), host=BIND_HOST, port=UI_SERVER_PORT, ssl_context=ssl_context, print=None)
