"""
=============================================================================
JARVIS SMART ROOM SYSTEM — HARDWARE API
=============================================================================
Runs on: Windows 11 Pro laptop (the "brain" of the system)
Talks to: NodeMCU 1 (relay_server.ino) and NodeMCU 2 (ac_ir_server.ino)
          over the local Wi-Fi network via plain HTTP GET requests.

WHY THIS FILE EXISTS
---------------------
Your local LLM does not speak HTTP or GPIO. It speaks function calls (tool
calls). This module is the bridge: it exposes clean Python methods like
turn_on_relay(1) and set_ac_temperature(22), each of which internally builds
the right HTTP request to the right NodeMCU and turns the response into a
plain Python dict the LLM's tool-calling loop can consume.

The two NodeMCU boards are addressed by STATIC IP by default (Windows'
mDNS/.local resolution proved unreliable in testing — getaddrinfo failed
even though both boards were reachable and confirmed working via browser
at their raw IPs). If you'd rather use the .local hostnames from the
.ino files' MDNS_HOSTNAME once mDNS is sorted out on your network, pass
relay_host="jarvis-relays.local" / ac_host="jarvis-ac.local" explicitly
when constructing SmartRoomController — the class still accepts either
form, since _get() just builds a URL from whatever string it's given.

DEPENDENCIES
-------------
Only one external package is required:
    pip install requests

Everything else used here (dataclasses, typing, json, logging, time) is
Python standard library.
=============================================================================
"""

import json
import logging
import time
from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # type: ignore

import citra_logging

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
# A local automation system that silently swallows hardware failures is
# worse than useless — it lies to the LLM about whether a command actually
# worked. This logger gives you a persistent trail of what was sent, what
# came back, and what failed, which matters a lot when debugging "why didn't
# my lamp turn on" three rooms away from the laptop.
citra_logging.configure()
logger = logging.getLogger("jarvis_hardware")


# -----------------------------------------------------------------------------
# RESULT OBJECT
# -----------------------------------------------------------------------------
@dataclass
class HardwareResult:
    """
    Uniform return type for every controller method.

    WHY A DATACLASS INSTEAD OF JUST RETURNING True/False:
    An LLM tool-calling loop generally works best when every tool returns a
    small, consistent JSON-serializable shape it can reason about — "did it
    succeed, and if not, why not" — rather than some methods returning bools,
    others returning strings, and others raising exceptions the LLM's tool
    runner has to catch. `to_dict()` below is what you'll actually hand back
    to the LLM framework as the tool call's result.
    """
    success: bool
    endpoint: str
    message: str
    data: dict | None = field(default=None)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "endpoint": self.endpoint,
            "message": self.message,
            "data": self.data,
        }


# -----------------------------------------------------------------------------
# SMART ROOM CONTROLLER
# -----------------------------------------------------------------------------
class SmartRoomController:
    """
    Object-oriented controller for the Jarvis smart room hardware.

    One instance of this class manages both NodeMCU boards for the room.
    Instantiate it once when your LLM framework starts up and reuse it for
    the lifetime of the process — the internal `requests.Session` keeps
    TCP connections alive between calls, which is both faster and kinder to
    the ESP8266's limited connection-handling resources than opening a brand
    new connection for every single command.
    """

    def __init__(
        self,
        relay_host: str = "192.168.0.8",  # confirmed working relay NodeMCU
                                            # static IP.
        ac_host: str = "192.168.0.11",  # confirmed working via browser
                                         # test: http://192.168.0.11/ac/power?state=on
        timeout: float = 3.0,
        max_retries: int = 2,
        registry=None,
    ):
        """
        Args:
            relay_host: Static IP (or mDNS hostname, if you get mDNS
                        working later) of the relay NodeMCU. Confirmed
                        working at 192.168.0.8.
            ac_host:    Static IP of the AC IR NodeMCU. Confirmed working
                        at 192.168.0.11.
            timeout:    Seconds to wait for a single HTTP request before
                        giving up. 3 seconds is generous for a local Wi-Fi
                        LAN request — if a board is genuinely reachable,
                        ESP8266's tiny web server responds in milliseconds.
                        A short timeout means a dead board fails FAST instead
                        of hanging your LLM's response for a long time.
            max_retries: How many times to retry a failed request before
                        giving up entirely. Covers transient Wi-Fi hiccups
                        (a single dropped packet) without retrying forever
                        against a board that's genuinely powered off.
        """
        self.relay_host = relay_host
        self.ac_host = ac_host
        self.timeout = timeout

        # A citra_devices.DeviceRegistry, or None for the original
        # single-board behaviour. Everything above this line still works
        # exactly as before when it is None: relay_host/ac_host remain the
        # addresses for turn_on_relay(1..4) and the AC methods, so the
        # voice protocols, the dashboard and the existing tests were not
        # touched by the move to eleven boards. The registry-aware methods
        # at the bottom of this class are additive.
        self.registry = registry

        # ---------------------------------------------------------------
        # requests.Session + Retry adapter
        # ---------------------------------------------------------------
        # WHY A RETRY ADAPTER INSTEAD OF A MANUAL try/except LOOP:
        # urllib3's Retry class handles the fiddly correct behavior for us —
        # exponential backoff between attempts, which specific failure types
        # are worth retrying (connection errors, 5xx server errors) versus
        # which aren't (a 400 Bad Request will never succeed just by trying
        # again), and how many total attempts to allow. Reimplementing this
        # by hand is a classic source of subtle bugs (e.g. retrying on 400s
        # forever, or not backing off and hammering a struggling board).
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,  # 0.5s, 1s, 2s... between retries
            status_forcelist=[500, 502, 503, 504],  # retry on server errors
            allowed_methods=["GET"],  # this system only ever issues GETs
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)

    # -------------------------------------------------------------------
    # INTERNAL: single shared request method
    # -------------------------------------------------------------------
    def _get(self, host: str, path: str, params: dict | None = None) -> HardwareResult:
        """
        Every public method below funnels through here. Centralizing the
        actual HTTP call and its error handling in one place means every
        single controller method automatically gets the same timeout
        behavior, the same retry behavior, and the same failure-message
        format — instead of copy-pasting a try/except block 15 times and
        having them slowly drift out of sync with each other as the file
        grows.
        """
        url = f"http://{host}{path}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)

            # raise_for_status() turns HTTP error codes (400, 404, 500...)
            # into a Python exception we catch below, instead of us having
            # to manually check response.status_code after every call.
            response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError:
                # The board responded, but not with valid JSON. This
                # shouldn't happen with the firmware provided here, but if
                # you hand-edit the .ino files later and break a JSON string,
                # you want a clear error here rather than an unhandled
                # exception deeper in your LLM framework.
                logger.warning("Non-JSON response from %s: %s", url, response.text[:200])
                return HardwareResult(
                    success=False,
                    endpoint=url,
                    message="Board responded but payload was not valid JSON.",
                    data=None,
                )

            logger.info("OK  %s -> %s", url, data)
            return HardwareResult(success=True, endpoint=url, message="OK", data=data)

        except requests.exceptions.Timeout:
            # The board is likely powered off, Wi-Fi is down, or it's stuck
            # in a bad state. This is the single most common real-world
            # failure mode for this project, so the message is written to
            # be immediately actionable rather than a generic "error".
            msg = f"Timed out after {self.timeout}s — board may be offline or unreachable."
            logger.error("TIMEOUT %s -> %s", url, msg)
            return HardwareResult(success=False, endpoint=url, message=msg, data=None)

        except requests.exceptions.ConnectionError as exc:
            # Distinct from Timeout: this usually means DNS/mDNS resolution
            # itself failed (hostname unknown) or the connection was
            # actively refused — different enough from "no response at all"
            # that it's worth a separate message when you're debugging.
            msg = f"Connection failed — check mDNS hostname/IP and Wi-Fi. ({exc})"
            logger.error("CONN_ERROR %s -> %s", url, msg)
            return HardwareResult(success=False, endpoint=url, message=msg, data=None)

        except requests.exceptions.HTTPError as exc:
            # The board responded, but with an error status (our firmware
            # uses 400 for bad params, 404 for unknown routes). We still
            # try to surface the board's own JSON error message if present,
            # since relay_server.ino / ac_ir_server.ino both return
            # descriptive {"error": "..."} bodies on failure.
            detail = ""
            try:
                detail = response.json().get("error", "")
            except Exception:
                pass
            msg = f"HTTP error {response.status_code}: {detail or str(exc)}"
            logger.error("HTTP_ERROR %s -> %s", url, msg)
            return HardwareResult(success=False, endpoint=url, message=msg, data=None)

        except requests.exceptions.RequestException as exc:
            # Catch-all for anything else requests can throw (malformed URL,
            # SSL issues that don't apply here but could if you add HTTPS
            # later, etc.) so a genuinely unexpected error never propagates
            # up and crashes the LLM's tool-calling loop mid-conversation.
            msg = f"Unexpected request failure: {exc}"
            logger.error("REQUEST_EXC %s -> %s", url, msg)
            return HardwareResult(success=False, endpoint=url, message=msg, data=None)

    # =====================================================================
    # RELAY CONTROL METHODS  (NodeMCU 1 — relay_server.ino)
    # =====================================================================
    # NOTE: These accept relay_number as 1-4 (matching the physical labeling
    # and the URL scheme /relay1../relay4..) — the 0-indexing lives only
    # inside the firmware's internal array, not in this public API surface.

    def _validate_relay_number(self, relay_number: int) -> HardwareResult | None:
        """Shared bounds check so every relay method rejects bad input the
        same way, before it ever reaches the network."""
        if relay_number not in (1, 2, 3, 4):
            return HardwareResult(
                success=False,
                endpoint="(validation)",
                message=f"relay_number must be 1-4, got {relay_number}.",
                data=None,
            )
        return None

    def turn_on_relay(self, relay_number: int) -> HardwareResult:
        """Turn ON a specific relay channel (1-4)."""
        invalid = self._validate_relay_number(relay_number)
        if invalid:
            return invalid
        return self._get(self.relay_host, f"/relay{relay_number}/on")

    def turn_off_relay(self, relay_number: int) -> HardwareResult:
        """Turn OFF a specific relay channel (1-4)."""
        invalid = self._validate_relay_number(relay_number)
        if invalid:
            return invalid
        return self._get(self.relay_host, f"/relay{relay_number}/off")

    def get_relay_status(self, relay_number: int) -> HardwareResult:
        """Get the current commanded ON/OFF state of a specific relay (1-4)."""
        invalid = self._validate_relay_number(relay_number)
        if invalid:
            return invalid
        return self._get(self.relay_host, f"/relay{relay_number}/status")

    def get_all_relay_status(self) -> HardwareResult:
        """Get the current commanded ON/OFF state of all 4 relays at once."""
        return self._get(self.relay_host, "/status")

    def turn_off_all_relays(self) -> HardwareResult:
        """
        Convenience 'all off' — e.g. for a 'goodnight' or 'leaving home'
        voice command. Note this issues 4 separate sequential HTTP calls
        (the firmware has no single 'all off' endpoint); if any individual
        call fails, that failure is logged but the method still attempts
        the remaining relays rather than aborting early, so one stuck
        channel doesn't leave the other three lights or appliances on.
        """
        results = []
        all_succeeded = True
        for i in range(1, 5):
            result = self.turn_off_relay(i)
            results.append(result.to_dict())
            if not result.success:
                all_succeeded = False
        return HardwareResult(
            success=all_succeeded,
            endpoint=f"http://{self.relay_host}/relay[1-4]/off",
            message="All relays commanded off." if all_succeeded
                    else "One or more relays failed to respond — see data for details.",
            data={"results": results},
        )

    # =====================================================================
    # RELATIVE BRIGHTNESS STEPPING  (NodeMCU 1 — relay_server.ino)
    # =====================================================================
    # This relay board switches lights on/off, it doesn't dim them — so
    # "increase/decrease lighting" is approximated by how many of the 4
    # relays are on at once, not real PWM brightness. See the matching
    # comment above handleBrightnessUp() in relay_server.ino for the exact
    # stepping rule (lowest-index-off turns on next; highest-index-on
    # turns off next).

    def increase_brightness(self) -> HardwareResult:
        """Turn on one more light than are currently on (no-op if all 4 are already on)."""
        return self._get(self.relay_host, "/brightness/up")

    def decrease_brightness(self) -> HardwareResult:
        """Turn off one light from among those currently on (no-op if none are on)."""
        return self._get(self.relay_host, "/brightness/down")

    def set_max_brightness(self) -> HardwareResult:
        """Turn on all 4 lights — the brightest this relay-switched hardware can go."""
        return self._get(self.relay_host, "/brightness/max")

    def set_min_brightness(self) -> HardwareResult:
        """Drop to just one light on (relay 1) — dimmest while still lit, not fully off."""
        return self._get(self.relay_host, "/brightness/min")

    # =====================================================================
    # AC / IR CONTROL METHODS  (NodeMCU 2 — ac_ir_server.ino)
    # =====================================================================

    def set_ac_power(self, state: bool) -> HardwareResult:
        """Turn the AC on (True) or off (False) via IR blast."""
        return self._get(self.ac_host, "/ac/power", params={"state": "on" if state else "off"})

    def set_ac_temperature(self, temp_celsius: int) -> HardwareResult:
        """
        Set AC target temperature in Celsius.
        Firmware-side range checking (16-30 for the Carrier example wrapper)
        applies on the NodeMCU; this method does not duplicate that range
        check here so there is exactly one source of truth for valid range
        (the firmware) rather than two places that could drift out of sync
        if you change AC brands later.
        """
        return self._get(self.ac_host, "/ac/temp", params={"val": temp_celsius})

    def set_ac_mode(self, mode: str) -> HardwareResult:
        """
        Set AC mode. Valid values: 'cool', 'heat', 'fan', 'dry', 'auto'
        (matches the whitelist enforced in ac_ir_server.ino's handleMode()).
        """
        return self._get(self.ac_host, "/ac/mode", params={"val": mode.lower()})

    def set_ac_fan_speed(self, speed: str) -> HardwareResult:
        """
        Set AC fan speed. Valid values: 'auto', 'low', 'med', 'high'
        (matches the whitelist enforced in ac_ir_server.ino's handleFan()).
        The firmware endpoint this wraps (/ac/fan) already existed, but had
        no corresponding controller method until now — an existing gap,
        not something new added alongside the up/down steppers below.
        """
        return self._get(self.ac_host, "/ac/fan", params={"val": speed.lower()})

    def increase_ac_fan_speed(self) -> HardwareResult:
        """
        Step the fan speed up one rung (low -> med -> high). From 'auto',
        steps onto the ladder at 'low' — see the comment above
        handleFanUp() in ac_ir_server.ino for why 'auto' isn't itself a
        rung. No-op if already at 'high'.
        """
        return self._get(self.ac_host, "/ac/fan/up")

    def decrease_ac_fan_speed(self) -> HardwareResult:
        """
        Step the fan speed down one rung (high -> med -> low). No-op if
        already at 'low', or currently on 'auto' (nothing lower to step to).
        """
        return self._get(self.ac_host, "/ac/fan/down")

    def get_ac_status(self) -> HardwareResult:
        """
        Get the last COMMANDED AC state (power/temp/mode). Remember: IR is
        one-way, so this is not a live read of the physical AC unit — see
        the AcState comment in ac_ir_server.ino for why.
        """
        return self._get(self.ac_host, "/ac/status")

    # =====================================================================
    # SYSTEM-WIDE HEALTH CHECK
    # =====================================================================

    def check_system_health(self) -> HardwareResult:
        """
        Pings both boards' root endpoints. Useful as a startup sanity check
        for your LLM framework, or as a periodic heartbeat before trusting
        that a batch of commands will actually land on real hardware.
        """
        relay_ok = self._get(self.relay_host, "/")
        ac_ok = self._get(self.ac_host, "/")
        both_ok = relay_ok.success and ac_ok.success
        return HardwareResult(
            success=both_ok,
            endpoint="(system health)",
            message="All boards online." if both_ok else "One or more boards unreachable.",
            data={
                "relay_board": relay_ok.to_dict(),
                "ac_board": ac_ok.to_dict(),
            },
        )

    # =====================================================================
    # MULTI-BOARD: addressing switches by room and name
    # =====================================================================
    # Everything above addresses ONE relay board, because for a long time
    # there was only one. These methods take a citra_devices.Switch (or a
    # spoken phrase to resolve into one) and talk to whichever board
    # actually carries it. The firmware is identical on every board, so
    # the only thing that varies is the host - which is exactly what the
    # registry knows and nothing else needed to.
    #
    # WHY EVERY FAN-OUT HERE IS SEQUENTIAL: an ESP8266 running the Arduino
    # WebServer library serves one connection at a time on very little
    # RAM, and requests.Session is not thread-safe either. Hitting six
    # switches across three hall boards concurrently would be two
    # different bugs at once. Six sequential calls at ~30-100ms is still
    # inside a blink; see citra_ui_server's own comment on the same point.

    def _no_registry(self, what: str) -> HardwareResult:
        return HardwareResult(
            success=False,
            endpoint="(registry)",
            message=(
                f"Cannot {what}: no device registry attached. Construct "
                "SmartRoomController(registry=DeviceRegistry.load())."
            ),
        )

    def set_switch(self, switch, state: bool) -> HardwareResult:
        """Turn one named switch on or off, on whichever board carries it."""
        action = "on" if state else "off"
        result = self._get(switch.host, f"/relay{switch.channel}/{action}")
        # Rewrite the message in the user's terms. "Turned on relay 2" is
        # meaningless to anyone standing in the room; "bedroom 1 fan" is
        # the thing they asked for and the thing a log should show.
        if result.success:
            return HardwareResult(
                success=True,
                endpoint=result.endpoint,
                message=f"{switch.label} {action}.",
                data={"switch": switch.label, "board": switch.board,
                      "channel": switch.channel, "state": action},
            )
        return result

    def set_switch_by_name(self, spoken: str, state: bool) -> HardwareResult:
        """Resolve free text to a switch and set it."""
        if self.registry is None:
            return self._no_registry("resolve a switch by name")
        switch = self.registry.find_switch(spoken)
        if switch is None:
            return HardwareResult(
                success=False,
                endpoint="(registry)",
                message=f"I don't have anything called {spoken!r}.",
            )
        return self.set_switch(switch, state)

    def set_room(self, room: str, state: bool) -> HardwareResult:
        """
        Turn every switch in a room on or off, across ALL of its boards.

        This is what "turn off the hall" has to mean once the hall is
        three separate switchboards. Reports partial failure honestly:
        with eleven boards, one being unreachable while the rest work is
        the normal failure, not an exceptional one, and saying "done"
        because five of six succeeded is how people stop trusting it.
        """
        if self.registry is None:
            return self._no_registry("address a whole room")
        switches = self.registry.switches_in(room)
        if not switches:
            return HardwareResult(
                success=False, endpoint="(registry)",
                message=f"I don't know a room called {room!r}.",
            )

        action = "on" if state else "off"
        failed = []
        for switch in switches:
            if not self.set_switch(switch, state).success:
                failed.append(switch.label)

        if not failed:
            return HardwareResult(
                success=True, endpoint=f"(room {room})",
                message=f"Everything in the {room} {action}.",
                data={"room": room, "switched": len(switches), "failed": []},
            )
        return HardwareResult(
            success=False, endpoint=f"(room {room})",
            message=(
                f"{len(switches) - len(failed)} of {len(switches)} in the {room} went "
                f"{action}. Couldn't reach: {', '.join(failed)}."
            ),
            data={"room": room, "switched": len(switches) - len(failed), "failed": failed},
        )

    def set_ac_power_in(self, room: str, state: bool) -> HardwareResult:
        """Power one room's AC, resolved through the registry."""
        if self.registry is None:
            return self._no_registry("address an AC by room")
        ac = self.registry.ac_in(room)
        if ac is None:
            return HardwareResult(
                success=False, endpoint="(registry)",
                message=f"There's no AC in the {room}.",
            )
        return self._get(ac.host, "/ac/power", params={"state": "on" if state else "off"})

    def set_ac_temperature_in(self, room: str, temp_celsius: int) -> HardwareResult:
        """Set one room's AC temperature, resolved through the registry."""
        if self.registry is None:
            return self._no_registry("address an AC by room")
        ac = self.registry.ac_in(room)
        if ac is None:
            return HardwareResult(
                success=False, endpoint="(registry)",
                message=f"There's no AC in the {room}.",
            )
        return self._get(ac.host, "/ac/temp", params={"val": temp_celsius})

    def everything_off(self) -> HardwareResult:
        """
        Every switch in the flat, plus every AC. What "Citra, turn off
        everything" means once there is more than one board.
        """
        if self.registry is None:
            return self._no_registry("turn off the whole flat")
        failed = []
        for switch in self.registry.switches:
            if not self.set_switch(switch, False).success:
                failed.append(switch.label)
        for ac in self.registry.air_conditioners:
            if not self._get(ac.host, "/ac/power", params={"state": "off"}).success:
                failed.append(f"{ac.room} AC")
        total = len(self.registry.switches) + len(self.registry.air_conditioners)
        if not failed:
            return HardwareResult(
                success=True, endpoint="(whole flat)",
                message="Everything off.", data={"switched": total, "failed": []},
            )
        return HardwareResult(
            success=False, endpoint="(whole flat)",
            message=f"{total - len(failed)} of {total} off. Couldn't reach: {', '.join(failed)}.",
            data={"switched": total - len(failed), "failed": failed},
        )

    def board_health(self) -> HardwareResult:
        """
        Pings every board in the registry and reports which answered.

        check_system_health above pings the two hardcoded hosts. With
        eleven boards the useful question is not "is the system up" but
        "WHICH board is down" - that is the difference between a support
        call you can answer from the sofa and one that needs a visit.
        """
        if self.registry is None:
            return self._no_registry("check every board")
        alive, dead = [], []
        for board in self.registry.boards:
            host = self.registry.host_for_board(board)
            (alive if self._get(host, "/").success else dead).append(f"{board} ({host})")
        for ac in self.registry.air_conditioners:
            (alive if self._get(ac.host, "/").success else dead).append(
                f"{ac.room} AC ({ac.host})")
        return HardwareResult(
            success=not dead,
            endpoint="(board health)",
            message=("All %d boards online." % len(alive)) if not dead
                    else ("%d up, %d down: %s" % (len(alive), len(dead), ", ".join(dead))),
            data={"alive": alive, "dead": dead},
        )


# =============================================================================
# SAMPLE EXECUTION / MANUAL TEST BLOCK
# =============================================================================
if __name__ == "__main__":
    # This block tests BOTH the relay board and the AC IR board.
    controller = SmartRoomController(
        relay_host="192.168.0.8",
        ac_host="192.168.0.11",
        timeout=3.0,
        max_retries=2,
    )

    print("\n=== Jarvis Hardware API — Full System Manual Test ===\n")

    print("--- System Health Check ---")
    result = controller.check_system_health()
    print(json.dumps(result.to_dict(), indent=2))
    time.sleep(1)

    print("\n--- Relays: Turn ON all (1 by 1) ---")
    for i in range(1, 5):
        result = controller.turn_on_relay(i)
        print(f"Relay {i}: {result.message}")
        time.sleep(0.5)

    print("\n--- Relays: Get all status ---")
    result = controller.get_all_relay_status()
    print(json.dumps(result.to_dict(), indent=2))
    time.sleep(1)

    print("\n--- Relays: Turn OFF all ---")
    result = controller.turn_off_all_relays()
    print(json.dumps(result.to_dict(), indent=2))
    time.sleep(1)

    print("\n--- AC: power on ---")
    result = controller.set_ac_power(True)
    print(json.dumps(result.to_dict(), indent=2))
    time.sleep(1)

    print("\n--- AC: set temperature to 22C ---")
    result = controller.set_ac_temperature(22)
    print(json.dumps(result.to_dict(), indent=2))
    time.sleep(1)

    print("\n--- AC: set mode to cool ---")
    result = controller.set_ac_mode("cool")
    print(json.dumps(result.to_dict(), indent=2))
    time.sleep(1)

    print("\n--- AC: get current commanded status ---")
    result = controller.get_ac_status()
    print(json.dumps(result.to_dict(), indent=2))

    print("\n--- AC: power off ---")
    result = controller.set_ac_power(False)
    print(json.dumps(result.to_dict(), indent=2))

    print("\n=== Full manual test complete ===\n")


# =============================================================================
# OPENAI-STYLE TOOL CALLING SCHEMA
# =============================================================================
# This is a plain Python list/dict structure — importable as-is — matching
# the JSON Schema shape used by OpenAI's function-calling / tool-calling
# API, and widely adopted by local LLM frameworks (llama.cpp server,
# text-generation-webui, LM Studio, Ollama's OpenAI-compatible endpoint,
# etc.) since it's become a de facto standard shape, not an OpenAI-only one.
#
# Each entry's "name" matches a real method on SmartRoomController above —
# your tool-calling loop's dispatcher should map tool_call.function.name
# straight to getattr(controller, name)(**arguments) once the LLM emits a
# call.
#
# WHY THIS LIVES AT MODULE LEVEL (not inside the class):
# The schema describes the *interface* your LLM sees, which is a slightly
# different concern from the *implementation* the class provides. Keeping it
# separate means you can hand this list directly to your LLM framework's
# `tools=` parameter without any transformation, while the class itself
# stays a normal, framework-agnostic Python object usable anywhere.
# =============================================================================

JARVIS_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "turn_on_relay",
            "description": "Turn ON a specific relay channel, switching on whatever appliance or light is wired to that channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relay_number": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4],
                        "description": "Which relay channel to turn on (1-4).",
                    }
                },
                "required": ["relay_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_off_relay",
            "description": "Turn OFF a specific relay channel, switching off whatever appliance or light is wired to that channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relay_number": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4],
                        "description": "Which relay channel to turn off (1-4).",
                    }
                },
                "required": ["relay_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_relay_status",
            "description": "Get the current commanded ON/OFF state of a single relay channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relay_number": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4],
                        "description": "Which relay channel to check (1-4).",
                    }
                },
                "required": ["relay_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_relay_status",
            "description": "Get the current commanded ON/OFF state of all 4 relay channels at once.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_off_all_relays",
            "description": "Turn OFF all 4 relay channels at once. Useful for 'goodnight', 'leaving home', or 'turn everything off' style commands.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "increase_brightness",
            "description": "Turn on one more light than are currently on. This relay hardware switches lights on/off rather than truly dimming them, so 'brighter' means more lights on, not a smooth PWM increase. No effect if all 4 lights are already on.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decrease_brightness",
            "description": "Turn off one of the currently-on lights, making the room dimmer. No effect if no lights are currently on.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_max_brightness",
            "description": "Turn on all 4 lights at once — the brightest this hardware supports. Use for 'maximum brightness', 'brightest', 'full lights'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_min_brightness",
            "description": "Drop to just one light on — the dimmest setting that's still lit (not the same as turning everything off). Use for 'minimum brightness', 'dimmest', 'just a little light'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ac_power",
            "description": "Turn the air conditioner on or off by sending an infrared command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "boolean",
                        "description": "True to turn the AC on, false to turn it off.",
                    }
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ac_temperature",
            "description": "Set the air conditioner's target temperature in Celsius by sending an infrared command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp_celsius": {
                        "type": "integer",
                        "description": "Target temperature in Celsius, typically 16-30 depending on the AC unit's supported range.",
                    }
                },
                "required": ["temp_celsius"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ac_mode",
            "description": "Set the air conditioner's operating mode by sending an infrared command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["cool", "heat", "fan", "dry", "auto"],
                        "description": "The AC operating mode to switch to.",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_ac_fan_speed",
            "description": "Set the air conditioner's fan speed directly by sending an infrared command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "string",
                        "enum": ["auto", "low", "med", "high"],
                        "description": "The fan speed to switch to.",
                    }
                },
                "required": ["speed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "increase_ac_fan_speed",
            "description": "Step the AC fan speed up one level (low -> med -> high). From 'auto', steps onto the ladder at 'low'. No effect if already at 'high'. Use for 'increase fan speed', 'fan speed up', 'make the fan faster'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decrease_ac_fan_speed",
            "description": "Step the AC fan speed down one level (high -> med -> low). No effect if already at 'low' or currently on 'auto'. Use for 'decrease fan speed', 'fan speed down', 'make the fan slower'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ac_status",
            "description": "Get the last commanded AC state (power, temperature, mode). Note this reflects the last command sent, not a live read from the AC unit, since infrared communication is one-directional.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_system_health",
            "description": "Check whether both the relay board and the AC IR board are online and reachable on the network.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


if __name__ == "__main__":
    # Also dump the schema when run directly, so you can eyeball it or pipe
    # it straight into a file: python jarvis_hardware_api.py > schema_check.txt
    print("\n=== OpenAI-Style Tool Schema ===\n")
    print(json.dumps(JARVIS_TOOL_SCHEMA, indent=2))
