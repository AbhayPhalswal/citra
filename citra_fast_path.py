"""
=============================================================================
CITRA — FAST PATH: THE REGEX-TO-BOARD TABLE
=============================================================================

The part of Citra that works with the internet down. A spoken hardware
command ("turn on light 2", "set the ac to 22", "turn off everything")
is matched against a small table of compiled regexes and dispatched
straight to a SmartRoomController method. No model is ever involved,
which is what makes sub-millisecond matching possible at all.

Extracted from jarvis_router.py so the table, its handlers and its
matcher can be read and tested on their own, without the Smart Path's
LLM plumbing in the same file. jarvis_router.JarvisRouter still owns
the decision of WHEN to use this (it tries here first, then falls back
to the LLM); this module only answers "does this text name a hardware
command, and what does it map to".

Adding a command means appending one HardwareIntent to INTENTS. The
table is first-match-wins, so put the more specific phrasing first -
see the comments inline.
=============================================================================
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from re import Pattern

import citra_logging
from citra_route_result import RouteResult
from jarvis_hardware_api import HardwareResult, SmartRoomController

citra_logging.configure()
logger = logging.getLogger("citra_fast_path")


# -----------------------------------------------------------------------------
# FAST PATH: HARDWARE INTENT DEFINITION
# -----------------------------------------------------------------------------
@dataclass
class HardwareIntent:
    """
    Binds one compiled regex pattern to the controller method it should
    trigger on a match.

    WHY A DATACLASS OF (pattern, handler) PAIRS INSTEAD OF ONE GIANT
    if/elif LADDER:
    Adding a new hardware command later means appending one entry to the
    INTENTS list below — you don't touch routing logic, you don't risk
    breaking an existing elif chain, and every intent's pattern and handler
    sit next to each other instead of being split across a regex block and
    a separate dispatch block that could drift out of sync.

    `handler` is a callable that takes the SmartRoomController instance and
    the regex Match object, and returns a HardwareResult. Keeping the
    handler signature identical across every intent is what lets the fast
    path's core loop (see JarvisRouter._try_fast_path) stay a single,
    generic few lines rather than one hand-written branch per command.
    """
    pattern: Pattern[str]
    handler: Callable[[SmartRoomController, "re.Match[str]"], HardwareResult]
    description: str  # human-readable, used only for the --list-intents debug helper


# -----------------------------------------------------------------------------
# FAST PATH: INDIVIDUAL HANDLER FUNCTIONS
# -----------------------------------------------------------------------------
# Each handler is intentionally tiny — just pull the matched group(s) out of
# the regex Match object, coerce to the right type, and call exactly one
# SmartRoomController method. All the actual hardware-facing error handling
# (timeouts, retries, connection errors) already lives inside
# SmartRoomController._get() from jarvis_hardware_api.py — this router does
# not duplicate that logic, it just calls into it and passes the
# HardwareResult straight through.

def _handle_relay_on(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    relay_number = int(match.group("channel"))
    return controller.turn_on_relay(relay_number)


def _handle_relay_off(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    relay_number = int(match.group("channel"))
    return controller.turn_off_relay(relay_number)


def _handle_relay_status(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    relay_number = int(match.group("channel"))
    return controller.get_relay_status(relay_number)


def _handle_ac_power_on(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.set_ac_power(True)


def _handle_ac_power_off(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.set_ac_power(False)


def _handle_ac_temperature(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    temp_celsius = int(match.group("temp"))
    return controller.set_ac_temperature(temp_celsius)


def _handle_ac_mode(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    mode = match.group("mode").lower()
    return controller.set_ac_mode(mode)


def _handle_ac_fan_speed(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    # The regex accepts "medium" as a natural-language spelling, but the
    # firmware's whitelist (handleFan() in ac_ir_server.ino) only accepts
    # the short form "med" — normalize here so a match on "medium" doesn't
    # turn into a firmware-side 400 error.
    speed = match.group("speed").lower()
    if speed == "medium":
        speed = "med"
    return controller.set_ac_fan_speed(speed)


def _handle_ac_fan_up(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.increase_ac_fan_speed()


def _handle_ac_fan_down(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.decrease_ac_fan_speed()


def _handle_brightness_up(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.increase_brightness()


def _handle_brightness_down(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.decrease_brightness()


def _handle_brightness_max(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.set_max_brightness()


def _handle_brightness_min(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.set_min_brightness()


def _handle_everything_off(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    """
    "Turn off everything" per your instruction means ALL 4 relays AND the
    AC power — there is no single SmartRoomController method that does
    both (turn_off_all_relays() only covers the relay board), so this
    handler calls both and merges the two HardwareResults into one.

    WHY MERGE RATHER THAN RETURN JUST ONE OF THEM:
    If we silently returned only the relay result, a failure on the AC
    board (e.g. it lost Wi-Fi) would be swallowed and you'd see "success"
    in the terminal while the AC was actually still running. Merging
    means a partial failure is always visible.
    """
    relay_result = controller.turn_off_all_relays()
    ac_result = controller.set_ac_power(False)

    both_succeeded = relay_result.success and ac_result.success
    merged_message = (
        f"Relays: {relay_result.message} | AC: {ac_result.message}"
    )
    return HardwareResult(
        success=both_succeeded,
        endpoint="(fast_path: everything_off)",
        message=merged_message,
        data={"relays": relay_result.to_dict(), "ac": ac_result.to_dict()},
    )


def _handle_system_health(controller: SmartRoomController, match: "re.Match[str]") -> HardwareResult:
    return controller.check_system_health()


# -----------------------------------------------------------------------------
# FAST PATH: COMPILED INTENT TABLE
# -----------------------------------------------------------------------------
# Patterns are compiled ONCE at import time (module load), not per-request.
# Recompiling a regex on every user input would add avoidable overhead to
# every single fast-path call — re.compile() is a real, if small, cost, and
# doing it once up front is the difference between "fast path is fast" and
# "fast path pays a hidden tax on every command."
#
# All patterns use re.IGNORECASE so "Turn On Light 1" and "turn on light 1"
# both match — voice-to-text and manual typing are both realistic inputs
# for a smart-room system, and capitalization shouldn't be load-bearing.
#
# Named groups (?P<channel>...) are used instead of positional groups
# (group(1), group(2)...) so each handler function reads self-documenting
# group names instead of magic numbers — match.group("channel") is
# unambiguous, match.group(1) is not, especially once you have 9 patterns
# in one table.
INTENTS: list[HardwareIntent] = [
    HardwareIntent(
        pattern=re.compile(
            r"\bturn\s+on\s+(?:light|relay|switch|channel)\s*(?P<channel>[1-4])\b",
            re.IGNORECASE,
        ),
        handler=_handle_relay_on,
        description="turn on light/relay/switch <1-4>",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\bturn\s+off\s+(?:light|relay|switch|channel)\s*(?P<channel>[1-4])\b",
            re.IGNORECASE,
        ),
        handler=_handle_relay_off,
        description="turn off light/relay/switch <1-4>",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:status|state)\s+of\s+(?:light|relay|switch|channel)\s*(?P<channel>[1-4])\b",
            re.IGNORECASE,
        ),
        handler=_handle_relay_status,
        description="status of light/relay/switch <1-4>",
    ),
    HardwareIntent(
        # Deliberately checked BEFORE the general relay-off pattern's cousin
        # phrasing would be, but since this uses a distinct "everything"
        # keyword rather than a channel digit, there's no real ordering
        # conflict with the numbered relay patterns above — listed here
        # for readability, grouped with the other whole-room commands.
        pattern=re.compile(
            r"\bturn\s+off\s+everything\b",
            re.IGNORECASE,
        ),
        handler=_handle_everything_off,
        description="turn off everything (all relays + AC power)",
    ),
    HardwareIntent(
        # Checked BEFORE the plain "brighten/dim the lights" pair below —
        # both "brighten"/"dim" and "max"/"min" would otherwise share the
        # word "light(s)", and this table's first-match-wins semantics
        # (see _try_fast_path's docstring) means the more specific
        # max/min phrasing has to come first or it'd never be reached.
        pattern=re.compile(
            r"\b(?:max(?:imum)?|full|brightest)\s+(?:brightness|lights?)\b",
            re.IGNORECASE,
        ),
        handler=_handle_brightness_max,
        description="maximum/full/brightest lights (all 4 relays on)",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:min(?:imum)?|dimmest)\s+(?:brightness|lights?)\b",
            re.IGNORECASE,
        ),
        handler=_handle_brightness_min,
        description="minimum/dimmest lights (drop to 1 relay on)",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:increase|brighten|turn\s+up)\s+(?:the\s+)?(?:brightness|lights?)\b",
            re.IGNORECASE,
        ),
        handler=_handle_brightness_up,
        description="increase/brighten the lights (one more relay on)",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:decrease|dim|turn\s+down)\s+(?:the\s+)?(?:brightness|lights?)\b",
            re.IGNORECASE,
        ),
        handler=_handle_brightness_down,
        description="decrease/dim the lights (one fewer relay on)",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:turn\s+on|power\s+on)\s+(?:the\s+)?ac\b",
            re.IGNORECASE,
        ),
        handler=_handle_ac_power_on,
        description="turn on the ac",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:turn\s+off|power\s+off)\s+(?:the\s+)?ac\b",
            re.IGNORECASE,
        ),
        handler=_handle_ac_power_off,
        description="turn off the ac",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\bset\s+(?:the\s+)?ac\s+to\s+(?P<temp>\d{1,2})\b",
            re.IGNORECASE,
        ),
        handler=_handle_ac_temperature,
        description="set ac to <temp>",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\bset\s+(?:the\s+)?ac\s+mode\s+to\s+(?P<mode>cool|heat|fan|dry|auto)\b",
            re.IGNORECASE,
        ),
        handler=_handle_ac_mode,
        description="set ac mode to <cool|heat|fan|dry|auto>",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\bset\s+(?:the\s+)?(?:ac\s+)?fan\s+(?:speed\s+)?to\s+(?P<speed>auto|low|med(?:ium)?|high)\b",
            re.IGNORECASE,
        ),
        handler=_handle_ac_fan_speed,
        description="set ac fan (speed) to <auto|low|med|high>",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:increase|turn\s+up)\s+(?:the\s+)?fan\s+speed\b",
            re.IGNORECASE,
        ),
        handler=_handle_ac_fan_up,
        description="increase/turn up fan speed",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:decrease|turn\s+down)\s+(?:the\s+)?fan\s+speed\b",
            re.IGNORECASE,
        ),
        handler=_handle_ac_fan_down,
        description="decrease/turn down fan speed",
    ),
    HardwareIntent(
        pattern=re.compile(
            r"\b(?:system\s+health|health\s+check|are\s+you\s+online)\b",
            re.IGNORECASE,
        ),
        handler=_handle_system_health,
        description="system health / health check",
    ),
]


# -----------------------------------------------------------------------------
# MATCHING AND DISPATCH
# -----------------------------------------------------------------------------
def match_intent(text: str) -> tuple[HardwareIntent, "re.Match[str]"] | None:
    """
    The first intent whose pattern matches `text`, with its Match, or
    None if the text is not a hardware command at all. Pure: no I/O.
    """
    for intent in INTENTS:
        match = intent.pattern.search(text)
        if match:
            return intent, match
    return None


def run_fast_path(controller: SmartRoomController, user_input: str) -> RouteResult | None:
    """
    Match `user_input` against INTENTS and, on a hit, dispatch it to the
    board through `controller`. Returns None (not a RouteResult) when
    nothing matches - that is the signal the caller uses to fall through
    to the Smart Path. A None here means "the fast path declines"; a
    RouteResult with success=False means "it handled it and the board
    call failed", which is a different thing.

    WHY "FIRST match wins" AND NOT "check for ambiguous multiple
    matches": with a couple of dozen patterns targeting genuinely
    distinct phrasings, the realistic risk of two patterns BOTH matching
    the same input is low, and optimising for that edge case would mean
    running every pattern against every input even after finding a hit -
    directly working against the <5ms target this path exists for.

    !!! WHAT THE <5MS TARGET ACTUALLY COVERS !!!
    Two phases are timed separately and both are reported:
      1. match_ms    - regex matching + group extraction ONLY. This is
         the part that "bypasses the LLM", and it genuinely runs in low
         single-digit MICROSECONDS.
      2. dispatch_ms - the SmartRoomController call, a real HTTP request
         over Wi-Fi to a physical NodeMCU. Genuine network I/O that this
         module cannot bound; it depends on your Wi-Fi and the board.
    Conflating them would misrepresent what "the Fast Path bypasses the
    LLM" guarantees: it guarantees (1), not (2).
    """
    match_start = time.perf_counter()
    hit = match_intent(user_input)
    match_ms = (time.perf_counter() - match_start) * 1000

    if hit is None:
        return None
    intent, match = hit

    dispatch_start = time.perf_counter()
    try:
        hw_result: HardwareResult = intent.handler(controller, match)
    except Exception as exc:
        # A regex matched, but the handler itself blew up in a way
        # SmartRoomController's own error handling didn't already catch
        # (a programming error in a handler, not a network failure -
        # those are already HardwareResult(success=False, ...)). A
        # last-resort catch so a bug in ONE handler can't take down the
        # whole voice loop.
        dispatch_ms = (time.perf_counter() - dispatch_start) * 1000
        logger.error("Fast path handler raised: %s", exc)
        return RouteResult(
            path="FAST",
            success=False,
            message=f"Internal handler error: {exc}",
            latency_ms=match_ms + dispatch_ms,
            match_latency_ms=match_ms,
            dispatch_latency_ms=dispatch_ms,
        )
    dispatch_ms = (time.perf_counter() - dispatch_start) * 1000

    return RouteResult(
        path="FAST",
        success=hw_result.success,
        message=hw_result.message,
        latency_ms=match_ms + dispatch_ms,
        match_latency_ms=match_ms,
        dispatch_latency_ms=dispatch_ms,
    )
