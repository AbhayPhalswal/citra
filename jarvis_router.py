"""
=============================================================================
CITRA SMART ROOM SYSTEM — DUAL-PATH INTENT ROUTER
=============================================================================
Runs on: Windows 11 Pro laptop, alongside jarvis_hardware_api.py

WHY THIS FILE EXISTS
---------------------
Routing EVERY user utterance through an LLM is wasteful and slow for the
90% of commands that are simple, deterministic hardware toggles ("turn on
light 1"). An LLM call costs hundreds of milliseconds to seconds even on
local hardware — completely unnecessary latency for a command a dozen
regex patterns can parse in microseconds.

This router implements two paths:

  FAST PATH  (deterministic, <5ms target)
    User input -> regex match -> extract variables -> call
    SmartRoomController method directly -> return result.
    The LLM is never invoked. This is the path for "turn on light 2",
    "set ac to 22", "turn off everything", etc.

  SMART PATH (AI fallback, seconds-scale latency)
    User input doesn't match any hardware regex -> forward the raw text
    to a local LLM server (LM Studio, in this file) -> return its response.
    This is the path for "how does quantum computing work", "write me
    a python script", or ANY hardware-adjacent phrasing loose enough
    that we'd rather have a real LLM interpret it than risk a regex
    false-positive.

WHY THE FAST PATH IS A REGEX LADDER, NOT AN LLM FUNCTION CALL:
Regex matching against a handful of compiled patterns is a bounded,
CPU-only operation with no I/O, no network round trip, and no model
inference — that's what makes single-digit-millisecond execution
achievable at all. An LLM-based tool-call approach (like the
JARVIS_TOOL_SCHEMA in jarvis_hardware_api.py) is more flexible and
handles novel phrasing, but it can NEVER hit this latency floor because
model inference itself takes tens to hundreds of milliseconds at best.
This router intentionally keeps the two paths separate so hardware
commands get the fast path's speed, and everything else gets the smart
path's flexibility.

DEPENDENCIES
-------------
    pip install requests
Everything else (re, time, dataclasses, typing) is standard library.

This file imports SmartRoomController from jarvis_hardware_api.py, which
must be in the same directory (or importable on your PYTHONPATH).
=============================================================================
"""

import json
import logging

import requests

import citra_logging
from citra_contacts import CONTACT_TOOL_SCHEMA, ContactController
from citra_fast_path import INTENTS, HardwareIntent, run_fast_path  # noqa: F401 - re-exported
from citra_llm_client import (  # noqa: F401 - constants re-exported for __main__ and callers
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
    LM_STUDIO_TIMEOUT_SECONDS,
    SMART_PATH_BACKEND,
    ConversationMemory,
    GeminiClient,
    LMStudioClient,
    to_gemini_tools,
)
from citra_models import CODE_TOOL_SCHEMA, CodeController
from citra_route_result import RouteResult
from jarvis_hardware_api import JARVIS_TOOL_SCHEMA, SmartRoomController
from jarvis_pc_control import PC_TOOL_SCHEMA, PCController
from jarvis_reminders import REMINDER_TOOL_SCHEMA, ReminderController
from jarvis_vision import VISION_TOOL_SCHEMA, VisionController

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
citra_logging.configure()
logger = logging.getLogger("jarvis_router")


# -----------------------------------------------------------------------------
# SMART PATH TOOL CALLING
# -----------------------------------------------------------------------------
# Previously, the Smart Path could only ever produce a text answer — a
# follow-up like "make it a bit warmer" after "set AC to 22" (routed via the
# Fast Path) had no way to actually change anything, even with conversation
# memory giving it the context to know WHAT to change. Wiring JARVIS_TOOL_SCHEMA
# (jarvis_hardware_api.py) into this request closes that gap: the model can
# now emit a real tool_calls response instead of just text, which gets
# dispatched against self.controller exactly like the Fast Path does.
#
# ALL_TOOL_SCHEMA combines every subsystem's tool definitions (smart-room
# hardware + local PC control + local-vision/input + reminders) into the
# ONE list actually sent to the LLM — from the model's point of view
# there's a single flat toolbox, not four separate ones, so it can freely
# mix "turn on the lights", "open notepad", "click the Save button", and
# "remind me in 10 minutes" in the same conversation without any
# subsystem needing to know the others exist.
ALL_TOOL_SCHEMA = (JARVIS_TOOL_SCHEMA + PC_TOOL_SCHEMA + VISION_TOOL_SCHEMA
                   + REMINDER_TOOL_SCHEMA + CODE_TOOL_SCHEMA
                   + CONTACT_TOOL_SCHEMA)

# _TOOL_NAME_TO_CONTROLLER_ATTR is built once at import time as an explicit
# whitelist — tool_call.function.name comes from the LLM's output, and even
# for a trusted model, dispatching via a bare getattr(some_object,
# arbitrary_string) is the kind of thing that should be validated against a
# known-safe set rather than trusted blindly. Every name here is required
# to already be a real method on ITS OWN subsystem's controller class
# (enforced by the assertions below at import time, not just documented) so
# a typo in either schema fails loudly at startup instead of silently
# 404-ing the first time the model picks that tool. Maps to the
# CONTROLLER'S ATTRIBUTE NAME on JarvisRouter ("controller" or
# "pc_controller"), not the method itself, since which object owns a given
# tool name is exactly the fact _dispatch_tool_call needs to route calls
# correctly across two different controllers.
_TOOL_NAME_TO_CONTROLLER_ATTR = {}
for _tool_entry in JARVIS_TOOL_SCHEMA:
    _name = _tool_entry["function"]["name"]
    assert hasattr(SmartRoomController, _name), (
        f"JARVIS_TOOL_SCHEMA lists '{_name}' but SmartRoomController has no "
        f"such method — schema and implementation have drifted apart."
    )
    _TOOL_NAME_TO_CONTROLLER_ATTR[_name] = "controller"
for _tool_entry in PC_TOOL_SCHEMA:
    _name = _tool_entry["function"]["name"]
    assert hasattr(PCController, _name), (
        f"PC_TOOL_SCHEMA lists '{_name}' but PCController has no such "
        f"method — schema and implementation have drifted apart."
    )
    assert _name not in _TOOL_NAME_TO_CONTROLLER_ATTR, (
        f"Tool name '{_name}' is defined in both JARVIS_TOOL_SCHEMA and "
        f"PC_TOOL_SCHEMA — tool names must be unique across every "
        f"subsystem, since the LLM sees one flat namespace."
    )
    _TOOL_NAME_TO_CONTROLLER_ATTR[_name] = "pc_controller"
for _tool_entry in VISION_TOOL_SCHEMA:
    _name = _tool_entry["function"]["name"]
    assert hasattr(VisionController, _name), (
        f"VISION_TOOL_SCHEMA lists '{_name}' but VisionController has no "
        f"such method — schema and implementation have drifted apart."
    )
    assert _name not in _TOOL_NAME_TO_CONTROLLER_ATTR, (
        f"Tool name '{_name}' is defined in more than one tool schema — "
        f"tool names must be unique across every subsystem, since the LLM "
        f"sees one flat namespace."
    )
    _TOOL_NAME_TO_CONTROLLER_ATTR[_name] = "vision_controller"
for _tool_entry in REMINDER_TOOL_SCHEMA:
    _name = _tool_entry["function"]["name"]
    assert hasattr(ReminderController, _name), (
        f"REMINDER_TOOL_SCHEMA lists '{_name}' but ReminderController has "
        f"no such method — schema and implementation have drifted apart."
    )
    assert _name not in _TOOL_NAME_TO_CONTROLLER_ATTR, (
        f"Tool name '{_name}' is defined in more than one tool schema — "
        f"tool names must be unique across every subsystem, since the LLM "
        f"sees one flat namespace."
    )
    _TOOL_NAME_TO_CONTROLLER_ATTR[_name] = "reminder_controller"
for _tool_entry in CODE_TOOL_SCHEMA:
    _name = _tool_entry["function"]["name"]
    assert hasattr(CodeController, _name), (
        f"CODE_TOOL_SCHEMA lists '{_name}' but CodeController has no such "
        f"method — schema and implementation have drifted apart."
    )
    assert _name not in _TOOL_NAME_TO_CONTROLLER_ATTR, (
        f"Tool name '{_name}' is defined in more than one tool schema — "
        f"tool names must be unique across every subsystem, since the LLM "
        f"sees one flat namespace."
    )
    _TOOL_NAME_TO_CONTROLLER_ATTR[_name] = "code_controller"
for _tool_entry in CONTACT_TOOL_SCHEMA:
    _name = _tool_entry["function"]["name"]
    assert hasattr(ContactController, _name), (
        f"CONTACT_TOOL_SCHEMA lists '{_name}' but ContactController has no "
        f"such method - schema and implementation have drifted apart."
    )
    assert _name not in _TOOL_NAME_TO_CONTROLLER_ATTR, (
        f"Tool name '{_name}' is defined in more than one tool schema."
    )
    _TOOL_NAME_TO_CONTROLLER_ATTR[_name] = "contact_controller"
del _tool_entry, _name

# The Gemini-shaped tool list, built from the same schema. Kept as a module
# name because the schema tests compare it against ALL_TOOL_SCHEMA.
_GEMINI_TOOLS = to_gemini_tools(ALL_TOOL_SCHEMA)

# -----------------------------------------------------------------------------
# JARVIS ROUTER
# -----------------------------------------------------------------------------
class JarvisRouter:
    """
    Dual-path router: tries the deterministic Fast Path first, and only
    falls back to the LLM-backed Smart Path if no hardware regex matches.

    One instance owns one SmartRoomController and one requests.Session for
    the LLM calls — construct it once at program start and reuse it for
    every command, same reasoning as SmartRoomController's own internal
    session (connection reuse, not re-establishing TCP per call).
    """

    def __init__(
        self,
        controller: SmartRoomController | None = None,
        pc_controller: PCController | None = None,
        vision_controller: VisionController | None = None,
        reminder_controller: ReminderController | None = None,
        lm_studio_base_url: str = LM_STUDIO_BASE_URL,
        lm_studio_model: str = LM_STUDIO_MODEL,
        lm_studio_timeout: float = LM_STUDIO_TIMEOUT_SECONDS,
    ):
        # Accepting an already-constructed controller (rather than only
        # building one internally) means you can pass in a SmartRoomController
        # you've already pointed at non-default hostnames, or a test double,
        # without this class needing extra constructor parameters just to
        # forward them through.
        #
        # AC-ONLY TESTING NOTE: with controller=None (the default), this
        # falls through to SmartRoomController()'s own defaults in
        # jarvis_hardware_api.py — currently ac_host="192.168.0.7"
        # (confirmed working) and relay_host="192.168.0.X" (placeholder,
        # not yet flashed/IP-confirmed). That means JarvisRouter() bare
        # is safe to use for AC commands right now; any relay command
        # routed through the Fast Path's INTENTS table will fail against
        # the placeholder IP until you flash the relay board and update
        # relay_host either here (SmartRoomController(relay_host="...")
        # passed in as `controller`) or in jarvis_hardware_api.py's
        # constructor default directly.
        self.controller = controller or SmartRoomController()

        # Same reasoning as self.controller above — accept a pre-built
        # PCController (for tests or non-default config) or construct the
        # default one, which needs no arguments at all since it talks to
        # the local OS directly rather than a networked board.
        self.pc_controller = pc_controller or PCController()

        # Same again — VisionController also needs no arguments (it talks
        # to LM Studio's local server on demand, not a networked board or
        # a pre-existing connection to hold open).
        self.vision_controller = vision_controller or VisionController()

        # ReminderController() bare works too (its own speak_fn default
        # just logs instead of actually speaking — see its docstring),
        # but jarvis_voice_assistant.py always passes its real speak_async
        # in explicitly so reminders are actually heard, not just logged.
        self.reminder_controller = reminder_controller or ReminderController()

        # Writes code with Nemotron 3 Ultra and opens it in Notepad.
        # Measured at 74s for a single paragraph, so this one ALWAYS
        # returns immediately and finishes in the background - see
        # CodeController's docstring. Given the same notify_fn as
        # reminders where one is available, so "your code is ready" is
        # actually said rather than only logged.
        self.code_controller = CodeController(
            notify_fn=getattr(self.reminder_controller, "_speak", None)
        )

        # Placing a real WhatsApp call. Like codegen, this returns
        # immediately and runs on its own thread - a call lasts up to
        # three minutes and blocking the router for that long would
        # leave Abhay standing in a silent room.
        # The phonebook, so numbers can be saved by voice.
        self.contact_controller = ContactController()

        # Separate session from the controller's — this one talks to the
        # LLM server, the controller's talks to the NodeMCUs. Keeping them
        # distinct means a problem with one connection pool (e.g. the LLM
        # server hanging) can't interfere with the hardware session's
        # retry/backoff configuration or vice versa.
        self._llm_session = requests.Session()

        # Smart Path conversation memory, shared by whichever backend is
        # in use — see citra_llm_client.ConversationMemory for the bounds.
        self.memory = ConversationMemory()

        # The Smart Path backend. Picked once, at construction, by
        # SMART_PATH_BACKEND (Gemini when a key is set, LM Studio
        # otherwise) — see that constant in citra_llm_client.py. Both
        # get the SAME tool schema and the SAME dispatch function; the
        # router stays the one owner of the controllers.
        if SMART_PATH_BACKEND == "gemini":
            self.smart_path = GeminiClient(
                self._llm_session, self.memory, self._dispatch_tool_call, ALL_TOOL_SCHEMA,
            )
        else:
            self.smart_path = LMStudioClient(
                self._llm_session, self.memory, self._dispatch_tool_call, ALL_TOOL_SCHEMA,
                base_url=lm_studio_base_url, model=lm_studio_model, timeout=lm_studio_timeout,
            )

    # =====================================================================
    # PUBLIC ENTRY POINT
    # =====================================================================
    def route(self, user_input: str) -> RouteResult:
        """
        The single method external callers use. Tries the Fast Path first;
        falls back to the Smart Path only on a genuine non-match.

        Timing is measured around EACH path individually (not one timer
        spanning both), so a Smart Path fallback's multi-second LLM latency
        never gets blamed on the Fast Path's regex matching — the two
        numbers you see in the terminal are honest about which stage of
        the pipeline actually spent the time.
        """
        if not user_input or not user_input.strip():
            return RouteResult(
                path="FAST",
                success=False,
                message="Empty input — nothing to route.",
                latency_ms=0.0,
            )

        fast_result = self._try_fast_path(user_input)
        if fast_result is not None:
            return fast_result

        # No hardware regex matched — fall back to the LLM.
        return self._try_smart_path(user_input)

    # =====================================================================
    # FAST PATH
    # =====================================================================
    def _try_fast_path(self, user_input: str) -> RouteResult | None:
        """
        The regex-to-board table lives in citra_fast_path.py; this is the
        router's decision to try it FIRST. Returns None when the text is
        not a hardware command - that is the signal route() uses to fall
        through to the Smart Path - and a RouteResult (success or not)
        when it was. See citra_fast_path.run_fast_path for the timing
        split and why "first match wins".
        """
        return run_fast_path(self.controller, user_input)


    # =====================================================================
    # SMART PATH — TOOL CALL DISPATCH
    # =====================================================================
    def _dispatch_tool_call(self, tool_call: dict) -> str:
        """
        Executes ONE tool_call the LLM emitted, against whichever
        controller owns that tool name (self.controller for smart-room
        hardware, self.pc_controller for local PC actions — see
        _TOOL_NAME_TO_CONTROLLER_ATTR's comment for why that routing
        table exists), and returns a JSON string suitable for a "tool"
        role message's content.

        Every failure mode here (unknown tool name, malformed arguments JSON,
        a raised exception from the controller method) is turned into a
        JSON error payload rather than raised — the point of a tool result
        message is to hand the LLM SOMETHING it can react to (apologize,
        try different arguments, pick a different tool), not to crash the
        Smart Path over a single bad call the way an uncaught exception
        would.
        """
        function = tool_call.get("function", {}) or {}
        tool_name = function.get("name", "")
        raw_arguments = function.get("arguments", "{}")

        if tool_name not in _TOOL_NAME_TO_CONTROLLER_ATTR:
            logger.error("LLM requested unknown tool: %r", tool_name)
            return json.dumps({"success": False, "message": f"Unknown tool '{tool_name}'."})

        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            logger.error("LLM sent malformed tool arguments for %s: %s", tool_name, exc)
            return json.dumps({
                "success": False,
                "message": f"Arguments for '{tool_name}' were not valid JSON: {exc}",
            })

        controller_obj = getattr(self, _TOOL_NAME_TO_CONTROLLER_ATTR[tool_name])
        method = getattr(controller_obj, tool_name)
        try:
            result = method(**arguments)
        except TypeError as exc:
            # Wrong argument names/types — the LLM passed something that
            # doesn't match the tool's declared parameters.
            logger.error("Tool call %s(%r) had bad arguments: %s", tool_name, arguments, exc)
            return json.dumps({"success": False, "message": f"Invalid arguments for '{tool_name}': {exc}"})
        except Exception as exc:
            logger.error("Tool call %s(%r) raised: %s", tool_name, arguments, exc)
            return json.dumps({"success": False, "message": f"'{tool_name}' failed: {exc}"})

        logger.info("Tool call dispatched: %s(%r) -> success=%s", tool_name, arguments, result.success)
        return json.dumps(result.to_dict())

    def _try_smart_path(self, user_input: str) -> RouteResult:
        """Thin wrapper so route()'s control flow reads symmetrically with
        _try_fast_path — both are `_try_<path>(user_input) -> RouteResult`.
        The backend was picked once in __init__; see citra_llm_client.py
        for what each one actually does on the wire."""
        return self.smart_path.complete(user_input)


# =============================================================================
# INTERACTIVE TEST LOOP
# =============================================================================
if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (python jarvis_router.py) — it will NOT run when some other module
    # does `from jarvis_router import JarvisRouter`.

    print("=" * 70)
    print("CITRA DUAL-PATH INTENT ROUTER — Interactive Test")
    print("=" * 70)
    if SMART_PATH_BACKEND == "gemini":
        print(f"Smart Path target: Gemini API (model: {GEMINI_MODEL})")
    else:
        print(f"Smart Path target: {LM_STUDIO_BASE_URL} (model: {LM_STUDIO_MODEL})")
        print("  (set GEMINI_API_KEY in your environment to use the faster Gemini backend)")
    print()
    print("Try hardware commands (Fast Path, should be <5ms):")
    for intent in INTENTS:
        print(f"  - {intent.description}")
    print()
    print("Try anything else (Smart Path, falls back to the local LLM):")
    print('  - "how does quantum computing work?"')
    print('  - "write a python script that reverses a string"')
    print()
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70)
    print()

    router = JarvisRouter()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        result = router.route(user_input)
        print(result.format_for_terminal())

        # Explicit latency callout, checked against the CORRECT number.
        # The <5ms target is about routing logic bypassing the LLM — that's
        # match_latency_ms (pure regex, zero I/O). dispatch_latency_ms is a
        # real network call to physical hardware and has no reason to be
        # bounded by this router; a slow Wi-Fi network or a NodeMCU that's
        # briefly busy can push it up without that meaning the ROUTER is
        # slow. Checking latency_ms (the total) against 5ms would be
        # checking the wrong thing and would fail intermittently for
        # reasons that have nothing to do with routing logic.
        if result.path == "FAST" and result.match_latency_ms is not None:
            target_met = (
                "✓ under 5ms target" if result.match_latency_ms < 5.0
                else "✗ OVER 5ms target"
            )
            print(f"  Regex match latency check: {target_met}")
            if result.dispatch_latency_ms and result.dispatch_latency_ms > 50:
                print(
                    f"  Note: hardware dispatch took {result.dispatch_latency_ms:.1f}ms — "
                    f"this is real network time to your NodeMCU, not routing overhead. "
                    f"First call to a given board is often slower (connection warm-up); "
                    f"subsequent calls should be faster."
                )
        print()
