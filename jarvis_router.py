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

import copy
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from re import Pattern

import requests

from citra_contacts import CONTACT_TOOL_SCHEMA, ContactController
from citra_models import CODE_TOOL_SCHEMA, CodeController
from jarvis_hardware_api import JARVIS_TOOL_SCHEMA, HardwareResult, SmartRoomController
from jarvis_pc_control import PC_TOOL_SCHEMA, PCController
from jarvis_reminders import REMINDER_TOOL_SCHEMA, ReminderController
from jarvis_vision import VISION_TOOL_SCHEMA, VisionController

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("jarvis_router")


# -----------------------------------------------------------------------------
# LLM SERVER CONFIGURATION
# -----------------------------------------------------------------------------
# CORRECTED: this router now targets LM Studio's OpenAI-compatible server,
# not Ollama. The two are NOT wire-compatible — different URL path,
# different request body shape, different response shape (see the detailed
# comment inside _query_llm below). If you ever load a different model in
# LM Studio, only LM_STUDIO_MODEL below needs to change; the
# request/response handling stays the same because LM Studio's
# OpenAI-compatible endpoint shape doesn't vary by model.
LM_STUDIO_BASE_URL = "http://localhost:1234"
LM_STUDIO_MODEL = "qwen2.5-vl-7b-instruct"  # <-- must match the model
                                              # identifier LM Studio shows
                                              # for your loaded model. Run
                                              # `curl http://localhost:1234/v1/models`
                                              # (or check the Server tab)
                                              # to confirm the exact string
                                              # — LM Studio's identifiers
                                              # don't always match a
                                              # model's display name
                                              # exactly (e.g. casing,
                                              # hyphens vs underscores).
LM_STUDIO_PREFLIGHT_TIMEOUT_SECONDS = 5.0  # /v1/models responds near-
                                    # instantly whether or not a model is
                                    # loaded (it's metadata, not
                                    # inference) — used to fail FAST with
                                    # a clear message when LM Studio is
                                    # up but has no model loaded, instead
                                    # of silently hanging for the full
                                    # LM_STUDIO_TIMEOUT_SECONDS on the
                                    # chat completions call the way a
                                    # "server running, nothing loaded"
                                    # state actually did in practice.
LM_STUDIO_TIMEOUT_SECONDS = 60.0  # local 7B-class inference on modest
                                    # hardware can genuinely take tens of
                                    # seconds for a longer answer; this
                                    # timeout is generous on purpose so a
                                    # legitimately-thinking model isn't cut
                                    # off mid-generation. Tune this down if
                                    # your hardware is consistently faster
                                    # than this ceiling implies.

# -----------------------------------------------------------------------------
# GEMINI API CONFIGURATION (cloud Smart Path backend)
# -----------------------------------------------------------------------------
# Added as a faster alternative to local LM Studio inference — a 7B-class
# model on modest local hardware is inherently slower than a cloud-hosted
# "flash" model built specifically for low latency, which is the actual
# complaint this responds to. GEMINI_API_KEY is read from the environment,
# never hardcoded here — set it in your own shell/OS environment (e.g.
# `setx GEMINI_API_KEY "..."` on Windows, then open a NEW terminal), not in
# this file, so it never ends up committed to git. Get a free key at
# https://aistudio.google.com/apikey (a Google account, no payment method
# needed for the free tier).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"  # chosen by measuring, not guessing.
                                    # An earlier value here (gemini-2.0-flash)
                                    # had been RETIRED and returned HTTP 404
                                    # "no longer available" — Google's lineup
                                    # moves fast enough that a model id which
                                    # was current when this was written can
                                    # stop existing. If Smart Path suddenly
                                    # fails with a 404 naming this constant,
                                    # that's what happened; list what your key
                                    # can actually call with:
                                    #   curl -H "x-goog-api-key: $GEMINI_API_KEY" \
                                    #     https://generativelanguage.googleapis.com/v1beta/models
                                    # and pick a current one.
                                    #
                                    # Among the candidates benchmarked against
                                    # this key, this was the fastest (~1.3s for
                                    # a short answer vs ~1.6s for the plain
                                    # flash models and ~8.7s for 3.1-flash-lite)
                                    # AND gave the tersest replies, which suits
                                    # JARVIS_SYSTEM_PROMPT's brevity rule.
                                    # Function calling is verified working on
                                    # it end-to-end against the real relay
                                    # board — always re-verify that after
                                    # changing this, since hardware control
                                    # depends on it, not just chat.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT_SECONDS = 20.0  # cloud inference over the network — much
                                    # tighter than LM_STUDIO_TIMEOUT_SECONDS's
                                    # 60s, because a reachable flash-class
                                    # cloud model answers in a couple of
                                    # seconds, not tens of seconds. A request
                                    # hanging past 20s means something is
                                    # actually wrong (network, a stalled
                                    # free-tier rate limit), not "still
                                    # thinking".
GEMINI_503_MAX_RETRIES = 2  # HTTP 503 ("model is currently experiencing
                                    # high demand") is Google's own way of
                                    # saying "transient, try again shortly"
                                    # — confirmed for real, not theoretical:
                                    # a genuine live query ("open my
                                    # downloads folder") hit exactly this
                                    # and failed outright with no retry at
                                    # all. Unlike 429 (a hard rate-limit
                                    # that retrying immediately won't fix)
                                    # a 503 spike is specifically the kind
                                    # of failure a short retry resolves —
                                    # Google's own error text says so
                                    # ("Spikes in demand are usually
                                    # temporary"). 2 retries (3 attempts
                                    # total) costs at most a few seconds in
                                    # the worst case, which is a good trade
                                    # against turning a single bad moment
                                    # into a dead command the user has to
                                    # notice and repeat themselves.
GEMINI_503_RETRY_BACKOFF_SECONDS = 1.5  # flat backoff, not exponential —
                                    # this is a short-lived overload
                                    # signal, not a sustained outage, so
                                    # there's no need for the escalating
                                    # delay a real outage would call for
                                    # (see watchdog.py's backoff for that
                                    # case instead).

SMART_PATH_BACKEND = "gemini" if GEMINI_API_KEY else "lm_studio"  # which
                                    # backend _try_smart_path calls, decided
                                    # once at import time. Defaults to
                                    # Gemini automatically the moment
                                    # GEMINI_API_KEY is set in your
                                    # environment; falls back to the local
                                    # LM Studio path when it isn't, so this
                                    # file keeps working out of the box
                                    # either way instead of hard-failing on
                                    # a missing key. Set this constant
                                    # directly (overriding the auto-pick) if
                                    # you ever want to force one backend
                                    # regardless of whether a key is present
                                    # — e.g. forcing "lm_studio" to test
                                    # offline, or if Gemini's free-tier rate
                                    # limit (a few dozen requests/minute) gets
                                    # exhausted during heavy use.

# -----------------------------------------------------------------------------
# SYSTEM PROMPT (Jarvis persona)
# -----------------------------------------------------------------------------
# Added once a genuinely natural-sounding TTS engine (Piper, replacing
# pyttsx3) was in place — a witty, conversational-pacing persona sounds
# charming through a neural voice and uncanny through a robotic one, so
# this prompt is deliberately paired with that TTS upgrade rather than
# introduced earlier. Kept as a module-level constant (not inlined in
# _query_llm) so it's easy to find and tune without digging through
# request-building code, matching how LM_STUDIO_MODEL etc. are exposed
# above.
#
# "Conversational pacing" fillers ("Hmm," "Let me see...") are requested
# explicitly here because they read naturally in synthesized SPEECH but
# would look like padding in typed text — this prompt only shapes the
# Smart Path (LLM fallback for non-hardware queries), not the Fast
# Path's deterministic regex responses or the state-aware protocol
# handlers' fixed sentences, both of which stay exactly as scripted.
JARVIS_SYSTEM_PROMPT = (
    "You are Citra, a witty and highly capable AI assistant for a smart "
    "room system. Your responses are spoken aloud through a natural "
    "text-to-speech voice. "
    "Keep answers MEDIUM length — usually two to four sentences. Enough to "
    "actually explain something or give real context, but never a lecture. "
    "If a question genuinely only needs one line, give one line; don't pad "
    "it out to hit a length. Skip preambles and filler pacing words like "
    "'Hmm' or 'Let me see...', and don't restate the question back — start "
    "with the answer. Remember it's being spoken aloud, so avoid lists, "
    "headings and anything that only works visually; say it the way you'd "
    "say it out loud. Wit is welcome when it fits naturally. "
    "You have real tools to control the room's lights and air conditioner "
    "directly — when the user's request calls for actually changing a "
    "physical device's state (turning something on/off, changing the AC's "
    "temperature or mode, checking what's currently on), call the "
    "appropriate tool rather than just describing what you would do. "
    "After a tool call's result comes back, confirm what actually happened "
    "briefly — a hardware confirmation genuinely is a one-liner, even "
    "though normal answers run longer."
)

# -----------------------------------------------------------------------------
# SMART PATH CONVERSATION MEMORY
# -----------------------------------------------------------------------------
# Short-term multi-turn memory for the Smart Path only — the Fast Path stays
# stateless (a regex match doesn't need history). Without this, every Smart
# Path call was a fresh, context-free request: "what's the capital of
# France" -> "and its population?" would fail because the LLM never saw the
# first question. Kept deliberately SHORT (a handful of exchanges, not an
# unbounded log) for two reasons: (1) latency — every stored message is
# tokens LM Studio has to re-process on every subsequent call, and this
# router's whole reason for existing is keeping the fast stuff fast; (2) a
# long-lived assistant hearing commands across an entire day should NOT be
# reasoning about something asked hours ago as if it were just said.
CONVERSATION_HISTORY_MAX_MESSAGES = 12  # 6 user/assistant exchange pairs
CONVERSATION_MEMORY_TIMEOUT_SECONDS = 300.0  # 5 minutes of Smart Path
                                    # silence clears history — a follow-up
                                    # after a long gap is more likely a new
                                    # topic than a continuation, and this
                                    # keeps stale context from leaking into
                                    # an unrelated later conversation.

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

# Gemini's function-calling wire format differs from LM Studio's OpenAI-
# compatible one: no {"type": "function", "function": {...}} wrapper, just
# the inner {name, description, parameters} directly, grouped under a
# single {"functionDeclarations": [...]} entry rather than one dict per
# tool. Converted once here (not duplicated as a second hand-authored
# schema) so JARVIS_TOOL_SCHEMA in jarvis_hardware_api.py stays the ONE
# place tool definitions are written — this just reshapes it, it doesn't
# redefine it, so the two schemas can never silently drift apart.
def _to_gemini_declaration(function_entry: dict) -> dict:
    """
    Reshapes one OpenAI-style function entry into Gemini's dialect.

    The one real incompatibility (found by actually calling the API, which
    rejected the entire request with HTTP 400): Gemini permits `enum` ONLY
    on STRING-typed properties, while OpenAI/LM Studio happily accepts it
    on integers too. Three tools here declare `"type": "integer"` with
    `"enum": [1, 2, 3, 4]` for the relay channel, and their presence made
    Gemini reject EVERY request — not just calls to those three tools —
    because the tool list is validated as a whole.

    Rather than weaken the shared schema for every backend, the integer
    enum is dropped on the way out to Gemini only. Nothing is really lost:
    each of those parameters already documents its range in its
    description ("Which relay channel to turn on (1-4)."), and an
    out-of-range value still can't reach the hardware — _dispatch_tool_call
    turns a bad argument into a JSON error payload the model can read and
    retry from.
    """
    declaration = copy.deepcopy(function_entry)
    properties = (declaration.get("parameters") or {}).get("properties") or {}
    for prop in properties.values():
        if "enum" in prop and prop.get("type") != "string":
            prop.pop("enum")
    return declaration


_GEMINI_TOOLS = [{
    "functionDeclarations": [
        _to_gemini_declaration(entry["function"]) for entry in ALL_TOOL_SCHEMA
    ]
}]

# WHY THIS EXISTS: measured live, repeatedly, that Gemini sometimes
# narrates a code-writing action in plain text instead of actually
# calling write_python_code — a reply like "I'm writing that script now
# and opening it in Notepad" that reads exactly like success but
# dispatched nothing. Confirmed live: retrying the identical prompt fresh
# usually works (this is the same non-determinism measured earlier for
# which model gets picked, not a bug specific to code generation), but a
# blind same-prompt retry can fail the same way twice. Forcing the model
# to call the tool on the retry — via toolConfig, not hope — is what
# _query_gemini's function_calls-empty branch below actually does.
#
# The phrase list mirrors CODE_TOOL_SCHEMA's own trigger wording
# ("write me a python script that...", "create code for...", "make a
# program to...") rather than inventing a second, divergent one — if that
# schema's wording changes, this should move with it. Deliberately a
# rough net: a false negative here just means the pre-existing bug isn't
# caught for phrasing outside it (no worse than before this fix existed);
# a false positive costs one extra forced-tool round on a request that
# didn't need one, which is cheap.
_CODE_REQUEST_VERBS = ("write", "create", "generate", "make", "build")
_CODE_REQUEST_NOUNS = ("script", "program", "code", "function", "app", "website", "tool")


def _looks_like_code_request(text: str) -> bool:
    lowered = text.lower()
    return (any(v in lowered for v in _CODE_REQUEST_VERBS)
            and any(n in lowered for n in _CODE_REQUEST_NOUNS))


MAX_TOOL_CALL_ROUNDS = 3  # how many request/dispatch round-trips a single
                                    # Smart Path call may take before giving
                                    # up and returning whatever text answer
                                    # (if any) the model has produced. Bounds
                                    # worst-case latency and guards against a
                                    # model stuck in a tool-call loop instead
                                    # of ever producing a final answer —
                                    # 3 rounds comfortably covers "call one
                                    # tool, see the result, answer" and even
                                    # "call two tools in sequence, then
                                    # answer" without letting a confused model
                                    # spin indefinitely.

SMART_PATH_MAX_TOKENS = 180  # backstop on response length, on top of
                                    # JARVIS_SYSTEM_PROMPT's length guidance —
                                    # a prompt instruction is a request the
                                    # model can ignore, this one is enforced
                                    # by the server cutting generation off.
                                    # ~180 tokens is roughly 4-5 spoken
                                    # sentences, sized to sit just ABOVE the
                                    # prompt's "two to four sentences" target
                                    # so it acts as a runaway guard rather
                                    # than routinely truncating normal
                                    # answers. That distinction matters: a
                                    # cap that bites often produces replies
                                    # chopped off mid-sentence, which sounds
                                    # far worse spoken aloud than a slightly
                                    # long answer. Raised from 60, which was
                                    # tuned for a much stricter one-sentence
                                    # rule and cut medium answers short.


# -----------------------------------------------------------------------------
# RESULT OBJECT
# -----------------------------------------------------------------------------
@dataclass
class RouteResult:
    """
    Uniform return type for both paths, so __main__ doesn't need to know
    which path produced an answer to display it consistently — it just
    reads .path, .success, .message, and .latency_ms off whatever came back.

    match_latency_ms / dispatch_latency_ms are populated ONLY for FAST path
    results (None on SMART path results, where the distinction doesn't
    apply — there's no separate "match" phase, just one LLM call). See the
    detailed comment in JarvisRouter._try_fast_path for why these two are
    kept separate rather than folded into one number.
    """
    path: str            # "FAST" or "SMART"
    success: bool
    message: str
    latency_ms: float
    match_latency_ms: float | None = None
    dispatch_latency_ms: float | None = None

    def format_for_terminal(self) -> str:
        """Human-readable one-block summary for the interactive test loop."""
        status = "OK" if self.success else "FAILED"
        header = f"[{self.path} PATH | {status} | {self.latency_ms:.2f}ms total]"
        if self.path == "FAST" and self.match_latency_ms is not None:
            header += (
                f"\n  regex match+extract: {self.match_latency_ms:.4f}ms"
                f"  |  hardware dispatch: {self.dispatch_latency_ms:.4f}ms"
            )
        return f"{header}\n{self.message}"


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

        self.lm_studio_base_url = lm_studio_base_url.rstrip("/")
        self.lm_studio_model = lm_studio_model
        self.lm_studio_timeout = lm_studio_timeout

        # Separate session from the controller's — this one talks to the
        # LLM server, the controller's talks to the NodeMCUs. Keeping them
        # distinct means a problem with one connection pool (e.g. the LLM
        # server hanging) can't interfere with the hardware session's
        # retry/backoff configuration or vice versa.
        self._llm_session = requests.Session()

        # Smart Path conversation memory — see CONVERSATION_HISTORY_MAX_MESSAGES's
        # comment above for why this exists and why it's bounded. A deque
        # with maxlen handles the bound for us: once full, appending drops
        # the oldest message automatically, no manual trimming needed.
        self._conversation_history: deque[dict] = deque(maxlen=CONVERSATION_HISTORY_MAX_MESSAGES)
        self._last_smart_path_time: float | None = None

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
        Walks the compiled INTENTS table in order and returns a RouteResult
        on the FIRST match. Returns None (not a RouteResult) if nothing
        matches, which is the signal route() uses to fall through to the
        Smart Path — None here specifically means "fast path declines to
        handle this," distinct from a HardwareResult failure, which means
        "fast path handled it but the hardware call itself failed."

        WHY "FIRST match wins" AND NOT "check for ambiguous multiple
        matches": with 9 patterns targeting genuinely distinct phrasings
        (different verbs, different keywords), the realistic risk of two
        patterns BOTH matching the same input is low, and optimizing for
        that edge case would mean running every pattern against every
        input even after finding a hit — directly working against the
        <5ms target this path exists for. If you add many more intents
        later and start seeing real overlap, that's the point to revisit
        this trade-off, not before.

        !!! IMPORTANT — WHAT THE <5MS TARGET ACTUALLY COVERS !!!
        This method times TWO distinct phases separately and reports both:
          1. match_ms   — regex matching + group extraction ONLY. This is
             the part that "bypasses the LLM," and it genuinely runs in
             low single-digit MICROSECONDS (measured ~0.001ms on this
             machine) — regex against ~9 compiled patterns has no I/O and
             no reason to ever approach 5ms.
          2. dispatch_ms — the SmartRoomController method call, which
             sends a real HTTP request over Wi-Fi to a physical NodeMCU.
             This is genuine network I/O to real hardware and CANNOT be
             bounded by this router's code — it depends on your Wi-Fi,
             the NodeMCU's response time, and (on a cold requests.Session)
             a one-time connection-pool warm-up cost on the very first
             call to a given host, which testing showed adds several
             milliseconds only once per host, not per call.

        Conflating these two into one number would misrepresent what
        "the Fast Path bypasses the LLM" actually guarantees: it guarantees
        (1), not (2). A slow or offline NodeMCU can make total latency
        exceed 5ms without that meaning the routing logic itself is slow —
        those are different failures with different fixes (check your
        Wi-Fi/board vs. check your regex).
        """
        match_start = time.perf_counter()
        matched_intent = None
        matched_group = None
        for intent in INTENTS:
            match = intent.pattern.search(user_input)
            if match:
                matched_intent = intent
                matched_group = match
                break
        match_ms = (time.perf_counter() - match_start) * 1000

        if matched_intent is None:
            # No pattern matched at all — not an error, just "not a
            # hardware command." Caller falls through to the Smart Path.
            return None

        assert matched_group is not None, "matched_group must be set when matched_intent is not None"

        dispatch_start = time.perf_counter()
        try:
            hw_result: HardwareResult = matched_intent.handler(self.controller, matched_group)
        except Exception as exc:
            # A regex matched, but the handler itself blew up in a way
            # SmartRoomController's own error handling didn't already
            # catch (e.g. a programming error in a handler function, not
            # a network failure — those are already
            # HardwareResult(success=False, ...) from
            # jarvis_hardware_api.py). This is a genuine last-resort catch
            # so a bug in ONE handler can't crash the whole interactive
            # loop.
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

    # =====================================================================
    # SMART PATH — PREFLIGHT CHECK
    # =====================================================================
    def _check_model_loaded(self) -> str | None:
        """
        Quick check against LM Studio's /v1/models before committing to a
        full chat completions call. That endpoint is metadata-only and
        responds near-instantly regardless of whether a model is loaded —
        unlike /v1/chat/completions, which was observed hanging for the
        FULL LM_STUDIO_TIMEOUT_SECONDS when LM Studio was up (TCP
        connection succeeds) but had no model loaded, giving no useful
        feedback for tens of seconds on something checkable in
        milliseconds.

        Returns None if a model appears loaded (proceed normally), or a
        ready-to-speak/log error message if not. Any failure of the
        preflight check ITSELF (not a clean "no models" response) returns
        None rather than blocking the real call — this is a fast-fail
        optimization, not a replacement for _query_llm's own error
        handling below, so when in doubt it gets out of the way.
        """
        try:
            response = self._llm_session.get(
                f"{self.lm_studio_base_url}/v1/models",
                timeout=LM_STUDIO_PREFLIGHT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            if not models:
                return (
                    f"LM Studio is running but no model is loaded. Open LM "
                    f"Studio's Server tab (or Developer tab in newer "
                    f"versions) and load '{self.lm_studio_model}' — the "
                    f"Chat tab's loaded model does not carry over to the "
                    f"server automatically."
                )
            return None
        except requests.exceptions.ConnectionError:
            return (
                f"Could not connect to LM Studio at {self.lm_studio_base_url}. "
                f"Open LM Studio, go to the Server tab (or Developer tab in "
                f"newer versions), and press 'Start Server'."
            )
        except requests.exceptions.RequestException:
            return None

    # =====================================================================
    # SMART PATH
    # =====================================================================
    def _query_llm(self, prompt: str) -> RouteResult:
        """
        Sends `prompt` to a local LM Studio server's OpenAI-compatible
        /v1/chat/completions endpoint and returns its response wrapped in
        a RouteResult.

        REQUEST/RESPONSE SHAPE — verified against LM Studio's actual API
        docs (this is the OpenAI Chat Completions shape, NOT Ollama's
        native /api/generate format — the two are not wire-compatible):
          Request body:  {"model": ..., "messages": [{"role": "user",
                          "content": ...}], "tools": [...], "temperature": ...}
          Response body: {"choices": [{"message": {"role": "assistant",
                          "content": "...", "tool_calls": [...]}, ...}], ...}

        TOOL CALLING: `tools=JARVIS_TOOL_SCHEMA` is sent on every request, so
        the model CAN act on a follow-up like "make it a bit warmer" instead
        of only being able to talk about it — previously the Smart Path was
        text-only even with conversation memory giving it the context to
        know what to change. When the model responds with `tool_calls`
        instead of (or alongside) text, this method dispatches each call via
        _dispatch_tool_call, appends the results as "tool" role messages,
        and sends another request so the model can produce a final answer
        that reflects what actually happened. Bounded to
        MAX_TOOL_CALL_ROUNDS round-trips — see that constant's comment.

        Three shape differences from the Ollama version this replaced, both
        of which matter for getting this right:
          1. The prompt goes in a `messages` array of role/content pairs,
             not a bare `prompt` string. We send JARVIS_SYSTEM_PROMPT as a
             system-role message, then any remembered exchanges from
             self._conversation_history, then the current transcribed
             text as the final user-role message — see
             CONVERSATION_HISTORY_MAX_MESSAGES's comment above for how
             that history is bounded and expired.
          2. The answer text is nested at choices[0].message.content, not
             a top-level `response` key. LM Studio's server does NOT
             stream-by-default the way Ollama's does (there's no
             "stream": false footgun to worry about here), but the
             response is still a nested structure we have to index into
             correctly rather than read off the top level.
          3. A tool-calling turn is message.tool_calls, a list of
             {id, type, function: {name, arguments}} — arguments is a JSON
             *string*, not a nested object, per the OpenAI wire format.
        """
        now = time.monotonic()
        if (
            self._last_smart_path_time is not None
            and now - self._last_smart_path_time > CONVERSATION_MEMORY_TIMEOUT_SECONDS
        ):
            logger.info(
                "Conversation memory expired after %.0fs of Smart Path silence — starting fresh.",
                now - self._last_smart_path_time,
            )
            self._conversation_history.clear()

        preflight_error = self._check_model_loaded()
        if preflight_error is not None:
            logger.error(preflight_error)
            return RouteResult(path="SMART", success=False, message=preflight_error, latency_ms=0.0)

        url = f"{self.lm_studio_base_url}/v1/chat/completions"
        working_messages = (
            [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]
            + list(self._conversation_history)
            + [{"role": "user", "content": prompt}]
        )

        start = time.perf_counter()

        for _round in range(MAX_TOOL_CALL_ROUNDS):
            payload = {
                "model": self.lm_studio_model,
                "messages": working_messages,
                "tools": ALL_TOOL_SCHEMA,
                "max_tokens": SMART_PATH_MAX_TOKENS,
            }

            try:
                response = self._llm_session.post(url, json=payload, timeout=self.lm_studio_timeout)
                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError:
                    # Response wasn't valid JSON. Less likely here than it was
                    # with Ollama's streaming default (LM Studio's chat
                    # completions endpoint returns one JSON object per call,
                    # not NDJSON fragments), but a malformed proxy or an LM
                    # Studio version mismatch could still produce this.
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return RouteResult(
                        path="SMART",
                        success=False,
                        message="LLM server returned non-JSON output. Check your LM Studio version.",
                        latency_ms=elapsed_ms,
                    )

                # Defensive extraction: choices[0].message, but guard each
                # level so a malformed or unexpected response shape (e.g.
                # an error payload that still returned HTTP 200, which LM
                # Studio does for some error conditions) produces a clear
                # message instead of an unhandled KeyError/IndexError bubbling
                # out of this method.
                choices = data.get("choices") or []
                if not choices:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    # LM Studio returns HTTP 200 with an "error" field (rather
                    # than a proper error status) for some failure modes, most
                    # commonly "no model loaded" — the model shown in the Chat
                    # tab does NOT automatically carry over to the server, so
                    # this is a realistic thing to hit even when LM Studio
                    # itself is running fine.
                    server_error = data.get("error")
                    detail = f" Server said: {server_error}" if server_error else ""
                    return RouteResult(
                        path="SMART",
                        success=False,
                        message=(
                            f"LLM server returned no choices.{detail} "
                            f"Confirm a model is loaded on the Server tab in "
                            f"LM Studio (the Chat tab's loaded model does not "
                            f"carry over automatically), and that "
                            f"'{self.lm_studio_model}' matches its identifier "
                            f"exactly (check with: curl {self.lm_studio_base_url}/v1/models)."
                        ),
                        latency_ms=elapsed_ms,
                    )

                message = choices[0].get("message", {}) or {}

            except requests.exceptions.Timeout:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = (
                    f"LLM request timed out after {self.lm_studio_timeout}s. "
                    f"The model may still be loading into memory (first call "
                    f"after loading a model is often much slower), or "
                    f"'{self.lm_studio_model}' may not fit comfortably on your "
                    f"hardware. Try increasing lm_studio_timeout, or check the "
                    f"Server tab in LM Studio to see if generation is still "
                    f"in progress."
                )
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            except requests.exceptions.ConnectionError:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = (
                    f"Could not connect to LM Studio at {self.lm_studio_base_url}. "
                    f"Open LM Studio, go to the Server tab (or Developer tab in "
                    f"newer versions), and press 'Start Server'. Also confirm a "
                    f"model is loaded there — the server can be running with no "
                    f"model selected."
                )
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            except requests.exceptions.RequestException as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = f"Unexpected LLM request failure: {exc}"
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                # The next request must include the assistant's own
                # tool-call message before the tool results, or the model
                # has no idea what the results are answering.
                working_messages.append(message)
                for tool_call in tool_calls:
                    result_json = self._dispatch_tool_call(tool_call)
                    working_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": result_json,
                    })
                continue  # next round: let the model react to the results

            # No tool_calls -- this is the final answer.
            answer = (message.get("content") or "").strip()
            elapsed_ms = (time.perf_counter() - start) * 1000

            if not answer:
                return RouteResult(
                    path="SMART",
                    success=False,
                    message=(
                        "LLM server responded but returned an empty answer. "
                        "This can happen if the model hit its max token "
                        "limit before producing visible content, or "
                        "returned only a reasoning/thinking block."
                    ),
                    latency_ms=elapsed_ms,
                )

            # Remember this exchange for the NEXT Smart Path call. Appended
            # only on a genuine success — a failed/empty/timed-out call has
            # nothing useful to add to history, and would just teach the
            # LLM to reference its own error message on the next turn. Only
            # the ORIGINAL prompt and FINAL answer are kept, not the
            # intermediate tool-call/tool-result plumbing — the final
            # answer's own wording ("I've set the AC to 20") already carries
            # forward what happened, without bloating every future request
            # with this call's tool-call bookkeeping.
            self._conversation_history.append({"role": "user", "content": prompt})
            self._conversation_history.append({"role": "assistant", "content": answer})
            self._last_smart_path_time = time.monotonic()

            return RouteResult(
                path="SMART",
                success=True,
                message=answer,
                latency_ms=elapsed_ms,
            )

        # Exhausted MAX_TOOL_CALL_ROUNDS without a final text answer — the
        # model kept calling tools instead of ever wrapping up. Better to
        # surface this plainly than to silently return nothing.
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Smart Path gave up after %d tool-call rounds without a final answer.", MAX_TOOL_CALL_ROUNDS)
        return RouteResult(
            path="SMART",
            success=False,
            message="Sorry, that took more steps than I'm allowed to keep trying — could you rephrase?",
            latency_ms=elapsed_ms,
        )

    # =====================================================================
    # SMART PATH — GEMINI BACKEND
    # =====================================================================
    def _query_gemini(self, prompt: str) -> RouteResult:
        """
        Sends `prompt` to Google's Gemini API (generateContent) and returns
        its response wrapped in a RouteResult. Mirrors _query_llm's overall
        shape (conversation memory, tool-call round-trip loop, error
        handling) but speaks Gemini's REST wire format, which differs from
        LM Studio's OpenAI-compatible one in several real ways:
          - The system prompt goes in a separate top-level
            `systemInstruction` field, not a "system"-role message inside
            the messages list.
          - Turns live in `contents`, shaped {"role": "user"|"model",
            "parts": [...]} — Gemini's "model" role is the equivalent of
            OpenAI's "assistant". self._conversation_history is still
            stored in the shared OpenAI-ish {"role", "content"} shape (so
            switching SMART_PATH_BACKEND doesn't lose memory format
            compatibility) and converted to Gemini's shape per-call here.
          - A function call from the model arrives as a `functionCall`
            part inside a "model"-role content entry; the result is sent
            back as a `functionResponse` part inside a "user"-role content
            entry — Gemini has no separate "tool" role the way OpenAI does.
          - The answer text is at candidates[0].content.parts[*].text,
            concatenated (a response can legitimately have multiple text
            parts).
        """
        now = time.monotonic()
        if (
            self._last_smart_path_time is not None
            and now - self._last_smart_path_time > CONVERSATION_MEMORY_TIMEOUT_SECONDS
        ):
            logger.info(
                "Conversation memory expired after %.0fs of Smart Path silence — starting fresh.",
                now - self._last_smart_path_time,
            )
            self._conversation_history.clear()

        if not GEMINI_API_KEY:
            return RouteResult(
                path="SMART",
                success=False,
                message=(
                    "Gemini API key isn't set. Set the GEMINI_API_KEY "
                    "environment variable to your free Google AI Studio "
                    "key and restart."
                ),
                latency_ms=0.0,
            )

        url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
        headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

        working_contents = [
            {
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}],
            }
            for msg in self._conversation_history
        ] + [{"role": "user", "parts": [{"text": prompt}]}]

        start = time.perf_counter()
        response = None

        # Set for exactly one round when round 0 comes back with no tool
        # call on what looks like a code request — see the comment on
        # _looks_like_code_request. Cleared immediately below so it can
        # never apply twice, which is what keeps this bounded within
        # MAX_TOOL_CALL_ROUNDS rather than risking a retry loop.
        force_tool_next_round: str | None = None
        already_retried_for_tool = False

        for _round in range(MAX_TOOL_CALL_ROUNDS):
            payload = {
                "systemInstruction": {"parts": [{"text": JARVIS_SYSTEM_PROMPT}]},
                "contents": working_contents,
                "tools": _GEMINI_TOOLS,
                "generationConfig": {"maxOutputTokens": SMART_PATH_MAX_TOKENS},
            }
            if force_tool_next_round:
                payload["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [force_tool_next_round],
                    }
                }
                force_tool_next_round = None

            try:
                # Retries ONLY 503 ("high demand" — see
                # GEMINI_503_MAX_RETRIES's comment for why that specific
                # code, and why 429 below is deliberately NOT retried the
                # same way). This loop re-sends the exact same request; it
                # does not consume a MAX_TOOL_CALL_ROUNDS round, since a
                # 503 means the request was never actually processed at
                # all, not that a tool-calling round completed.
                for _retry in range(GEMINI_503_MAX_RETRIES + 1):
                    response = self._llm_session.post(
                        url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_SECONDS,
                    )
                    if response.status_code != 503 or _retry == GEMINI_503_MAX_RETRIES:
                        break
                    logger.warning(
                        "Gemini returned 503 (overloaded) — retrying in %.1fs (attempt %d/%d)...",
                        GEMINI_503_RETRY_BACKOFF_SECONDS, _retry + 1, GEMINI_503_MAX_RETRIES,
                    )
                    time.sleep(GEMINI_503_RETRY_BACKOFF_SECONDS)

                if response.status_code == 429:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return RouteResult(
                        path="SMART",
                        success=False,
                        message="Gemini's free tier rate limit was hit — wait a moment and try again.",
                        latency_ms=elapsed_ms,
                    )
                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    return RouteResult(
                        path="SMART",
                        success=False,
                        message="Gemini returned non-JSON output.",
                        latency_ms=elapsed_ms,
                    )

                candidates = data.get("candidates") or []
                if not candidates:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    block_reason = (data.get("promptFeedback") or {}).get("blockReason")
                    detail = f" (blocked: {block_reason})" if block_reason else ""
                    return RouteResult(
                        path="SMART",
                        success=False,
                        message=f"Gemini returned no response{detail}.",
                        latency_ms=elapsed_ms,
                    )

                content = candidates[0].get("content", {}) or {}
                parts = content.get("parts", []) or []

            except requests.exceptions.Timeout:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = f"Gemini request timed out after {GEMINI_TIMEOUT_SECONDS}s."
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            except requests.exceptions.ConnectionError:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = "Could not reach Gemini's API — check your internet connection."
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            except requests.exceptions.HTTPError:
                elapsed_ms = (time.perf_counter() - start) * 1000
                detail = ""
                try:
                    detail = f" Server said: {response.json().get('error', {}).get('message', '')}"
                except Exception:
                    pass
                msg = f"Gemini API error ({response.status_code}).{detail}"
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            except requests.exceptions.RequestException as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = f"Unexpected Gemini request failure: {exc}"
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if function_calls:
                # Same reasoning as _query_llm's tool_calls branch: the
                # model's own call turn must go back in before the results,
                # or the next request has no idea what the results answer.
                working_contents.append({"role": "model", "parts": parts})
                response_parts = []
                for fc in function_calls:
                    result_json = self._dispatch_tool_call({
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {})),
                        },
                    })
                    response_parts.append({
                        "functionResponse": {
                            "name": fc.get("name", ""),
                            "response": json.loads(result_json),
                        },
                    })
                working_contents.append({"role": "user", "parts": response_parts})
                continue  # next round: let the model react to the results

            # No functionCall parts. Usually the final answer -- except
            # when this is round 0 replying to what looks like a code
            # request: that shape (text but no tool call) is exactly what
            # "narrated it instead of doing it" looks like on the wire.
            # One forced retry, never more than one.
            if (_round == 0 and not already_retried_for_tool
                    and _looks_like_code_request(prompt)):
                already_retried_for_tool = True
                force_tool_next_round = "write_python_code"
                logger.warning(
                    "Gemini replied to a code-shaped request with no tool call "
                    "('%s') -- forcing write_python_code on retry.",
                    "".join(p.get("text", "") for p in parts).strip()[:120],
                )
                # Deliberately NOT appending the failed narration turn to
                # working_contents. It already ends on the original user
                # prompt (untouched since before round 0), which is a
                # valid request on its own — Gemini's API rejects a
                # request whose `contents` ends on a "model" turn
                # ("Requests ending with a model turn are not
                # supported."), which is exactly what appending here did
                # and how this was actually caught, live, against the
                # real API, not assumed correct from reading the docs.
                continue

            # No functionCall parts -- this is the final answer.
            answer = "".join(p.get("text", "") for p in parts).strip()
            elapsed_ms = (time.perf_counter() - start) * 1000

            if not answer:
                return RouteResult(
                    path="SMART",
                    success=False,
                    message="Gemini responded but returned an empty answer.",
                    latency_ms=elapsed_ms,
                )

            self._conversation_history.append({"role": "user", "content": prompt})
            self._conversation_history.append({"role": "assistant", "content": answer})
            self._last_smart_path_time = time.monotonic()

            return RouteResult(path="SMART", success=True, message=answer, latency_ms=elapsed_ms)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Smart Path (Gemini) gave up after %d tool-call rounds without a final answer.", MAX_TOOL_CALL_ROUNDS)
        return RouteResult(
            path="SMART",
            success=False,
            message="Sorry, that took more steps than I'm allowed to keep trying — could you rephrase?",
            latency_ms=elapsed_ms,
        )

    def _try_smart_path(self, user_input: str) -> RouteResult:
        """Thin wrapper kept separate from _query_llm/_query_gemini so
        route()'s control flow reads symmetrically with _try_fast_path —
        both are `_try_<path>(user_input) -> RouteResult`. Dispatches to
        whichever backend SMART_PATH_BACKEND names — see that constant's
        comment for how it's picked."""
        if SMART_PATH_BACKEND == "gemini":
            return self._query_gemini(user_input)
        return self._query_llm(user_input)


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
