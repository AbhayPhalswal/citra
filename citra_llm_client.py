"""
=============================================================================
CITRA — SMART PATH BACKENDS: LM STUDIO AND GEMINI
=============================================================================

Everything that speaks to a language model over HTTP. jarvis_router.py
decides WHETHER a request goes to a model (only after the Fast Path
declines); this module is HOW: the wire formats, the tool-call
round-trip loop, the retries, the error messages, and the bounded
conversation memory both backends share.

Two backends, one contract:

    client = LMStudioClient(session, memory, dispatch_tool_call, tools)
    client = GeminiClient(session, memory, dispatch_tool_call, tools, api_key)
    result = client.complete("what's the weather like")   # -> RouteResult

`dispatch_tool_call` is injected rather than imported because the thing
that owns the controllers is the router, and this module must stay
importable (and testable, with a fake session) without it.

Extracted verbatim from jarvis_router.py - the comments explaining each
wire-format decision were written against the real APIs and are kept.
=============================================================================
"""

import copy
import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable

import requests

import citra_logging
from citra_route_result import RouteResult

citra_logging.configure()
logger = logging.getLogger("citra_llm_client")


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
# CONVERSATION MEMORY OBJECT
# -----------------------------------------------------------------------------
class ConversationMemory:
    """
    The Smart Path's short-term memory, shared by whichever backend is
    in use so switching backends never loses context mid-conversation.
    Stored in the OpenAI-ish {"role", "content"} shape; GeminiClient
    converts per call.

    A deque with maxlen handles the size bound: once full, appending
    drops the oldest message automatically. expire_if_stale() handles
    the time bound - see the two constants above for why both exist.
    """

    def __init__(
        self,
        max_messages: int = CONVERSATION_HISTORY_MAX_MESSAGES,
        timeout_seconds: float = CONVERSATION_MEMORY_TIMEOUT_SECONDS,
    ):
        self.history: deque[dict] = deque(maxlen=max_messages)
        self.timeout_seconds = timeout_seconds
        self.last_time: float | None = None

    def expire_if_stale(self) -> None:
        now = time.monotonic()
        if self.last_time is not None and now - self.last_time > self.timeout_seconds:
            logger.info(
                "Conversation memory expired after %.0fs of Smart Path silence — starting fresh.",
                now - self.last_time,
            )
            self.clear()

    def remember(self, prompt: str, answer: str) -> None:
        """
        Called only on a genuine success — a failed/empty/timed-out call
        has nothing useful to add, and would just teach the model to
        reference its own error message on the next turn. Only the
        ORIGINAL prompt and FINAL answer are kept, not the intermediate
        tool-call plumbing: the answer's own wording ("I've set the AC
        to 20") already carries forward what happened.
        """
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": answer})
        self.last_time = time.monotonic()

    def clear(self) -> None:
        self.history.clear()
        self.last_time = None

    def __len__(self) -> int:
        return len(self.history)


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


def to_gemini_tools(tool_schema: list[dict]) -> list[dict]:
    """The whole OpenAI-style tool list, reshaped into Gemini's one-entry
    `tools` field. Called once per GeminiClient, never per request."""
    return [{
        "functionDeclarations": [
            _to_gemini_declaration(entry["function"]) for entry in tool_schema
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
# LM STUDIO (local, OpenAI-compatible)
# -----------------------------------------------------------------------------
class LMStudioClient:
    """
    The local backend: LM Studio's OpenAI-compatible /v1/chat/completions.
    See complete()'s docstring for the exact wire shape and why it is
    NOT Ollama's.
    """

    def __init__(
        self,
        session: requests.Session,
        memory: ConversationMemory,
        dispatch_tool_call: Callable[[dict], str],
        tools: list[dict],
        base_url: str = LM_STUDIO_BASE_URL,
        model: str = LM_STUDIO_MODEL,
        timeout: float = LM_STUDIO_TIMEOUT_SECONDS,
        preflight_timeout: float = LM_STUDIO_PREFLIGHT_TIMEOUT_SECONDS,
    ):
        self.session = session
        self.memory = memory
        self.dispatch_tool_call = dispatch_tool_call
        self.tools = tools
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.preflight_timeout = preflight_timeout

    def check_model_loaded(self) -> str | None:
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
            response = self.session.get(
                f"{self.base_url}/v1/models",
                timeout=self.preflight_timeout,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            if not models:
                return (
                    f"LM Studio is running but no model is loaded. Open LM "
                    f"Studio's Server tab (or Developer tab in newer "
                    f"versions) and load '{self.model}' — the "
                    f"Chat tab's loaded model does not carry over to the "
                    f"server automatically."
                )
            return None
        except requests.exceptions.ConnectionError:
            return (
                f"Could not connect to LM Studio at {self.base_url}. "
                f"Open LM Studio, go to the Server tab (or Developer tab in "
                f"newer versions), and press 'Start Server'."
            )
        except requests.exceptions.RequestException:
            return None

    # =====================================================================
    # SMART PATH
    # =====================================================================
    def complete(self, prompt: str) -> RouteResult:
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
             self.memory.history, then the current transcribed
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
        self.memory.expire_if_stale()

        preflight_error = self.check_model_loaded()
        if preflight_error is not None:
            logger.error(preflight_error)
            return RouteResult(path="SMART", success=False, message=preflight_error, latency_ms=0.0)

        url = f"{self.base_url}/v1/chat/completions"
        working_messages = (
            [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]
            + list(self.memory.history)
            + [{"role": "user", "content": prompt}]
        )

        start = time.perf_counter()

        for _round in range(MAX_TOOL_CALL_ROUNDS):
            payload = {
                "model": self.model,
                "messages": working_messages,
                "tools": self.tools,
                "max_tokens": SMART_PATH_MAX_TOKENS,
            }

            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
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
                            f"'{self.model}' matches its identifier "
                            f"exactly (check with: curl {self.base_url}/v1/models)."
                        ),
                        latency_ms=elapsed_ms,
                    )

                message = choices[0].get("message", {}) or {}

            except requests.exceptions.Timeout:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = (
                    f"LLM request timed out after {self.timeout}s. "
                    f"The model may still be loading into memory (first call "
                    f"after loading a model is often much slower), or "
                    f"'{self.model}' may not fit comfortably on your "
                    f"hardware. Try increasing the timeout, or check the "
                    f"Server tab in LM Studio to see if generation is still "
                    f"in progress."
                )
                logger.error(msg)
                return RouteResult(path="SMART", success=False, message=msg, latency_ms=elapsed_ms)

            except requests.exceptions.ConnectionError:
                elapsed_ms = (time.perf_counter() - start) * 1000
                msg = (
                    f"Could not connect to LM Studio at {self.base_url}. "
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
                    result_json = self.dispatch_tool_call(tool_call)
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
            self.memory.remember(prompt, answer)

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


# -----------------------------------------------------------------------------
# GEMINI (cloud)
# -----------------------------------------------------------------------------
class GeminiClient:
    """
    The cloud backend: Google's generateContent REST API. Same contract
    and memory as LMStudioClient, different wire format - see complete().
    """

    def __init__(
        self,
        session: requests.Session,
        memory: ConversationMemory,
        dispatch_tool_call: Callable[[dict], str],
        tools: list[dict],
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        base_url: str = GEMINI_BASE_URL,
        timeout: float = GEMINI_TIMEOUT_SECONDS,
    ):
        self.session = session
        self.memory = memory
        self.dispatch_tool_call = dispatch_tool_call
        # Converted ONCE here, not per request - and from the SAME
        # OpenAI-style schema LM Studio gets, so the two can never drift.
        self.tools = to_gemini_tools(tools)
        self.api_key = GEMINI_API_KEY if api_key is None else api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str) -> RouteResult:
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
            OpenAI's "assistant". self.memory.history is still
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
        self.memory.expire_if_stale()

        if not self.api_key:
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

        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        working_contents = [
            {
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}],
            }
            for msg in self.memory.history
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
                "tools": self.tools,
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
                    response = self.session.post(
                        url, headers=headers, json=payload, timeout=self.timeout,
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
                msg = f"Gemini request timed out after {self.timeout}s."
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
                    result_json = self.dispatch_tool_call({
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

            self.memory.remember(prompt, answer)

            return RouteResult(path="SMART", success=True, message=answer, latency_ms=elapsed_ms)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Smart Path (Gemini) gave up after %d tool-call rounds without a final answer.", MAX_TOOL_CALL_ROUNDS)
        return RouteResult(
            path="SMART",
            success=False,
            message="Sorry, that took more steps than I'm allowed to keep trying — could you rephrase?",
            latency_ms=elapsed_ms,
        )
