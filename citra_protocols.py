"""
=============================================================================
CITRA — SPOKEN PROTOCOLS AND THE SEMANTIC ROUTER
=============================================================================

The state-aware spoken commands that sit between the wake word and the
LLM: lighting on/off (with individual light targeting), cooling on/off,
the time, and the weather. Each is a ProtocolIntent - a bank of example
phrases in English, Hindi and Hinglish, plus a handler that reads the
board's state, decides what to say, and says it. SemanticRouter picks
the protocol by embedding similarity, so "andhera ho raha hai" and
"it's too dark in here" land on the same handler without a regex for
every phrasing.

Extracted from jarvis_voice_assistant.py so the protocols can be read
and tested without the audio pipeline. The one thing they need from
the assistant is a voice: call set_speaker(speak_async) once at startup
and every handler speaks through it. Until then they log instead, which
is what the tests rely on.
=============================================================================
"""

import datetime
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

import requests

import citra_logging
from jarvis_hardware_api import SmartRoomController

citra_logging.configure()
logger = logging.getLogger("citra_protocols")


# -----------------------------------------------------------------------------
# THE VOICE
# -----------------------------------------------------------------------------
# Handlers speak their own reply (see ProtocolIntent's docstring for why
# the reply is ALSO returned). The assistant injects its real speak_async
# at startup; without one, replies are logged - the tests use that.
def _log_only_speaker(text: str) -> None:
    logger.info("[no speaker configured] Would have said: %s", text)


_speak: Callable[[str], object] = _log_only_speaker


def set_speaker(speak: Callable[[str], object]) -> None:
    """Route every protocol reply through `speak` (the assistant's TTS)."""
    global _speak
    _speak = speak


# -----------------------------------------------------------------------------
# INDIVIDUAL LIGHT NAMING (used by _extract_target_relays, near
# _handle_lighting_on_protocol / _handle_lighting_off_protocol)
# -----------------------------------------------------------------------------
LIGHT_NAME_ALIASES = {
    # Relay 1 = Warm (brightest), Relay 2-4 = White groups — per the
    # confirmed switchboard wiring. Update this table directly if the
    # physical wiring or your preferred names for each light change;
    # nothing else needs to change to support new names.
    "warm": {1},
    "bright": {1},
    "brightest": {1},
    "white": {2, 3, 4},
    "dim": {2, 3, 4},
    "dimmest": {4},
}

_LIGHT_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}

# -----------------------------------------------------------------------------
# SENTENCE-TRANSFORMERS (SEMANTIC INTENT ROUTING)
# -----------------------------------------------------------------------------
SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # per your
                                # instruction — this specific model is
                                # trained on 50+ languages including
                                # Hindi, and handles code-mixed
                                # (Hinglish) input reasonably well since
                                # its training data includes parallel
                                # multilingual paraphrase pairs.
SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # cosine similarity floor for a
                                # protocol to be considered "matched"
                                # rather than falling through to the
                                # Smart Path. Raised from 0.55 after
                                # observing short Whisper hallucinations
                                # ("you", "thank you") scoring high
                                # enough against protocol example phrases
                                # to falsely trigger real hardware. This
                                # is a STARTING POINT at the new value,
                                # not a final tuned constant — see the
                                # tuning note in SemanticRouter.route()
                                # below for how to adjust it further
                                # based on your own false-positive/
                                # false-negative rate.


# =============================================================================
# INTENT REGISTRY — EXTENSIBLE PATTERN (per your request)
# =============================================================================
@dataclass
class ProtocolIntent:
    """
    One entry in the intent registry: a protocol name, a bank of example
    phrases (in English/Hindi/Hinglish) that represent what this protocol
    means, and the handler function that executes it.

    WHY THIS MIRRORS HardwareIntent FROM jarvis_router.py:
    Your V1 router already established the pattern of "(matching logic,
    handler function) pairs in a list, extend by appending" — this keeps
    that same shape for semantic matching instead of regex matching, so
    the two files feel like one coherent system rather than two different
    architectural styles bolted together. Adding a new protocol later
    (your "Extra Plug" on Relay 4) means writing one handler function and
    appending one ProtocolIntent to PROTOCOL_REGISTRY below — nothing else
    in this file needs to change.

    `handler` receives the SmartRoomController and the raw transcribed
    text (in case a handler wants to extract something from it, e.g. a
    temperature value spoken in a sentence — none of the two protocols
    below need this, but the signature leaves room for a future one that
    does), and returns the sentence Jarvis should speak.

    `example_phrases` are embedded ONCE at startup (see SemanticRouter),
    not per-request — same "compile once, match many times" principle as
    the regex patterns in citra_fast_path.py's INTENTS table.
    """
    name: str
    example_phrases: list[str]
    handler: Callable[[SmartRoomController, str], str]


# -----------------------------------------------------------------------------
# PROTOCOL HANDLERS — STATE-AWARE, SEMANTIC DIRECTION
# -----------------------------------------------------------------------------
# Direction (on vs off) is now determined by WHICH protocol matched,
# not by brittle keyword parsing. The handler just checks hardware state
# to generate the correct spoken response and avoid redundant commands.

# -----------------------------------------------------------------------------
# INDIVIDUAL LIGHT TARGETING
# -----------------------------------------------------------------------------
# Extracts WHICH specific relay(s) a lighting command refers to, if any.
# This is deliberately a REGEX pass over the raw transcribed text, run
# BEFORE the handler touches hardware state — not something the semantic
# router does. SemanticRouter's job is classifying an utterance into one
# of the 4 protocol buckets (lighting-on/off, cooling-on/off) by overall
# MEANING via embedding similarity; it was never designed to and doesn't
# extract entities like "which specific light number was mentioned" from
# within an utterance it already classified. Regex is the right tool for
# that different, narrower job — pulling explicit numbers and known
# names out of text — the same way jarvis_router.py's Fast Path already
# uses regex for structured extraction (temperature values, mode names)
# rather than asking an embedding model to do it.
#
# Empty return means "no specific light was named" — callers fall back
# to acting on all 4 relays, which is the exact behavior this file had
# before individual targeting existed. That fallback is preserved
# EXACTLY, so "turn on the lights" / "andhera ho raha hai" / any ambient
# phrase with no light name in it behaves identically to before this was
# added.


def _join_with_and(numbers: list) -> str:
    """
    Formats a list of relay numbers as natural spoken English: "2" for
    one, "2 and 3" for two, "2, 3, and 4" for three or more. Used
    wherever a handler needs to speak multiple light numbers in one
    sentence, so "turning on lights 2, 3, sir" (grammatically off)
    reads as "turning on lights 2 and 3, sir" instead.
    """
    strs = [str(n) for n in numbers]
    if len(strs) == 1:
        return strs[0]
    if len(strs) == 2:
        return f"{strs[0]} and {strs[1]}"
    return f"{', '.join(strs[:-1])}, and {strs[-1]}"


def _extract_target_relays(transcribed_text: str) -> set:
    """
    Returns the set of relay numbers (1-4) explicitly referenced in
    transcribed_text — via digit ("light 2"), number word ("light two"),
    or descriptive name ("the warm light", "the white lights"). Handles
    multiple targets in one sentence ("lights 2 and 3", "the warm light
    and light 3"). Returns an empty set if no specific light is named,
    which callers treat as "act on all 4" — the pre-existing whole-room
    behavior, unchanged.
    """
    text_lower = transcribed_text.lower()
    targets = set()

    for match in re.finditer(r'\b(?:light|relay)s?\s+(\d)\b', text_lower):
        n = int(match.group(1))
        if 1 <= n <= 4:
            targets.add(n)

    for match in re.finditer(r'\band\s+(\d)\b|,\s*(\d)\b', text_lower):
        n = int(match.group(1) or match.group(2))
        if 1 <= n <= 4:
            targets.add(n)

    for word, n in _LIGHT_NUMBER_WORDS.items():
        if re.search(rf'\b(?:light|relay)s?\s+{word}\b', text_lower):
            targets.add(n)
        if re.search(rf'\band\s+{word}\b', text_lower):
            targets.add(n)

    for name, relay_set in LIGHT_NAME_ALIASES.items():
        if re.search(rf'\b{name}\b', text_lower):
            targets |= relay_set

    return targets


def _handle_lighting_on_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    target_relays = _extract_target_relays(transcribed_text)
    relay_numbers = sorted(target_relays) if target_relays else [1, 2, 3, 4]
    is_whole_room = not target_relays

    statuses = {i: controller.get_relay_status(i) for i in relay_numbers}
    if any(not s.success for s in statuses.values()):
        response = "Sorry sir, I couldn't reach the relay board. Please check its connection."
        _speak(response)
        return response

    already_on = [i for i in relay_numbers if statuses[i].data and statuses[i].data.get("state") == "on"]
    needs_on = [i for i in relay_numbers if i not in already_on]

    if not needs_on:
        response = ("All the lights are already on, sir." if is_whole_room
                    else "That light is already on, sir." if len(relay_numbers) == 1
                    else "Those lights are already on, sir.")
    else:
        if is_whole_room:
            response = "Right away sir, turning on the lights."
        elif len(needs_on) == 1:
            response = f"Turning on light {needs_on[0]}, sir."
        else:
            response = f"Turning on lights {_join_with_and(needs_on)}, sir."

        def _dispatch():
            for i in needs_on:
                controller.turn_on_relay(i)
        threading.Thread(target=_dispatch, daemon=True).start()

    _speak(response)
    return response

def _handle_lighting_off_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    target_relays = _extract_target_relays(transcribed_text)
    relay_numbers = sorted(target_relays) if target_relays else [1, 2, 3, 4]
    is_whole_room = not target_relays

    statuses = {i: controller.get_relay_status(i) for i in relay_numbers}
    if any(not s.success for s in statuses.values()):
        response = "Sorry sir, I couldn't reach the relay board. Please check its connection."
        _speak(response)
        return response

    currently_on = [i for i in relay_numbers if statuses[i].data and statuses[i].data.get("state") == "on"]

    if not currently_on:
        response = ("All the lights are already off, sir." if is_whole_room
                    else "That light is already off, sir." if len(relay_numbers) == 1
                    else "Those lights are already off, sir.")
    else:
        if is_whole_room:
            response = "Turning off all the lights, sir."
        elif len(currently_on) == 1:
            response = f"Turning off light {currently_on[0]}, sir."
        else:
            response = f"Turning off lights {_join_with_and(currently_on)}, sir."

        def _dispatch():
            for i in currently_on:
                controller.turn_off_relay(i)
        threading.Thread(target=_dispatch, daemon=True).start()

    _speak(response)
    return response

def _handle_cooling_on_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    ac_status = controller.get_ac_status()
    if not ac_status.success:
        response = "Sorry sir, I couldn't check the AC status."
        _speak(response)
        return response

    ac_on = ac_status.data and ac_status.data.get("power") == "on"
    if ac_on:
        response = "The AC is already running, sir."
    else:
        response = "Sure sir, turning on the AC."
        def _dispatch():
            controller.set_ac_power(True)
        threading.Thread(target=_dispatch, daemon=True).start()

    _speak(response)
    return response

def _handle_cooling_off_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    ac_status = controller.get_ac_status()
    if not ac_status.success:
        response = "Sorry sir, I couldn't check the AC status."
        _speak(response)
        return response

    ac_on = ac_status.data and ac_status.data.get("power") == "on"
    if not ac_on:
        response = "The AC is already off, sir."
    else:
        response = "Turning off the AC, sir."
        def _dispatch():
            controller.set_ac_power(False)
        threading.Thread(target=_dispatch, daemon=True).start()

    _speak(response)
    return response


# -----------------------------------------------------------------------------
# TIME & WEATHER — Siri-style ambient queries, no LLM round-trip needed
# -----------------------------------------------------------------------------
# Both answers are fully deterministic (a clock read or a live API call),
# so — same reasoning as lighting/cooling above — these are Fast Path
# protocols with direct handlers, not questions handed to the Smart
# Path's LLM. That means no hallucination risk on "what time is it" (an
# LLM has no way to actually know that without a tool call anyway) and
# no per-query latency or cost hit for something this simple.

_WEATHER_CODE_DESCRIPTIONS = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with frost",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with light hail",
    99: "thunderstorms with heavy hail",
}  # WMO weather interpretation codes, per Open-Meteo's own published
   # table (open-meteo.com/en/docs) — not invented here, so any code the
   # API actually returns should already have a matching entry above.

# Hardcoded to this specific installation rather than auto-detected via
# IP geolocation. That approach (this project's first version) had two
# real problems, not just theoretical ones: (1) a laptop's IP-based
# location is only as reliable as its ISP's registration data, which can
# resolve to the wrong city entirely, and (2) it was one more network
# call — and therefore one more way for weather to fail — that a fixed,
# already-known installation address doesn't need at all. Update these
# three values directly if this build of Citra is ever installed
# somewhere else; nothing else in the weather handler needs to change.
#
# Coordinates verified against OpenStreetMap's Nominatim geocoder for
# "<your area>, <your city>" (not guessed or recalled from memory),
# and cross-checked by confirming Open-Meteo's own forecast endpoint
# resolves them to the same metro Delhi timezone/region.
WEATHER_DEFAULT_LOCATION = (28.6139, 77.2090, "New Delhi")


def _handle_time_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    now = datetime.datetime.now()
    text_lower = transcribed_text.lower()

    if "date" in text_lower or "day" in text_lower or "today" in text_lower:
        # %-d (no leading zero) isn't portable to Windows' strftime, so
        # the day-of-month is formatted manually rather than relying on
        # a platform-specific format code.
        response = f"It's {now.strftime('%A')}, {now.strftime('%B')} {now.day}, sir."
    else:
        # Same portability issue as above applies to %-I for the hour —
        # strip a leading zero manually instead of using a Linux-only
        # strftime extension.
        hour_12 = now.strftime("%I").lstrip("0") or "12"
        response = f"It's {hour_12}:{now.strftime('%M %p')}, sir."

    _speak(response)
    return response


WEATHER_FORECAST_DAYS = 4  # today + 3 more. A real "full forecast" request
                                    # (as opposed to "what's the weather
                                    # right now") reasonably means several
                                    # upcoming days, but a genuine 7-day
                                    # readout is a lot to sit through
                                    # spoken aloud one day at a time — 4
                                    # days covers "the next few days" /
                                    # "this week" well enough for a spoken
                                    # answer without turning into a
                                    # monologue. Bump this if you want more
                                    # and don't mind a longer answer.


def _handle_weather_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    lat, lon, city = WEATHER_DEFAULT_LOCATION
    text_lower = transcribed_text.lower()

    # Checked before asked_about_rain: "will it rain this week" or
    # "weather forecast, is rain coming" should get the multi-day answer,
    # not the single-day one — forecast-scope words win when both are
    # present, since they're the more specific signal of what's actually
    # being asked.
    asked_for_forecast = any(
        w in text_lower for w in (
            "forecast", "this week", "next few days", "coming days",
            "upcoming days", "tomorrow", "next couple",
        )
    )
    asked_about_rain = any(
        w in text_lower for w in ("rain", "umbrella", "wet", "shower", "drizzle", "storm")
    )

    try:
        api_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,apparent_temperature",
                # "current" ONLY reflects this exact instant — the bug the
                # rain-question fix addressed: "will it rain today" got
                # answered from a snapshot that only knows whether it's
                # raining THIS SECOND, so on a merely overcast moment with
                # real rain forecast for later the same day it never
                # actually addressed rain at all. The daily block below is
                # a real FORECAST, not a snapshot, for exactly that reason
                # — and now covers WEATHER_FORECAST_DAYS days, not just
                # today, so "what's the forecast" has real multi-day data
                # to answer from instead of only ever being able to talk
                # about right now.
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": WEATHER_FORECAST_DAYS,
            },
            timeout=8,
        )
        api_response.raise_for_status()
        payload = api_response.json()
        current = payload["current"]
        daily = payload["daily"]
        rain_chance_today = round(daily["precipitation_probability_max"][0])
    except (requests.exceptions.RequestException, KeyError, ValueError, IndexError):
        response = "Sorry sir, I couldn't reach the weather service right now."
        _speak(response)
        return response

    temp = round(current["temperature_2m"])
    feels_like = round(current["apparent_temperature"])
    description = _WEATHER_CODE_DESCRIPTIONS.get(current["weather_code"], "clear")

    # "Feels like" is only worth saying when it actually diverges from the
    # air temperature — in humid climates it routinely runs several
    # degrees hotter, which is genuinely useful; when it's within a
    # couple of degrees, saying both is just redundant noise.
    feels_like_clause = f" — feels like {feels_like}" if abs(feels_like - temp) >= 3 else ""

    if asked_for_forecast:
        day_names = [datetime.date.fromisoformat(d).strftime("%A") for d in daily["time"]]
        day_parts = []
        for i in range(len(daily["time"])):
            label = "Today" if i == 0 else "Tomorrow" if i == 1 else day_names[i]
            day_desc = _WEATHER_CODE_DESCRIPTIONS.get(daily["weather_code"][i], "clear")
            hi = round(daily["temperature_2m_max"][i])
            lo = round(daily["temperature_2m_min"][i])
            rain = round(daily["precipitation_probability_max"][i])
            day_parts.append(f"{label}, {day_desc}, high {hi}, low {lo}, {rain} percent chance of rain")
        response = f"Here's the forecast for {city}, sir. " + ". ".join(day_parts) + "."

    elif asked_about_rain:
        # Directly answer the yes/no that was actually asked, THEN give
        # temperature as supporting context — not the other way around,
        # which is exactly the ordering that made the old response feel
        # like it never answered the question at all.
        if rain_chance_today >= 60:
            response = (
                f"Yes sir, good chance of rain today — about {rain_chance_today} percent. "
                f"I'd bring an umbrella. It's {temp} degrees and {description} right now."
            )
        elif rain_chance_today >= 30:
            response = (
                f"Maybe, sir — about a {rain_chance_today} percent chance of rain today. "
                f"It's {temp} degrees and {description} right now."
            )
        else:
            response = (
                f"No rain expected today, sir — only about {rain_chance_today} percent. "
                f"It's {temp} degrees and {description} right now."
            )
    else:
        response = f"It's {temp} degrees and {description} in {city} right now, sir{feels_like_clause}."
        # Volunteered even when not asked directly, the way a real weather
        # app leads with rain when it matters — matches how Siri/Google
        # Assistant answer a plain "what's the weather" query.
        if rain_chance_today >= 40:
            response += f" There's a {rain_chance_today} percent chance of rain later today."

    _speak(response)
    return response


# -----------------------------------------------------------------------------
# THE REGISTRY ITSELF
# -----------------------------------------------------------------------------
PROTOCOL_REGISTRY: list[ProtocolIntent] = [
    ProtocolIntent(
        name="LIGHTING_ON_PROTOCOL",
        example_phrases=[
            # English
            "turn on the lights", "switch on the lights", "turn the lights on",
            "put the lights on", "lights on", "illuminate the room",
            "brighten the room", "let there be light", "i can't see anything",
            "it's too dark in here", "i need some light", "enable lights",
            # Hindi/Hinglish
            "light jalao", "lights on karo", "roshni chahiye", "andhera ho raha hai",
            "andhera hai kamre mein", "bulb jalao", "light chalu karo", "ujala karo",
            "kamre ki light on karo", "light on kar do", "roshni karo",
            # Individual/named light targeting — these teach the router
            # that a light-specific command is still LIGHTING_ON_PROTOCOL;
            # WHICH light is then extracted separately by
            # _extract_target_relays(), not by this classification step.
            "turn on light 2", "turn on light one", "switch on light 3",
            "turn on the warm light", "turn on the white lights",
            "turn on lights 2 and 3", "light 4 on please",
            "can you turn on the bright light", "turn on relay 2",
            # Compound "reason + request" phrasing — a real gap found by
            # testing, not a hypothetical: a sentence embedding for a
            # two-clause utterance sits meaningfully further from EITHER
            # clause's own short example than either clause's similarity
            # to the OTHER clause's example, so "it's dark in here, turn
            # the lights on" can score below SEMANTIC_SIMILARITY_THRESHOLD
            # even though "it's dark in here" alone and "turn the lights
            # on" alone both score ~1.0. A few real compound examples
            # closes that gap directly rather than hoping the model
            # generalizes.
            "it's dark in here can you turn on the lights",
            "i can't see anything turn the lights on please",
        ],
        handler=_handle_lighting_on_protocol,
    ),
    ProtocolIntent(
        name="LIGHTING_OFF_PROTOCOL",
        example_phrases=[
            # English
            "turn off the lights", "switch off the lights", "turn the lights off",
            "kill the lights", "put the lights out", "lights off",
            "shut off the lights", "disable lights", "i want it dark",
            "make it dark", "turn off all the lights",
            # Hindi/Hinglish
            "light band karo", "lights off karo", "bulb band kar do",
            "light band kar do", "kamre ki light off karo", "light bujha do",
            "roshni band karo", "ujala band karo",
            # Individual/named light targeting — same reasoning as the ON
            # protocol above.
            "turn off light 2", "turn off light three", "switch off light 4",
            "turn off the warm light", "turn off the white lights",
            "turn off lights 2 and 3", "light 1 off please",
            "can you turn off the dim lights", "turn off relay 3",
            # Compound phrasing — see LIGHTING_ON_PROTOCOL's comment above
            # for why this is needed, not just belt-and-suspenders.
            "it's too bright in here turn off the lights",
            "i'm heading to bed turn off the lights please",
        ],
        handler=_handle_lighting_off_protocol,
    ),
    ProtocolIntent(
        name="COOLING_ON_PROTOCOL",
        example_phrases=[
            # English
            "turn on the ac", "switch on the ac", "start the ac", "power on the ac",
            "ac on", "turn the air conditioner on", "it's really hot in here",
            "i'm sweating", "cool the room down", "make it cooler", "i need some cooling",
            "turn on the cooling", "activate the ac",
            # Hindi/Hinglish
            "ac chalao", "ac on karo", "ac chalu karo", "ac on kar do",
            "garmi lag rahi hai", "garmi lagri hai yaar", "bahut garam hai",
            "thanda karo", "thand chahiye", "cooling on karo", "ac start karo",
            # Compound phrasing — measured directly: "im sweating turn on
            # the ac" alone scored 0.69 against the single-clause examples
            # above, below the 0.75 match threshold, despite each clause
            # individually scoring near 1.0. Real compound examples fix
            # that rather than relying on the model to generalize across
            # clause boundaries it wasn't shown.
            "it's really hot in here turn on the ac",
            "i'm sweating can you turn on the ac please",
        ],
        handler=_handle_cooling_on_protocol,
    ),
    ProtocolIntent(
        name="COOLING_OFF_PROTOCOL",
        example_phrases=[
            # English
            "turn off the ac", "switch off the ac", "turn the ac off",
            "stop the ac", "power off the ac", "turn the air conditioner off",
            "ac off", "shut down the ac", "deactivate the ac", "turn off the cooling",
            "stop the cooling", "i'm cold turn off the ac",
            # Hindi/Hinglish
            "ac band karo", "ac off karo", "ac band kar do", "cooling band karo",
            "thanda band karo", "ac stop karo", "ac band kar do please",
            # Compound phrasing — see COOLING_ON_PROTOCOL's comment above.
            "i'm cold now turn off the ac",
            "it's freezing in here turn the ac off",
        ],
        handler=_handle_cooling_off_protocol,
    ),
    ProtocolIntent(
        name="TIME_PROTOCOL",
        example_phrases=[
            "what time is it", "what's the time", "tell me the time",
            "do you know the time", "what time is it right now",
            "what's the current time", "what's today's date",
            "what's the date today", "what day is it today",
            "what's the date", "tell me today's date",
            "kitne baje hain", "time kya hua hai", "time batao",
            "aaj kya date hai", "aaj konsa din hai",
        ],
        handler=_handle_time_protocol,
    ),
    ProtocolIntent(
        name="WEATHER_PROTOCOL",
        example_phrases=[
            "what's the weather like", "what's the weather outside",
            "how's the weather today", "is it hot outside",
            "is it cold outside", "what's the temperature outside",
            "what's the temperature right now", "will it rain today",
            "is it going to rain", "how hot is it outside",
            "do i need an umbrella today", "what's it like outside",
            # Longer, more conversational umbrella phrasings — added after
            # a real query ("Do I have to take umbrella with me when I go
            # out?") scored 0.745 against the phrases above, JUST under
            # SEMANTIC_SIMILARITY_THRESHOLD (0.75), and fell through to
            # the Smart Path, which has no real weather data and gave a
            # deflecting non-answer. Measured directly, not guessed: these
            # five score 0.80-0.94 against that same real query (vs 0.745
            # for the shorter phrases above), comfortably clearing the
            # threshold — the gap was phrasing LENGTH/STYLE, not topic.
            "do i have to take an umbrella with me",
            "do i need to carry an umbrella when i go out",
            "will i need an umbrella if i go outside",
            "do i need to bring an umbrella",
            "should i take an umbrella with me today",
            "bahar mausam kaisa hai", "bahar garmi hai kya",
            "bahar thand hai kya", "temperature kitna hai",
            # Forecast-scope phrasing — routes to the same handler, which
            # detects the "forecast" intent itself (see
            # asked_for_forecast in _handle_weather_protocol) and answers
            # with several days instead of just right now.
            "what's the weather forecast", "give me the weather forecast",
            "what's the forecast for this week", "what's the weather like this week",
            "how's the weather looking the next few days",
            "what will the weather be like tomorrow",
            "what's tomorrow's weather", "this week ka mausam kaisa rahega",
        ],
        handler=_handle_weather_protocol,
    ),
]

    # -------------------------------------------------------------------
    # TEMPLATE FOR RELAY 4 ("EXTRA PLUG") — NOT YET ACTIVE
    # -------------------------------------------------------------------
    # This is intentionally commented out, not a real registered
    # protocol. What's plugged into Relay 4 hasn't been decided yet, so
    # there's no real device, phrase set, or handler behavior to encode
    # — inventing one here would mean guessing what a real device should
    # do, which risks producing a protocol that looks legitimate but
    # controls hardware based on a fabricated assumption.
    #
    # To activate this once you know what Relay 4 controls:
    #   1. Write a handler function above (near _handle_lighting_protocol
    #      and _handle_cooling_protocol) with the same signature and the
    #      same "check state (sequential) -> decide sentence ->
    #      speak_async() + dispatch thread (parallel)" shape.
    #   2. Uncomment the block below, rename it, fill in real example
    #      phrases (5-10, covering English/Hindi/Hinglish the way the
    #      two protocols above do), and point `handler` at your new
    #      function.
    #   3. That's it — SemanticRouter discovers and embeds every entry
    #      in PROTOCOL_REGISTRY automatically at startup. Nothing else
    #      in this file needs to change.
    #
    # ProtocolIntent(
    #     name="EXTRA_PLUG_PROTOCOL",
    #     example_phrases=[
    #         "turn on the <device>",
    #         "turn off the <device>",
    #         # ... add real phrases once the device is known
    #     ],
    #     handler=_handle_extra_plug_protocol,  # write this function first
    # ),



# =============================================================================
# SEMANTIC ROUTER
# =============================================================================
class SemanticRouter:
    """
    Embeds every protocol's example phrases once at construction time,
    then for each new transcribed utterance, embeds it and finds the
    highest-similarity protocol. If that similarity clears
    SEMANTIC_SIMILARITY_THRESHOLD, that protocol's handler runs; otherwise
    the caller should fall through to the Smart Path (LM Studio).

    WHY "highest similarity across ALL example phrases of ALL protocols",
    NOT "average similarity per protocol": a single very-close-matching
    example phrase is a stronger signal than a protocol's overall average
    — someone might phrase a lighting request in a way that's extremely
    close to ONE of your ten example phrases while being fairly distant
    from the other nine (different phrasing styles, different languages).
    Taking the max within each protocol, then the max ACROSS protocols,
    respects that a single strong match is meaningful evidence even if
    the rest of that protocol's phrase bank isn't a great fit for this
    particular utterance.
    """

    def __init__(self, registry: list[ProtocolIntent], model_name: str = SEMANTIC_MODEL_NAME):
        # Imported inside __init__, not at module level, so importing
        # jarvis_voice_assistant.py for its dataclasses/constants doesn't
        # require sentence-transformers (and the multi-GB PyTorch it pulls
        # in) to already be installed — only actually constructing a
        # SemanticRouter does.
        from sentence_transformers import SentenceTransformer, util
        self._util = util

        logger.info("Loading semantic model '%s' (first run downloads weights)...", model_name)
        self.model = SentenceTransformer(model_name)

        self.registry = registry

        # Pre-compute embeddings for every protocol's example phrases,
        # ONCE, at startup — not per-utterance. This is the direct
        # semantic-routing analog of jarvis_router.py's "compile regexes
        # once at import time" principle: the expensive part (running the
        # model) happens up front, so runtime matching is just a handful
        # of cosine similarity computations against already-computed
        # vectors.
        self._protocol_embeddings = []  # list of (protocol_index, tensor)
        for idx, intent in enumerate(registry):
            embeddings = self.model.encode(intent.example_phrases, convert_to_tensor=True)
            self._protocol_embeddings.append((idx, embeddings))
        logger.info("Semantic router ready with %d protocol(s).", len(registry))

    def route(self, transcribed_text: str) -> ProtocolIntent | None:
        """
        Returns the best-matching ProtocolIntent if its similarity clears
        the threshold, else None (signal to fall through to the Smart
        Path — same None-means-"no match" convention as
        jarvis_router.py's _try_fast_path).

        TUNING NOTE: SEMANTIC_SIMILARITY_THRESHOLD = 0.75 is a reasonable
        starting point for this multilingual MiniLM model, but the RIGHT
        value depends on your actual usage — run this with logging at
        INFO level for a few days, note the similarity scores for
        utterances that SHOULD have matched a protocol but didn't (raise
        the threshold... no, LOWER it to catch them) versus utterances
        that matched a protocol but shouldn't have (RAISE the threshold
        to be more conservative). This is a one-line constant to adjust,
        not something requiring code changes.
        """
        if not transcribed_text or not transcribed_text.strip():
            return None

        query_embedding = self.model.encode(transcribed_text, convert_to_tensor=True)

        best_score = -1.0
        best_protocol_idx: int | None = None

        for protocol_idx, phrase_embeddings in self._protocol_embeddings:
            # cos_sim(query, all_phrases_for_this_protocol) -> a 1xN
            # tensor of similarity scores; .max() gives this protocol's
            # single best-matching example phrase's score.
            scores = self._util.cos_sim(query_embedding, phrase_embeddings)
            protocol_best = scores.max().item()

            if protocol_best > best_score:
                best_score = protocol_best
                best_protocol_idx = protocol_idx

        best_matched = best_protocol_idx is not None and best_score >= SEMANTIC_SIMILARITY_THRESHOLD
        logger.info(
            "Semantic match: '%s' -> %s (score=%.3f, threshold=%.2f, matched=%s)",
            transcribed_text,
            self.registry[best_protocol_idx].name if best_protocol_idx is not None else "NONE",
            best_score,
            SEMANTIC_SIMILARITY_THRESHOLD,
            best_matched,
        )

        if best_matched:
            return self.registry[best_protocol_idx]
        return None
