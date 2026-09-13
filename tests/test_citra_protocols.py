"""
The spoken protocols, without a microphone or a model: individual light
targeting from free text, the state-aware lighting/cooling handlers
(what they say and which board calls they make), the time and weather
answers, the registry's shape, and SemanticRouter's pick-the-best-
protocol logic against a fake embedding model.

The handlers dispatch board calls on a background thread so the reply
can be spoken immediately; the tests join those threads before
asserting on the fake board's call log.
"""
import datetime
import sys
import threading
import types

import pytest
import requests

import citra_protocols
from citra_protocols import (
    PROTOCOL_REGISTRY,
    SEMANTIC_SIMILARITY_THRESHOLD,
    ProtocolIntent,
    SemanticRouter,
    _extract_target_relays,
    _handle_cooling_off_protocol,
    _handle_cooling_on_protocol,
    _handle_lighting_off_protocol,
    _handle_lighting_on_protocol,
    _handle_time_protocol,
    _handle_weather_protocol,
    _join_with_and,
    set_speaker,
)
from jarvis_hardware_api import HardwareResult


@pytest.fixture
def spoken():
    """Captures every reply the handlers speak, and restores the logger after."""
    said = []
    set_speaker(said.append)
    yield said
    set_speaker(citra_protocols._log_only_speaker)


def _join_dispatch_threads():
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.daemon and thread.name.startswith("Thread-"):
            thread.join(timeout=2)


class Board:
    """A fake board with real state, so the handlers' state checks mean something."""

    def __init__(self, relays=None, ac_power="off", reachable=True):
        self.relays = relays or {1: "off", 2: "off", 3: "off", 4: "off"}
        self.ac_power = ac_power
        self.reachable = reachable
        self.calls = []

    def _fail(self, endpoint):
        return HardwareResult(success=False, endpoint=endpoint, message="Timed out", data=None)

    def get_relay_status(self, n):
        if not self.reachable:
            return self._fail("relay")
        return HardwareResult(True, "relay", "OK", {"relay": n, "state": self.relays[n]})

    def turn_on_relay(self, n):
        self.calls.append(("turn_on_relay", n))
        self.relays[n] = "on"
        return HardwareResult(True, "relay", "OK", None)

    def turn_off_relay(self, n):
        self.calls.append(("turn_off_relay", n))
        self.relays[n] = "off"
        return HardwareResult(True, "relay", "OK", None)

    def get_ac_status(self):
        if not self.reachable:
            return self._fail("ac")
        return HardwareResult(True, "ac", "OK", {"power": self.ac_power})

    def set_ac_power(self, state):
        self.calls.append(("set_ac_power", state))
        self.ac_power = "on" if state else "off"
        return HardwareResult(True, "ac", "OK", None)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_join_with_and_reads_like_english():
    assert _join_with_and([2]) == "2"
    assert _join_with_and([2, 3]) == "2 and 3"
    assert _join_with_and([1, 2, 4]) == "1, 2, and 4"


@pytest.mark.parametrize("text, expected", [
    ("turn on the lights", set()),                  # nothing named -> whole room
    ("andhera ho raha hai", set()),
    ("turn on light 2", {2}),
    ("switch on relay 3", {3}),
    ("turn on light two", {2}),
    ("lights 2 and 3 please", {2, 3}),
    ("turn on light one and three", {1, 3}),
    ("turn on the warm light", {1}),
    ("turn on the white lights", {2, 3, 4}),
    ("the warm light and light 3", {1, 3}),
    ("light 9 on", set()),                          # out of range ignored
    ("turn on the DIMMEST light", {4}),
])
def test_extract_target_relays(text, expected):
    assert _extract_target_relays(text) == expected


# --------------------------------------------------------------------------
# lighting
# --------------------------------------------------------------------------
def test_lighting_on_whole_room_turns_on_only_what_is_off(spoken):
    board = Board(relays={1: "on", 2: "off", 3: "off", 4: "on"})
    reply = _handle_lighting_on_protocol(board, "turn on the lights")
    _join_dispatch_threads()

    assert reply == "Right away sir, turning on the lights."
    assert spoken == [reply]
    assert sorted(board.calls) == [("turn_on_relay", 2), ("turn_on_relay", 3)]


def test_lighting_on_already_on_says_so_and_touches_nothing(spoken):
    board = Board(relays={1: "on", 2: "on", 3: "on", 4: "on"})
    reply = _handle_lighting_on_protocol(board, "lights on")
    _join_dispatch_threads()
    assert reply == "All the lights are already on, sir."
    assert board.calls == []


def test_lighting_on_a_named_light(spoken):
    board = Board()
    reply = _handle_lighting_on_protocol(board, "turn on light 2")
    _join_dispatch_threads()
    assert reply == "Turning on light 2, sir."
    assert board.calls == [("turn_on_relay", 2)]


def test_lighting_on_several_named_lights(spoken):
    board = Board()
    reply = _handle_lighting_on_protocol(board, "turn on lights 2 and 3")
    _join_dispatch_threads()
    assert reply == "Turning on lights 2 and 3, sir."
    assert sorted(board.calls) == [("turn_on_relay", 2), ("turn_on_relay", 3)]


def test_lighting_on_named_light_already_on(spoken):
    board = Board(relays={1: "off", 2: "on", 3: "off", 4: "off"})
    assert _handle_lighting_on_protocol(board, "turn on light 2") == "That light is already on, sir."


def test_lighting_off_whole_room(spoken):
    board = Board(relays={1: "on", 2: "on", 3: "off", 4: "on"})
    reply = _handle_lighting_off_protocol(board, "lights off")
    _join_dispatch_threads()
    assert "off" in reply.lower()
    assert sorted(board.calls) == [("turn_off_relay", 1), ("turn_off_relay", 2), ("turn_off_relay", 4)]


def test_lighting_handlers_report_an_unreachable_board(spoken):
    board = Board(reachable=False)
    reply = _handle_lighting_on_protocol(board, "turn on the lights")
    assert "couldn't reach the relay board" in reply
    assert spoken == [reply]
    assert board.calls == []


# --------------------------------------------------------------------------
# cooling
# --------------------------------------------------------------------------
def test_cooling_on_and_off_check_state_first(spoken):
    board = Board(ac_power="off")
    assert _handle_cooling_on_protocol(board, "it's hot") == "Sure sir, turning on the AC."
    _join_dispatch_threads()
    assert board.calls == [("set_ac_power", True)]

    assert _handle_cooling_on_protocol(board, "ac on") == "The AC is already running, sir."
    assert _handle_cooling_off_protocol(board, "ac off") == "Turning off the AC, sir."
    _join_dispatch_threads()
    assert board.calls[-1] == ("set_ac_power", False)
    assert _handle_cooling_off_protocol(board, "ac off") == "The AC is already off, sir."


def test_cooling_handlers_report_an_unreachable_board(spoken):
    board = Board(reachable=False)
    assert "couldn't check the AC" in _handle_cooling_on_protocol(board, "ac on")
    assert "couldn't check the AC" in _handle_cooling_off_protocol(board, "ac off")
    assert board.calls == []


# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------
@pytest.fixture
def frozen_clock(monkeypatch):
    fixed = datetime.datetime(2026, 3, 4, 9, 5)  # Wednesday 09:05

    class Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(citra_protocols.datetime, "datetime", Frozen)
    return fixed


def test_time_protocol_reads_the_clock_without_leading_zeros(spoken, frozen_clock):
    assert _handle_time_protocol(Board(), "what time is it") == "It's 9:05 AM, sir."


def test_time_protocol_reads_the_date_when_asked(spoken, frozen_clock):
    assert _handle_time_protocol(Board(), "what's the date today") == "It's Wednesday, March 4, sir."


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------
class WeatherReply:
    def __init__(self, current, daily):
        self._payload = {"current": current, "daily": daily}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _weather(monkeypatch, temp=33, feels=38, code=2, rain_today=70, days=4):
    daily = {
        "time": [(datetime.date(2026, 3, 4) + datetime.timedelta(days=i)).isoformat() for i in range(days)],
        "weather_code": [code] * days,
        "temperature_2m_max": [35] * days,
        "temperature_2m_min": [24] * days,
        "precipitation_probability_max": [rain_today] + [10] * (days - 1),
    }
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return WeatherReply({"temperature_2m": temp, "apparent_temperature": feels, "weather_code": code}, daily)

    monkeypatch.setattr(citra_protocols.requests, "get", fake_get)
    return calls


def test_weather_now_leads_with_temperature_and_volunteers_rain(spoken, monkeypatch):
    calls = _weather(monkeypatch, temp=33, feels=38, rain_today=70)
    reply = _handle_weather_protocol(Board(), "what's the weather")

    assert reply.startswith("It's 33 degrees and partly cloudy in New Delhi right now, sir — feels like 38.")
    assert "70 percent chance of rain later today" in reply
    (url, params, timeout) = calls[0]
    assert url == "https://api.open-meteo.com/v1/forecast"
    assert params["forecast_days"] == 4 and timeout == 8


def test_weather_feels_like_is_omitted_when_close(spoken, monkeypatch):
    _weather(monkeypatch, temp=30, feels=31, rain_today=5)
    reply = _handle_weather_protocol(Board(), "weather")
    assert "feels like" not in reply
    assert "chance of rain" not in reply


@pytest.mark.parametrize("rain, opening", [
    (70, "Yes sir, good chance of rain today"),
    (40, "Maybe, sir"),
    (10, "No rain expected today"),
])
def test_weather_answers_a_rain_question_directly(spoken, monkeypatch, rain, opening):
    _weather(monkeypatch, rain_today=rain)
    assert _handle_weather_protocol(Board(), "do I need an umbrella").startswith(opening)


def test_weather_forecast_covers_the_next_days(spoken, monkeypatch):
    _weather(monkeypatch)
    reply = _handle_weather_protocol(Board(), "what's the forecast this week")
    assert reply.startswith("Here's the forecast for New Delhi, sir. Today, partly cloudy, high 35, low 24, 70 percent")
    assert "Tomorrow," in reply
    assert "Friday," in reply and "Saturday," in reply


def test_weather_service_down_is_a_spoken_apology(spoken, monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(citra_protocols.requests, "get", boom)
    reply = _handle_weather_protocol(Board(), "weather")
    assert "couldn't reach the weather service" in reply
    assert spoken == [reply]


# --------------------------------------------------------------------------
# the registry and the speaker
# --------------------------------------------------------------------------
def test_registry_is_well_formed():
    names = [p.name for p in PROTOCOL_REGISTRY]
    assert len(names) == len(set(names))
    for protocol in PROTOCOL_REGISTRY:
        assert isinstance(protocol, ProtocolIntent)
        assert len(protocol.example_phrases) >= 5
        assert callable(protocol.handler)
        assert all(isinstance(p, str) and p for p in protocol.example_phrases)


def test_without_a_speaker_replies_are_logged_not_lost(caplog):
    set_speaker(citra_protocols._log_only_speaker)
    with caplog.at_level("INFO", logger="citra_protocols"):
        _handle_cooling_on_protocol(Board(ac_power="on"), "ac on")
    assert any("Would have said: The AC is already running" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# SemanticRouter against a fake embedding model
# --------------------------------------------------------------------------
class _Scores:
    def __init__(self, value):
        self.value = value

    def max(self):
        return self

    def item(self):
        return self.value


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """
    A stand-in for the real package: the "embedding" of a text is just
    the text, and similarity is 1.0 when the query is one of a
    protocol's example phrases, 0.2 otherwise.
    """
    class FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, text, convert_to_tensor=False):
            return text

    def cos_sim(query, phrases):
        return _Scores(1.0 if query in phrases else 0.2)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    module.util = types.SimpleNamespace(cos_sim=cos_sim)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def test_semantic_router_picks_the_protocol_with_the_best_example(fake_sentence_transformers):
    router = SemanticRouter(PROTOCOL_REGISTRY)
    assert router.route("turn on the lights").name == "LIGHTING_ON_PROTOCOL"
    assert router.route("andhera ho raha hai").name == "LIGHTING_ON_PROTOCOL"
    assert router.route("turn off the ac").name == "COOLING_OFF_PROTOCOL"


def test_semantic_router_falls_through_below_the_threshold(fake_sentence_transformers):
    router = SemanticRouter(PROTOCOL_REGISTRY)
    assert SEMANTIC_SIMILARITY_THRESHOLD > 0.2
    assert router.route("write me a python script") is None
    assert router.route("") is None
    assert router.route("   ") is None
