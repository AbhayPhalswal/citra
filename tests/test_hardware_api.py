"""
SmartRoomController is the one place Citra talks to a NodeMCU. These
tests replace its requests.Session with a fake that scripts each board's
reply (JSON, garbage, HTTP error, timeout, refused) and pin the contract
every caller relies on: a failure is ALWAYS a HardwareResult with
success=False and an actionable message, never an exception, and a
multi-board fan-out reports partial failure honestly.
"""
import json

import pytest
import requests

from citra_devices import DeviceRegistry
from jarvis_hardware_api import HardwareResult, SmartRoomController


class FakeResponse:
    def __init__(self, status: int = 200, body=None, text: str = ""):
        self.status_code = status
        self._body = body
        self.text = text if text else (json.dumps(body) if body is not None else "")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        if self._body is None:
            raise json.JSONDecodeError("no json", self.text, 0)
        return self._body


class FakeSession:
    """
    Scripts what each URL returns. A value can be a FakeResponse or an
    exception instance to raise. Unscripted URLs answer {"ok": true}.
    Records every (url, params) so a test can assert on the exact
    endpoint and query string that would have hit the board.
    """

    def __init__(self):
        self.script: dict = {}
        self.requests: list = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, params))
        reply = self.script.get(url, FakeResponse(200, {"ok": True}))
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def mount(self, prefix, adapter):
        pass


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def controller(session) -> SmartRoomController:
    c = SmartRoomController(relay_host="relay.test", ac_host="ac.test", timeout=1.0)
    c.session = session
    return c


# --------------------------------------------------------------------------
# HardwareResult
# --------------------------------------------------------------------------
def test_hardware_result_to_dict_is_the_llm_facing_shape():
    result = HardwareResult(success=True, endpoint="e", message="m", data={"x": 1})
    assert result.to_dict() == {"success": True, "endpoint": "e", "message": "m", "data": {"x": 1}}
    assert HardwareResult(success=False, endpoint="e", message="m").data is None


# --------------------------------------------------------------------------
# the happy path: exact URLs and query strings
# --------------------------------------------------------------------------
@pytest.mark.parametrize("call, expected", [
    (lambda c: c.turn_on_relay(2), ("http://relay.test/relay2/on", None)),
    (lambda c: c.turn_off_relay(4), ("http://relay.test/relay4/off", None)),
    (lambda c: c.get_relay_status(1), ("http://relay.test/relay1/status", None)),
    (lambda c: c.get_all_relay_status(), ("http://relay.test/status", None)),
    (lambda c: c.increase_brightness(), ("http://relay.test/brightness/up", None)),
    (lambda c: c.set_min_brightness(), ("http://relay.test/brightness/min", None)),
    (lambda c: c.set_ac_power(True), ("http://ac.test/ac/power", {"state": "on"})),
    (lambda c: c.set_ac_power(False), ("http://ac.test/ac/power", {"state": "off"})),
    (lambda c: c.set_ac_temperature(22), ("http://ac.test/ac/temp", {"val": 22})),
    (lambda c: c.set_ac_mode("COOL"), ("http://ac.test/ac/mode", {"val": "cool"})),
    (lambda c: c.set_ac_fan_speed("High"), ("http://ac.test/ac/fan", {"val": "high"})),
    (lambda c: c.increase_ac_fan_speed(), ("http://ac.test/ac/fan/up", None)),
    (lambda c: c.get_ac_status(), ("http://ac.test/ac/status", None)),
])
def test_each_method_hits_the_firmware_route_it_documents(controller, session, call, expected):
    result = call(controller)
    assert result.success is True
    assert result.data == {"ok": True}
    assert session.requests == [expected]


def test_relay_numbers_are_validated_before_the_network(controller, session):
    for bad in (0, 5, -1, 99):
        result = controller.turn_on_relay(bad)
        assert result.success is False
        assert result.endpoint == "(validation)"
        assert "1-4" in result.message
    assert session.requests == []


# --------------------------------------------------------------------------
# failure modes - every one is a HardwareResult, never an exception
# --------------------------------------------------------------------------
def test_timeout_is_reported_as_offline(controller, session):
    session.script["http://relay.test/relay1/on"] = requests.exceptions.Timeout()
    result = controller.turn_on_relay(1)
    assert result.success is False
    assert "Timed out after 1.0s" in result.message
    assert "offline" in result.message


def test_connection_refused_is_distinguished_from_timeout(controller, session):
    session.script["http://ac.test/ac/status"] = requests.exceptions.ConnectionError("refused")
    result = controller.get_ac_status()
    assert result.success is False
    assert "Connection failed" in result.message
    assert "refused" in result.message


def test_http_error_surfaces_the_boards_own_error_text(controller, session):
    session.script["http://ac.test/ac/temp"] = FakeResponse(400, {"error": "temp out of range"})
    result = controller.set_ac_temperature(99)
    assert result.success is False
    assert result.message == "HTTP error 400: temp out of range"


def test_non_json_body_is_a_clear_failure(controller, session):
    session.script["http://relay.test/status"] = FakeResponse(200, None, text="<html>oops</html>")
    result = controller.get_all_relay_status()
    assert result.success is False
    assert "not valid JSON" in result.message


def test_any_other_requests_error_is_caught(controller, session):
    session.script["http://relay.test/relay1/on"] = requests.exceptions.InvalidURL("bad")
    result = controller.turn_on_relay(1)
    assert result.success is False
    assert "Unexpected request failure" in result.message


# --------------------------------------------------------------------------
# fan-outs keep going and report partial failure
# --------------------------------------------------------------------------
def test_turn_off_all_relays_continues_past_a_dead_channel(controller, session):
    session.script["http://relay.test/relay2/off"] = requests.exceptions.Timeout()
    result = controller.turn_off_all_relays()

    assert result.success is False
    assert "One or more relays failed" in result.message
    assert [u for u, _ in session.requests] == [
        f"http://relay.test/relay{n}/off" for n in (1, 2, 3, 4)
    ]
    assert [r["success"] for r in result.data["results"]] == [True, False, True, True]


def test_system_health_pings_both_boards(controller, session):
    session.script["http://ac.test/"] = requests.exceptions.ConnectionError("down")
    result = controller.check_system_health()

    assert result.success is False
    assert "unreachable" in result.message
    assert result.data["relay_board"]["success"] is True
    assert result.data["ac_board"]["success"] is False


# --------------------------------------------------------------------------
# registry-aware methods
# --------------------------------------------------------------------------
@pytest.fixture
def registry() -> DeviceRegistry:
    return DeviceRegistry(
        {
            "hall-lights": {"host": "10.0.0.1", "room": "hall",
                            "switches": {"1": "main light", "2": "corner light"}},
            "hall-fans": {"host": "10.0.0.2", "room": "hall", "switches": {"1": "fan"}},
        },
        {"hall": {"host": "10.0.0.9"}},
    )


@pytest.fixture
def flat(registry, session) -> SmartRoomController:
    c = SmartRoomController(registry=registry)
    c.session = session
    return c


def test_registry_methods_refuse_clearly_without_a_registry(controller, session):
    for result in (
        controller.set_switch_by_name("hall fan", True),
        controller.set_room("hall", False),
        controller.set_ac_power_in("hall", True),
        controller.set_ac_temperature_in("hall", 24),
        controller.everything_off(),
        controller.board_health(),
    ):
        assert result.success is False
        assert "no device registry" in result.message
    assert session.requests == []


def test_set_switch_by_name_talks_to_the_right_board(flat, session):
    result = flat.set_switch_by_name("turn on the hall fan", True)

    assert result.success is True
    assert result.message == "hall fan on."
    assert result.data["board"] == "hall-fans"
    assert session.requests == [("http://10.0.0.2/relay1/on", None)]


def test_set_switch_by_name_refuses_unknown_names(flat, session):
    result = flat.set_switch_by_name("garage door", True)
    assert result.success is False
    assert "garage door" in result.message
    assert session.requests == []


def test_set_room_fans_out_across_boards(flat, session):
    result = flat.set_room("hall", False)

    assert result.success is True
    assert result.message == "Everything in the hall off."
    assert sorted(u for u, _ in session.requests) == [
        "http://10.0.0.1/relay1/off",
        "http://10.0.0.1/relay2/off",
        "http://10.0.0.2/relay1/off",
    ]


def test_set_room_reports_which_switch_could_not_be_reached(flat, session):
    session.script["http://10.0.0.2/relay1/off"] = requests.exceptions.Timeout()
    result = flat.set_room("hall", False)

    assert result.success is False
    assert "2 of 3 in the hall went off" in result.message
    assert result.data["failed"] == ["hall fan"]


def test_set_room_unknown_room(flat):
    assert "don't know a room" in flat.set_room("attic", True).message


def test_ac_by_room(flat, session):
    assert flat.set_ac_power_in("hall", True).success is True
    assert flat.set_ac_temperature_in("hall", 24).success is True
    assert session.requests == [
        ("http://10.0.0.9/ac/power", {"state": "on"}),
        ("http://10.0.0.9/ac/temp", {"val": 24}),
    ]
    assert "no AC in the kitchen" in flat.set_ac_power_in("kitchen", True).message


def test_everything_off_covers_every_switch_and_every_ac(flat, session):
    result = flat.everything_off()

    assert result.success is True
    assert result.data == {"switched": 4, "failed": []}
    assert ("http://10.0.0.9/ac/power", {"state": "off"}) in session.requests
    assert len(session.requests) == 4


def test_everything_off_names_what_it_could_not_reach(flat, session):
    session.script["http://10.0.0.9/ac/power"] = requests.exceptions.Timeout()
    result = flat.everything_off()
    assert result.success is False
    assert "3 of 4 off" in result.message
    assert result.data["failed"] == ["hall AC"]


def test_board_health_says_which_board_is_down(flat, session):
    session.script["http://10.0.0.2/"] = requests.exceptions.ConnectionError("x")
    result = flat.board_health()

    assert result.success is False
    assert "2 up, 1 down" in result.message
    assert result.data["dead"] == ["hall-fans (10.0.0.2)"]
    assert "hall AC (10.0.0.9)" in result.data["alive"]
