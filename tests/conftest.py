"""
Shared fixtures. The whole suite runs without a single real board, model
server or microphone: every hardware or LLM boundary is replaced with a
fake that records what it was asked to do and returns a canned result.
"""
import os
import sys

import pytest

# The modules under test live at the repo root as flat files (no package),
# so make sure the root is importable no matter where pytest is invoked.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jarvis_hardware_api import HardwareResult  # noqa: E402


class FakeSmartRoomController:
    """
    Stands in for jarvis_hardware_api.SmartRoomController.

    Every method the fast-path handlers and the dashboard call is here.
    Each records a (method, args) tuple in `calls` and returns a
    successful HardwareResult whose endpoint names the method, so a test
    can assert both *that* the right board call happened and *what* it
    was given - without an ESP8266 anywhere near the test runner.
    """

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[str, tuple]] = []
        self.fail = fail
        self.relay_host = "relay.test"
        self.ac_host = "ac.test"

    def _record(self, method: str, *args) -> HardwareResult:
        self.calls.append((method, args))
        if self.fail:
            return HardwareResult(
                success=False, endpoint=method,
                message=f"{method} failed (fake)", data=None,
            )
        return HardwareResult(
            success=True, endpoint=method, message="OK",
            data={"method": method, "args": list(args)},
        )

    # -- relays ----------------------------------------------------------
    def turn_on_relay(self, relay_number: int):
        return self._record("turn_on_relay", relay_number)

    def turn_off_relay(self, relay_number: int):
        return self._record("turn_off_relay", relay_number)

    def get_relay_status(self, relay_number: int):
        return self._record("get_relay_status", relay_number)

    def get_all_relay_status(self):
        return self._record("get_all_relay_status")

    def turn_off_all_relays(self):
        return self._record("turn_off_all_relays")

    def increase_brightness(self):
        return self._record("increase_brightness")

    def decrease_brightness(self):
        return self._record("decrease_brightness")

    def set_max_brightness(self):
        return self._record("set_max_brightness")

    def set_min_brightness(self):
        return self._record("set_min_brightness")

    # -- air conditioner -------------------------------------------------
    def set_ac_power(self, state: bool):
        return self._record("set_ac_power", state)

    def set_ac_temperature(self, temp_celsius: int):
        return self._record("set_ac_temperature", temp_celsius)

    def set_ac_mode(self, mode: str):
        return self._record("set_ac_mode", mode)

    def set_ac_fan_speed(self, speed: str):
        return self._record("set_ac_fan_speed", speed)

    def increase_ac_fan_speed(self):
        return self._record("increase_ac_fan_speed")

    def decrease_ac_fan_speed(self):
        return self._record("decrease_ac_fan_speed")

    def get_ac_status(self):
        return self._record("get_ac_status")

    # -- whole room ------------------------------------------------------
    def check_system_health(self):
        return self._record("check_system_health")

    def everything_off(self):
        return self._record("everything_off")

    def board_health(self):
        return self._record("board_health")


@pytest.fixture
def fake_controller() -> FakeSmartRoomController:
    return FakeSmartRoomController()


@pytest.fixture
def failing_controller() -> FakeSmartRoomController:
    return FakeSmartRoomController(fail=True)
