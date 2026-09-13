"""
The Fast Path is the part of Citra that must work with the internet down:
a regex, then one HTTP call to a board. These tests pin every intent in
the table to the exact controller method and arguments it must produce,
against a fake controller - no board, no model, no network.
"""
import pytest

from citra_fast_path import INTENTS
from jarvis_router import JarvisRouter, RouteResult


@pytest.fixture
def router(fake_controller) -> JarvisRouter:
    return JarvisRouter(controller=fake_controller)


def test_turn_on_the_ac_hits_the_ac_board(router, fake_controller):
    result = router.route("turn on the ac")

    assert isinstance(result, RouteResult)
    assert result.path == "FAST"
    assert result.success is True
    assert fake_controller.calls == [("set_ac_power", (True,))]
    # A fast-path result reports the two phases separately, and the
    # regex phase must be sub-millisecond - that is the whole point.
    assert result.match_latency_ms is not None
    assert result.match_latency_ms < 5.0
    assert result.dispatch_latency_ms is not None


@pytest.mark.parametrize(
    "spoken, expected_call",
    [
        ("turn on light 1", ("turn_on_relay", (1,))),
        ("Turn On Relay 3", ("turn_on_relay", (3,))),
        ("turn off switch 4", ("turn_off_relay", (4,))),
        ("turn off channel2", ("turn_off_relay", (2,))),
        ("status of light 2", ("get_relay_status", (2,))),
        ("max brightness", ("set_max_brightness", ())),
        ("brightest lights", ("set_max_brightness", ())),
        ("minimum lights", ("set_min_brightness", ())),
        ("brighten the lights", ("increase_brightness", ())),
        ("turn up the brightness", ("increase_brightness", ())),
        ("dim the lights", ("decrease_brightness", ())),
        ("power on the ac", ("set_ac_power", (True,))),
        ("turn off ac", ("set_ac_power", (False,))),
        ("set the ac to 22", ("set_ac_temperature", (22,))),
        ("set ac mode to cool", ("set_ac_mode", ("cool",))),
        ("set ac fan speed to high", ("set_ac_fan_speed", ("high",))),
        ("set fan to medium", ("set_ac_fan_speed", ("med",))),
        ("increase the fan speed", ("increase_ac_fan_speed", ())),
        ("turn down the fan speed", ("decrease_ac_fan_speed", ())),
        ("system health", ("check_system_health", ())),
        ("are you online", ("check_system_health", ())),
    ],
)
def test_every_intent_maps_to_one_controller_call(router, fake_controller, spoken, expected_call):
    result = router.route(spoken)

    assert result.path == "FAST", f"{spoken!r} should not have reached the LLM"
    assert result.success is True
    assert fake_controller.calls == [expected_call]


def test_turn_off_everything_hits_both_boards_and_merges_the_result(router, fake_controller):
    # There is no single "all off" endpoint: the relay board and the AC
    # board are two different NodeMCUs, so this must be two calls, and
    # a failure on EITHER must surface rather than be swallowed.
    result = router.route("turn off everything")

    assert result.success is True
    assert fake_controller.calls == [
        ("turn_off_all_relays", ()),
        ("set_ac_power", (False,)),
    ]
    assert "Relays:" in result.message and "AC:" in result.message


def test_turn_off_everything_reports_a_partial_failure(fake_controller):
    def ac_is_down(state):
        from jarvis_hardware_api import HardwareResult
        fake_controller.calls.append(("set_ac_power", (state,)))
        return HardwareResult(success=False, endpoint="ac", message="Timed out", data=None)

    fake_controller.set_ac_power = ac_is_down
    result = JarvisRouter(controller=fake_controller).route("turn off everything")

    assert result.success is False
    assert "Timed out" in result.message


def test_ac_mode_only_accepts_the_firmware_whitelist(router, fake_controller):
    # "turbo" is not a mode the IR firmware knows, so the regex must not
    # match and the router must NOT call the board with garbage.
    assert router._try_fast_path("set ac mode to turbo") is None
    assert fake_controller.calls == []


def test_relay_numbers_outside_1_to_4_do_not_match(router, fake_controller):
    assert router._try_fast_path("turn on light 5") is None
    assert router._try_fast_path("turn on light 0") is None
    assert fake_controller.calls == []


def test_non_hardware_text_declines_the_fast_path(router, fake_controller):
    assert router._try_fast_path("how does quantum computing work") is None
    assert router._try_fast_path("write me a python script") is None
    assert fake_controller.calls == []


def test_empty_input_is_rejected_without_touching_hardware(router, fake_controller):
    for text in ("", "   ", None):
        result = router.route(text)
        assert result.path == "FAST"
        assert result.success is False
        assert "Empty" in result.message
    assert fake_controller.calls == []


def test_board_failure_is_reported_not_raised(failing_controller):
    router = JarvisRouter(controller=failing_controller)

    result = router.route("turn on light 2")

    assert result.path == "FAST"
    assert result.success is False
    assert "failed" in result.message
    assert failing_controller.calls == [("turn_on_relay", (2,))]


def test_a_handler_that_raises_becomes_a_failed_result(router, fake_controller, monkeypatch):
    def explode(controller, match):
        raise RuntimeError("boom")

    # INTENTS holds the handler function objects directly, so patching
    # the table entry is what actually changes what run_fast_path calls.
    for intent in INTENTS:
        if intent.description == "turn on the ac":
            monkeypatch.setattr(intent, "handler", explode)

    result = router.route("turn on the ac")

    assert result.path == "FAST"
    assert result.success is False
    assert "boom" in result.message
    assert fake_controller.calls == []


def test_first_match_wins_so_max_beats_brighten(router, fake_controller):
    # "max" and "brighten" both mention lights; the table order is what
    # guarantees "max brightness" is not swallowed by the generic verb.
    router.route("max brightness")
    assert fake_controller.calls == [("set_max_brightness", ())]


def test_format_for_terminal_includes_both_latencies(router):
    text = router.route("turn on the ac").format_for_terminal()
    assert "[FAST PATH | OK |" in text
    assert "regex match+extract" in text
    assert "hardware dispatch" in text
