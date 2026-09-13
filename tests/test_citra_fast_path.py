"""
citra_fast_path on its own: the matcher is pure (no I/O), the table is
first-match-wins, and every entry is well-formed. The end-to-end
behaviour through JarvisRouter.route() is in test_router_fast_path.py.
"""
from citra_fast_path import INTENTS, HardwareIntent, match_intent, run_fast_path
from citra_route_result import RouteResult


def test_match_intent_is_pure_and_returns_the_intent_and_match():
    hit = match_intent("please set the ac to 24 now")
    assert hit is not None
    intent, match = hit
    assert isinstance(intent, HardwareIntent)
    assert intent.description == "set ac to <temp>"
    assert match.group("temp") == "24"


def test_match_intent_returns_none_for_non_hardware_text():
    assert match_intent("what is the capital of France") is None
    assert match_intent("") is None


def test_every_intent_has_a_pattern_a_handler_and_a_description():
    seen = set()
    for intent in INTENTS:
        assert intent.pattern.flags & 2  # re.IGNORECASE
        assert callable(intent.handler)
        assert intent.description and intent.description not in seen
        seen.add(intent.description)


EXAMPLES = {
    "turn on light/relay/switch <1-4>": "turn on light 2",
    "turn off light/relay/switch <1-4>": "turn off switch 3",
    "status of light/relay/switch <1-4>": "status of relay 1",
    "turn off everything (all relays + AC power)": "turn off everything",
    "maximum/full/brightest lights (all 4 relays on)": "full brightness",
    "minimum/dimmest lights (drop to 1 relay on)": "dimmest lights",
    "increase/brighten the lights (one more relay on)": "increase the lights",
    "decrease/dim the lights (one fewer relay on)": "turn down the lights",
    "turn on the ac": "turn on the ac",
    "turn off the ac": "power off the ac",
    "set ac to <temp>": "set the ac to 19",
    "set ac mode to <cool|heat|fan|dry|auto>": "set ac mode to dry",
    "set ac fan (speed) to <auto|low|med|high>": "set the ac fan speed to low",
    "increase/turn up fan speed": "turn up the fan speed",
    "decrease/turn down fan speed": "decrease fan speed",
    "system health / health check": "health check",
}


def test_every_intent_has_a_documented_example_that_matches_it():
    # The description doubles as the --list-intents documentation, so
    # each one must be reachable by at least one real phrasing - and the
    # example must land on THAT intent, not an earlier one in the table.
    assert set(EXAMPLES) == {intent.description for intent in INTENTS}
    for intent in INTENTS:
        hit = match_intent(EXAMPLES[intent.description])
        assert hit is not None, intent.description
        assert hit[0] is intent, f"{EXAMPLES[intent.description]!r} matched {hit[0].description!r}"


def test_run_fast_path_returns_none_when_nothing_matches(fake_controller):
    assert run_fast_path(fake_controller, "tell me a joke") is None
    assert fake_controller.calls == []


def test_run_fast_path_reports_both_latency_phases(fake_controller):
    result = run_fast_path(fake_controller, "turn on light 1")
    assert isinstance(result, RouteResult)
    assert result.path == "FAST"
    assert result.match_latency_ms is not None and result.dispatch_latency_ms is not None
    assert result.latency_ms == result.match_latency_ms + result.dispatch_latency_ms
    assert fake_controller.calls == [("turn_on_relay", (1,))]
