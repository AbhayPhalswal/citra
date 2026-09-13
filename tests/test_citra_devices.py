"""
The device registry is what turns "turn off the hall" into the right set
of relays on the right boards. It has to refuse a broken config at
startup (duplicate host, duplicate label, bad channel) rather than let
the wrong switch answer, and it has to resolve spoken names sensibly.
Loads the shipped example config so the template stays valid.
"""
import json
import os

import pytest

from citra_devices import (
    CONFIG_PATH,
    MAX_CHANNEL,
    DeviceConfigError,
    DeviceRegistry,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_PATH = os.path.join(ROOT, "citra_devices.example.json")


@pytest.fixture
def registry() -> DeviceRegistry:
    return DeviceRegistry.load(EXAMPLE_PATH)


def _boards(**overrides) -> dict:
    base = {
        "hall-lights": {"host": "10.0.0.1", "room": "hall",
                        "switches": {"1": "main light", "2": "corner light"}},
        "hall-fans": {"host": "10.0.0.2", "room": "hall", "switches": {"1": "fan"}},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def test_the_shipped_example_config_loads(registry):
    assert registry.boards == ["bed1-lights", "bed1-fan", "hall-lights", "hall-fans", "kitchen"]
    assert registry.rooms == ["bedroom 1", "hall", "kitchen"]
    assert len(registry.switches) == 8
    assert [ac.room for ac in registry.air_conditioners] == ["bedroom 1"]


def test_missing_config_gives_an_actionable_error(tmp_path):
    with pytest.raises(DeviceConfigError, match="citra_devices.example.json"):
        DeviceRegistry.load(str(tmp_path / "nope.json"))


def test_invalid_json_is_reported_with_the_path(tmp_path):
    path = tmp_path / "citra_devices.json"
    path.write_text("{ nope", encoding="utf-8")
    with pytest.raises(DeviceConfigError, match="not valid JSON"):
        DeviceRegistry.load(str(path))


def test_default_config_path_lives_next_to_the_module():
    assert os.path.basename(CONFIG_PATH) == "citra_devices.json"
    assert os.path.dirname(CONFIG_PATH) == ROOT


def test_load_round_trips_a_written_config(tmp_path):
    path = tmp_path / "citra_devices.json"
    path.write_text(json.dumps({"boards": _boards(), "acs": {"hall": {"host": "10.0.0.9"}}}),
                    encoding="utf-8")
    registry = DeviceRegistry.load(str(path))
    assert registry.host_for_board("hall-fans") == "10.0.0.2"
    assert registry.ac_in("hall").host == "10.0.0.9"


# --------------------------------------------------------------------------
# validation - every one of these is a config mistake that must fail at
# startup, not at "why did the wrong light come on"
# --------------------------------------------------------------------------
def test_two_boards_on_one_host_are_refused():
    boards = _boards()
    boards["hall-fans"]["host"] = boards["hall-lights"]["host"]
    with pytest.raises(DeviceConfigError, match="both use host"):
        DeviceRegistry(boards, {})


def test_two_switches_with_the_same_label_in_one_room_are_refused():
    boards = _boards()
    boards["hall-fans"]["switches"] = {"1": "main light"}
    with pytest.raises(DeviceConfigError, match="both define"):
        DeviceRegistry(boards, {})


def test_a_board_needs_a_host_and_a_room():
    with pytest.raises(DeviceConfigError, match="no host"):
        DeviceRegistry({"x": {"room": "hall", "switches": {"1": "a"}}}, {})
    with pytest.raises(DeviceConfigError, match="no room"):
        DeviceRegistry({"x": {"host": "10.0.0.1", "switches": {"1": "a"}}}, {})


def test_channels_must_be_numeric_and_in_range():
    with pytest.raises(DeviceConfigError, match="non-numeric channel"):
        DeviceRegistry({"x": {"host": "h", "room": "r", "switches": {"one": "a"}}}, {})
    with pytest.raises(DeviceConfigError, match="outside"):
        DeviceRegistry({"x": {"host": "h", "room": "r", "switches": {str(MAX_CHANNEL + 1): "a"}}}, {})
    with pytest.raises(DeviceConfigError, match="outside"):
        DeviceRegistry({"x": {"host": "h", "room": "r", "switches": {"0": "a"}}}, {})


def test_an_ac_needs_a_host():
    with pytest.raises(DeviceConfigError, match="ac in room"):
        DeviceRegistry(_boards(), {"hall": {}})


def test_a_board_with_no_switches_is_allowed():
    registry = DeviceRegistry({"x": {"host": "h", "room": "r"}}, {})
    assert registry.switches == []
    assert registry.boards == ["x"]


# --------------------------------------------------------------------------
# lookups
# --------------------------------------------------------------------------
def test_a_room_fans_out_across_all_its_boards(registry):
    labels = sorted(s.label for s in registry.switches_in("hall"))
    assert labels == ["hall corner light", "hall fan", "hall main light"]
    assert registry.boards_in("hall") == ["hall-lights", "hall-fans"]


def test_switches_on_a_board_and_host_lookup(registry):
    assert [s.name for s in registry.switches_on("kitchen")] == ["light", "exhaust"]
    assert registry.host_for_board("kitchen") == "192.168.0.24"
    assert registry.host_for_board("no-such-board") is None


def test_ac_lookup_by_room(registry):
    assert registry.ac_in("bedroom 1").host == "192.168.0.30"
    assert registry.ac_in("kitchen") is None


# --------------------------------------------------------------------------
# spoken-name resolution
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spoken, expected_label",
    [
        ("kitchen light", "kitchen light"),             # exact
        ("Turn on the kitchen light please", "kitchen light"),  # contained
        ("hall main light", "hall main light"),         # longest label wins
        ("bedroom 1 bedside light", "bedroom 1 bedside light"),
        ("kitchen exhaust fan", "kitchen exhaust"),     # fuzzy
        ("kitchen lights", "kitchen light"),            # plural
    ],
)
def test_find_switch_resolves_natural_phrasing(registry, spoken, expected_label):
    switch = registry.find_switch(spoken)
    assert switch is not None
    assert switch.label == expected_label


def test_find_switch_returns_none_when_nothing_is_close(registry):
    assert registry.find_switch("garage door") is None
    assert registry.find_switch("") is None


def test_find_switch_prefers_the_most_specific_contained_label(registry):
    # "hall light" is not a label, but both "hall main light" and
    # "hall corner light" contain "hall" - the phrase "turn on hall main
    # light" contains exactly one label, and the longer one must win when
    # several match.
    assert registry.find_switch("hall main light and hall fan").label == "hall main light"


def test_find_room_resolves_contained_and_fuzzy(registry):
    assert registry.find_room("turn off the kitchen") == "kitchen"
    assert registry.find_room("bedroom one") == "bedroom 1"
    assert registry.find_room("") is None
    assert registry.find_room("the moon") is None


def test_describe_lists_every_board_and_ac(registry):
    text = registry.describe()
    for board in registry.boards:
        assert board in text
    assert "bedroom 1 AC" in text
    assert "(infrared)" in text
