"""
Smart Path tool dispatch: the LLM names a tool and hands over a JSON
string of arguments. Every way that can go wrong (unknown tool, broken
JSON, wrong argument names, a controller that raises) must come back as
a JSON error payload the model can react to - never as an exception that
would take the whole conversation down.
"""
import json

import pytest

from jarvis_router import (
    _TOOL_NAME_TO_CONTROLLER_ATTR,
    ALL_TOOL_SCHEMA,
    JarvisRouter,
    _to_gemini_declaration,
)


@pytest.fixture
def router(fake_controller) -> JarvisRouter:
    return JarvisRouter(controller=fake_controller)


def _call(name: str, arguments) -> dict:
    """Builds the tool_call shape LM Studio's OpenAI-compatible API emits."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"id": "call_1", "type": "function",
            "function": {"name": name, "arguments": arguments}}


def test_a_valid_hardware_tool_call_reaches_the_controller(router, fake_controller):
    payload = json.loads(router._dispatch_tool_call(_call("turn_on_relay", {"relay_number": 3})))

    assert payload["success"] is True
    assert payload["endpoint"] == "turn_on_relay"
    assert fake_controller.calls == [("turn_on_relay", (3,))]


def test_unknown_tool_names_are_refused_not_getattr_ed(router, fake_controller):
    payload = json.loads(router._dispatch_tool_call(_call("__class__", {})))

    assert payload["success"] is False
    assert "Unknown tool" in payload["message"]
    assert fake_controller.calls == []


def test_malformed_argument_json_is_reported(router, fake_controller):
    payload = json.loads(router._dispatch_tool_call(_call("turn_on_relay", "{not json")))

    assert payload["success"] is False
    assert "not valid JSON" in payload["message"]
    assert fake_controller.calls == []


def test_wrong_argument_names_are_reported(router, fake_controller):
    payload = json.loads(router._dispatch_tool_call(_call("turn_on_relay", {"relay": 1})))

    assert payload["success"] is False
    assert "Invalid arguments" in payload["message"]
    assert fake_controller.calls == []


def test_a_controller_exception_is_reported(router, fake_controller):
    def explode(**kwargs):
        raise RuntimeError("relay board caught fire")

    fake_controller.turn_on_relay = explode
    payload = json.loads(router._dispatch_tool_call(_call("turn_on_relay", {"relay_number": 1})))

    assert payload["success"] is False
    assert "caught fire" in payload["message"]


def test_empty_arguments_string_means_no_arguments(router, fake_controller):
    payload = json.loads(router._dispatch_tool_call(_call("get_ac_status", "")))

    assert payload["success"] is True
    assert fake_controller.calls == [("get_ac_status", ())]


def test_every_schema_tool_is_routed_to_exactly_one_controller():
    names = [entry["function"]["name"] for entry in ALL_TOOL_SCHEMA]

    # The LLM sees one flat namespace; a duplicate would be dispatched
    # to whichever subsystem registered first, silently.
    assert len(names) == len(set(names))
    assert set(names) == set(_TOOL_NAME_TO_CONTROLLER_ATTR)
    assert set(_TOOL_NAME_TO_CONTROLLER_ATTR.values()) <= {
        "controller", "pc_controller", "vision_controller",
        "reminder_controller", "code_controller", "contact_controller",
    }


def test_every_tool_name_is_a_real_method_on_its_controller(router):
    for name, attr in _TOOL_NAME_TO_CONTROLLER_ATTR.items():
        owner = getattr(router, attr)
        assert callable(getattr(owner, name, None)), f"{attr}.{name} missing"


def test_gemini_declaration_drops_integer_enums_but_keeps_string_ones():
    # Gemini rejects the WHOLE tool list (HTTP 400) if any integer
    # property carries an enum, so the relay-channel enum must be
    # stripped on the way out - while string enums (AC mode) survive.
    function = {
        "name": "demo",
        "description": "demo tool",
        "parameters": {
            "type": "object",
            "properties": {
                "relay_number": {"type": "integer", "enum": [1, 2, 3, 4]},
                "mode": {"type": "string", "enum": ["cool", "heat"]},
            },
        },
    }
    declaration = _to_gemini_declaration(function)

    props = declaration["parameters"]["properties"]
    assert "enum" not in props["relay_number"]
    assert props["mode"]["enum"] == ["cool", "heat"]
    # The shared schema must not be mutated - it is what LM Studio gets.
    assert function["parameters"]["properties"]["relay_number"]["enum"] == [1, 2, 3, 4]


def test_no_integer_enum_survives_in_the_real_gemini_tool_list():
    from jarvis_router import _GEMINI_TOOLS

    for declaration in _GEMINI_TOOLS[0]["functionDeclarations"]:
        for prop in (declaration.get("parameters") or {}).get("properties", {}).values():
            if "enum" in prop:
                assert prop["type"] == "string", declaration["name"]
