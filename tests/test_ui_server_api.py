"""
The /api/* routes are what an iPhone Shortcut hits: one GET with a
token header. They are the only part of Citra reachable from a phone,
so the token gate and the input validation are security boundaries,
not conveniences. Runs the real aiohttp app against the fake board
controller from conftest - no NodeMCU, no LLM, no browser.
"""
import pytest

import citra_mute
import citra_ui_server
from citra_ui_server import AC_TEMP_MAX, AC_TEMP_MIN, create_app

TOKEN = "test-token-not-secret"


@pytest.fixture
def app(monkeypatch, fake_controller, tmp_path):
    """The dashboard app with every external edge replaced."""
    monkeypatch.setattr(citra_ui_server, "API_TOKEN", TOKEN)
    # Both controllers point at the same fake so a test can assert on
    # exactly which board calls a phone tap produced.
    monkeypatch.setattr(citra_ui_server, "phone_controller", fake_controller)
    monkeypatch.setattr(citra_ui_server, "controller", fake_controller)
    # Do not build a JarvisRouter (and its five controllers) on startup.
    monkeypatch.setattr(citra_ui_server, "prewarm_text_router", _no_prewarm)
    # Mute state must not leak into (or out of) the developer's real file.
    monkeypatch.setattr(citra_mute, "MUTE_PATH", str(tmp_path / "muted.json"))
    citra_mute._cache.update({"checked_at": 0.0, "muted": False, "until": None})
    return create_app()


async def _no_prewarm(app=None):
    return None


@pytest.fixture
async def client(aiohttp_client, app):
    return await aiohttp_client(app)


def _auth() -> dict:
    return {"X-Citra-Token": TOKEN}


# --------------------------------------------------------------------------
# the token gate
# --------------------------------------------------------------------------
async def test_api_without_a_token_is_401_and_touches_no_hardware(client, fake_controller):
    response = await client.get("/api/lights/on")
    assert response.status == 401
    assert (await response.json()) == {"ok": False, "error": "unauthorized"}
    assert fake_controller.calls == []


async def test_api_with_a_wrong_token_is_401(client, fake_controller):
    response = await client.get("/api/lights/on", headers={"X-Citra-Token": "nope"})
    assert response.status == 401
    assert fake_controller.calls == []


async def test_token_is_accepted_in_a_header_or_a_query_param(client):
    assert (await client.get("/api/mute/status", headers=_auth())).status == 200
    assert (await client.get("/api/mute/status", params={"token": TOKEN})).status == 200


async def test_the_gate_only_covers_api_routes(client):
    # The dashboard itself is loopback-only and deliberately ungated.
    response = await client.get("/")
    assert response.status != 401


# --------------------------------------------------------------------------
# lights
# --------------------------------------------------------------------------
async def test_lights_on_switches_all_four_relays(client, fake_controller):
    response = await client.get("/api/lights/on", headers=_auth())

    assert response.status == 200
    assert (await response.json()) == {"ok": True, "did": "lights on"}
    assert fake_controller.calls == [("turn_on_relay", (n,)) for n in (1, 2, 3, 4)]


async def test_lights_rejects_states_other_than_on_off(client, fake_controller):
    response = await client.get("/api/lights/dim", headers=_auth())
    assert response.status == 400
    assert "on or off" in (await response.json())["error"]
    assert fake_controller.calls == []


async def test_one_light(client, fake_controller):
    response = await client.get("/api/light/3/off", headers=_auth())
    assert response.status == 200
    assert (await response.json())["did"] == "light 3 off"
    assert fake_controller.calls == [("turn_off_relay", (3,))]


@pytest.mark.parametrize("path", ["/api/light/0/on", "/api/light/5/on", "/api/light/x/on"])
async def test_light_number_is_validated(client, fake_controller, path):
    response = await client.get(path, headers=_auth())
    assert response.status == 400
    assert fake_controller.calls == []


async def test_a_dead_board_is_a_502_not_a_200(client, failing_controller, monkeypatch):
    monkeypatch.setattr(citra_ui_server, "phone_controller", failing_controller)
    response = await client.get("/api/light/1/on", headers=_auth())

    assert response.status == 502
    body = await response.json()
    assert body["ok"] is False
    assert "failed" in body["error"]


# --------------------------------------------------------------------------
# air conditioner
# --------------------------------------------------------------------------
async def test_ac_power(client, fake_controller):
    assert (await client.get("/api/ac/on", headers=_auth())).status == 200
    assert (await client.get("/api/ac/off", headers=_auth())).status == 200
    assert fake_controller.calls == [("set_ac_power", (True,)), ("set_ac_power", (False,))]


async def test_ac_temperature_in_range(client, fake_controller):
    response = await client.get("/api/ac/temp/24", headers=_auth())
    assert response.status == 200
    assert (await response.json())["did"] == "ac 24 degrees"
    assert fake_controller.calls == [("set_ac_temperature", (24,))]


@pytest.mark.parametrize("value", [AC_TEMP_MIN - 1, AC_TEMP_MAX + 1, "hot", 99])
async def test_ac_temperature_out_of_range_never_reaches_the_ir_blaster(client, fake_controller, value):
    response = await client.get(f"/api/ac/temp/{value}", headers=_auth())
    assert response.status == 400
    assert fake_controller.calls == []


# --------------------------------------------------------------------------
# everything off
# --------------------------------------------------------------------------
async def test_everything_off_hits_relays_then_ac(client, fake_controller):
    response = await client.get("/api/everything/off", headers=_auth())
    assert response.status == 200
    assert fake_controller.calls == [
        ("turn_off_relay", (1,)), ("turn_off_relay", (2,)),
        ("turn_off_relay", (3,)), ("turn_off_relay", (4,)),
        ("set_ac_power", (False,)),
    ]


# --------------------------------------------------------------------------
# mute
# --------------------------------------------------------------------------
async def test_mute_defaults_to_an_hour_and_unmute_clears_it(client):
    response = await client.get("/api/mute", headers=_auth())
    assert response.status == 200
    assert "muted for another" in (await response.json())["did"]
    assert citra_mute.is_muted() is True

    status = await (await client.get("/api/mute/status", headers=_auth())).json()
    assert status["state"]["muted"] is True
    assert status["state"]["reason"] == "muted from the phone"

    response = await client.get("/api/unmute", headers=_auth())
    assert (await response.json())["did"] == "Citra can talk again."
    assert citra_mute.is_muted() is False


async def test_mute_zero_means_indefinite(client):
    await client.get("/api/mute", params={"minutes": "0"}, headers=_auth())
    assert citra_mute.status()["until"] is None


async def test_mute_minutes_must_be_a_number(client):
    response = await client.get("/api/mute", params={"minutes": "later"}, headers=_auth())
    assert response.status == 400


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
async def test_status_reads_both_boards_fast_fail(client, fake_controller):
    response = await client.get("/api/status", headers=_auth())
    assert response.status == 200
    body = await response.json()
    assert body["ok"] is True
    assert body["relays"]["method"] == "get_all_relay_status"
    assert body["ac"]["method"] == "get_ac_status"


# --------------------------------------------------------------------------
# caching headers - a stale dashboard looked like an unfixed bug once
# --------------------------------------------------------------------------
async def test_the_dashboard_is_served_with_no_store(client):
    response = await client.get("/")
    assert response.headers.get("Cache-Control") == "no-store, must-revalidate"
