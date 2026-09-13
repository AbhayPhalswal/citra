"""
Two small modules with one property each that must never regress:

  citra_mute        - "mute means muted", it expires on its own, and a
                      broken mute file FAILS OPEN (she talks) rather than
                      silencing her forever.
  citra_quiet_hours - an overnight window (22:00 -> 07:00) must be
                      evaluated correctly across midnight, and a broken
                      config FAILS CLOSED (no 3am calls).

Both write their state next to the module; the tests point them at a
temp directory instead.
"""
import json
import time
from datetime import datetime, time as dtime

import pytest

import citra_mute
import citra_quiet_hours


# --------------------------------------------------------------------------
# citra_mute
# --------------------------------------------------------------------------
@pytest.fixture
def mute_file(tmp_path, monkeypatch):
    path = tmp_path / "citra_muted.json"
    monkeypatch.setattr(citra_mute, "MUTE_PATH", str(path))
    # Every test starts unmuted with a cold cache.
    citra_mute._cache.update({"checked_at": 0.0, "muted": False, "until": None})
    yield path
    citra_mute._cache.update({"checked_at": 0.0, "muted": False, "until": None})


def test_not_muted_by_default(mute_file):
    assert citra_mute.is_muted() is False
    assert citra_mute.status() == {"muted": False}
    assert citra_mute.describe() == "Citra is not muted."


def test_mute_writes_state_and_is_seen_immediately(mute_file):
    state = citra_mute.mute(30, reason="on a call")

    assert state["muted"] is True
    assert state["until"] > time.time()
    assert mute_file.exists()
    # The cache is invalidated by mute(), so no half-second lag.
    assert citra_mute.is_muted() is True
    assert "muted for another" in citra_mute.describe()
    assert "(on a call)" in citra_mute.describe()


def test_unmute_removes_the_file_and_is_seen_immediately(mute_file):
    citra_mute.mute(30)
    citra_mute.unmute()

    assert not mute_file.exists()
    assert citra_mute.is_muted() is False
    citra_mute.unmute()  # idempotent - no FileNotFoundError


def test_an_expired_mute_clears_itself(mute_file):
    citra_mute.mute(30)
    # Rewrite the file so it expired a second ago.
    state = json.loads(mute_file.read_text(encoding="utf-8"))
    state["until"] = time.time() - 1
    mute_file.write_text(json.dumps(state), encoding="utf-8")
    citra_mute._cache["checked_at"] = 0.0

    assert citra_mute.status() == {"muted": False, "note": "mute expired"}
    assert not mute_file.exists()
    assert citra_mute.is_muted() is False


def test_indefinite_mute_never_expires(mute_file):
    state = citra_mute.mute(None, reason="library")

    assert state["until"] is None
    assert citra_mute.is_muted() is True
    assert citra_mute.describe() == "Citra is muted indefinitely. (library)"


def test_a_corrupt_mute_file_fails_open(mute_file):
    mute_file.write_text("{ not json", encoding="utf-8")
    citra_mute._cache["checked_at"] = 0.0

    assert citra_mute.status()["muted"] is False
    assert citra_mute.is_muted() is False


def test_is_muted_is_cached_for_half_a_second(mute_file, monkeypatch):
    calls = {"n": 0}
    real_status = citra_mute.status

    def counting_status():
        calls["n"] += 1
        return real_status()

    monkeypatch.setattr(citra_mute, "status", counting_status)
    for _ in range(50):
        citra_mute.is_muted()
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# citra_quiet_hours
# --------------------------------------------------------------------------
@pytest.fixture
def quiet_config(tmp_path, monkeypatch):
    path = tmp_path / "citra_quiet_hours.json"
    monkeypatch.setattr(citra_quiet_hours, "CONFIG_PATH", str(path))
    return path


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 15, hour, minute)


def test_defaults_are_an_overnight_window(quiet_config):
    start, end, enabled = citra_quiet_hours.load_config()
    assert (start, end, enabled) == (dtime(22, 0), dtime(7, 0), True)


@pytest.mark.parametrize("hour, expected", [
    (21, False), (22, True), (23, True), (0, True), (3, True),
    (6, True), (7, False), (12, False),
])
def test_overnight_window_wraps_midnight(quiet_config, hour, expected):
    # The naive `start <= t < end` check gets this exactly backwards:
    # it would allow calls all night and block them all day.
    assert citra_quiet_hours.is_quiet(_at(hour)) is expected


def test_same_day_window(quiet_config):
    citra_quiet_hours.save_config("01:00", "06:00")
    assert citra_quiet_hours.is_quiet(_at(0, 59)) is False
    assert citra_quiet_hours.is_quiet(_at(1)) is True
    assert citra_quiet_hours.is_quiet(_at(5, 59)) is True
    assert citra_quiet_hours.is_quiet(_at(6)) is False


def test_disabled_config_never_blocks(quiet_config):
    citra_quiet_hours.save_config("22:00", "07:00", enabled=False)
    assert citra_quiet_hours.is_quiet(_at(3)) is False
    verdict = citra_quiet_hours.check(_at(3))
    assert verdict.allowed and "switched off" in verdict.reason


def test_start_equal_to_end_means_no_window(quiet_config):
    citra_quiet_hours.save_config("09:00", "09:00")
    assert citra_quiet_hours.is_quiet(_at(9)) is False


def test_check_blocks_inside_the_window_with_a_spoken_reason(quiet_config):
    verdict = citra_quiet_hours.check(_at(2, 30))
    assert bool(verdict) is False
    assert "02:30" in verdict.reason
    assert "22:00-07:00" in verdict.reason
    assert verdict.overridden is False


def test_check_allows_outside_the_window(quiet_config):
    verdict = citra_quiet_hours.check(_at(14))
    assert verdict.allowed is True
    assert verdict.overridden is False


def test_an_explicit_override_is_allowed_and_marked(quiet_config):
    verdict = citra_quiet_hours.check(_at(2), allow_quiet=True)
    assert verdict.allowed is True
    assert verdict.overridden is True
    assert "overridden" in verdict.reason


def test_a_corrupt_config_fails_closed_to_the_defaults(quiet_config):
    quiet_config.write_text("{ broken", encoding="utf-8")
    assert citra_quiet_hours.load_config() == (dtime(22, 0), dtime(7, 0), True)
    assert citra_quiet_hours.is_quiet(_at(3)) is True


def test_unparseable_times_fall_back_individually(quiet_config):
    quiet_config.write_text(json.dumps({"quiet_from": "midnight", "quiet_until": "06:30"}),
                            encoding="utf-8")
    start, end, _ = citra_quiet_hours.load_config()
    assert start == dtime(22, 0)   # the bad one fell back
    assert end == dtime(6, 30)     # the good one was kept


def test_save_config_is_atomic(quiet_config, tmp_path):
    citra_quiet_hours.save_config("23:00", "06:00")
    assert not (tmp_path / "citra_quiet_hours.json.tmp").exists()
    assert json.loads(quiet_config.read_text(encoding="utf-8"))["quiet_from"] == "23:00"
