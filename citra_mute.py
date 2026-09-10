"""
MUTE - stop Citra talking without stopping Citra.

THE PROBLEM THIS REPLACES: the only way to guarantee silence used to be
killing the voice assistant, and because watchdog.py exists to restart
it, that meant killing the watchdogs too - six processes, and everything
has to be started again afterwards. That is a heavy way to get through a
lecture.

Muted means MUTED, NOT OFF. She still hears you, still switches the
lights, still runs your 2am AC schedule, still answers the phone API.
She just does not make a sound doing it. That distinction is the whole
point: "quiet" and "not working" are different things, and conflating
them is why people end up disabling smart-home kit permanently.

WHY A FILE RATHER THAN A VARIABLE: there are several processes here - the
voice assistant, the UI server, the control panel - in separate
interpreters. A module-level flag would only mute whichever one set it.
A file on disk is visible to all of them, survives a restart (so a mute
set before a lecture is still in force if the watchdog cycles the
assistant mid-lecture), and can be toggled by hand, by the phone API, or
by voice without any of them talking to each other.

TIMED BY DEFAULT, and that is deliberate. A mute you have to remember to
undo is a mute that silently breaks the product a week later - somebody
mutes it for a nap, forgets, and concludes Citra stopped working. Every
mute here carries an expiry; an indefinite one has to be asked for.
"""

import json
import os
import time
from typing import Optional

MUTE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "citra_muted.json")

# Read at most this often. is_muted() is called on every single utterance
# and every earcon, so it must not stat the disk in a tight loop - but it
# also must not cache so long that "unmute" feels broken. Half a second
# is far below human reaction time and far above filesystem cost.
_CACHE_SECONDS = 0.5
_cache = {"checked_at": 0.0, "muted": False, "until": None}


def mute(minutes: Optional[float] = 60.0, reason: str = "") -> dict:
    """
    Go quiet. Returns the state written.

    minutes=None mutes indefinitely - use it only when somebody has
    explicitly asked for that, never as a default. See the module
    docstring on why every mute should normally expire.
    """
    state = {
        "muted": True,
        "since": time.time(),
        "until": (time.time() + minutes * 60) if minutes else None,
        "reason": reason,
    }
    with open(MUTE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    _cache["checked_at"] = 0.0        # force the next read to see this
    return state


def unmute() -> None:
    """Start talking again."""
    try:
        os.remove(MUTE_PATH)
    except FileNotFoundError:
        pass
    _cache["checked_at"] = 0.0


def status() -> dict:
    """Full state, for the API and for logging."""
    if not os.path.exists(MUTE_PATH):
        return {"muted": False}
    try:
        with open(MUTE_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt mute file must not be able to silence Citra forever -
        # failing open is the safe direction here, because the failure is
        # noticeable (she talks) rather than silent (she never does).
        return {"muted": False, "note": "mute file unreadable - ignoring it"}

    until = state.get("until")
    if until is not None and time.time() >= until:
        unmute()
        return {"muted": False, "note": "mute expired"}

    if until is not None:
        state["minutes_left"] = round((until - time.time()) / 60, 1)
    return state


def is_muted() -> bool:
    """
    The hot path - called before every utterance and every earcon.

    Cheap by design: a cached boolean refreshed at most twice a second.
    Never raises; if anything at all goes wrong it reports NOT muted,
    because a bug that makes Citra mute forever is far worse than one
    that makes her speak once when she should not have.
    """
    now = time.time()
    if now - _cache["checked_at"] < _CACHE_SECONDS:
        return _cache["muted"]
    try:
        state = status()
        _cache["muted"] = bool(state.get("muted"))
        _cache["until"] = state.get("until")
    except Exception:
        _cache["muted"] = False
    _cache["checked_at"] = now
    return _cache["muted"]


def describe() -> str:
    """One human-readable line, for logs and for the phone API."""
    state = status()
    if not state.get("muted"):
        return "Citra is not muted."
    if state.get("until") is None:
        return "Citra is muted indefinitely." + (
            f" ({state['reason']})" if state.get("reason") else "")
    left = state.get("minutes_left", 0)
    return (f"Citra is muted for another {left:g} minute(s)."
            + (f" ({state['reason']})" if state.get("reason") else ""))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "on":
        mins = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        mute(mins, reason="set from the command line")
        print(describe())
    elif len(sys.argv) > 1 and sys.argv[1] == "off":
        unmute()
        print(describe())
    else:
        print(describe())
        print("\nusage: python citra_mute.py [on [minutes] | off]")
