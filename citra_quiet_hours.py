"""
Citra will not ring anybody in the middle of the night.

WHY THIS IS CODE AND NOT A PROMISE. Every other guard around calling
protects the person being called from being called by MISTAKE - wrong
name, wrong row, ambiguous match. This one protects them from being
called CORRECTLY at a time when nobody wants a phone call. A misdial at
2pm is embarrassing. A perfectly correct call at 2am wakes a household.

Judgement is not a control. "Don't call people late" as an instruction
survives exactly as long as whoever holds it is paying attention - and
this system now has a scheduler, a router tool, and retry loops that can
all reach a dial without a human in the loop. So the window lives here,
in one place, checked at the point of dialling.

THE OVERRIDE IS DELIBERATE AND NARROW. Abhay's own 6am wake-up call is
inside the quiet window by design; a wake-up call that refused to fire
at dawn would be useless. So a caller may pass allow_quiet=True, and it
is then recorded in the log as an override rather than passing silently.
The override exists to be used on purpose, never by default.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime

logger = logging.getLogger("citra.quiet")

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "citra_quiet_hours.json")

# Defaults if there is no config file. 22:00 to 07:00 - late enough not
# to block an ordinary evening call, early enough to cover the hours
# where a ringing phone is genuinely unwelcome.
DEFAULT_START = dtime(22, 0)
DEFAULT_END = dtime(7, 0)


@dataclass
class Verdict:
    allowed: bool
    reason: str
    overridden: bool = False

    def __bool__(self) -> bool:
        return self.allowed


def _parse(value: str, fallback: dtime) -> dtime:
    try:
        hours, minutes = str(value).strip().split(":")
        return dtime(int(hours), int(minutes))
    except Exception:
        return fallback


def load_config() -> tuple[dtime, dtime, bool]:
    """(start, end, enabled). Falls back to the defaults on any problem."""
    start, end, enabled = DEFAULT_START, DEFAULT_END, True
    if not os.path.exists(CONFIG_PATH):
        return start, end, enabled
    try:
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            raw = json.load(handle)
        start = _parse(raw.get("quiet_from", "22:00"), DEFAULT_START)
        end = _parse(raw.get("quiet_until", "07:00"), DEFAULT_END)
        enabled = bool(raw.get("enabled", True))
    except Exception:
        # A broken config must FAIL CLOSED here, unlike the mute file
        # which fails open. Getting this wrong in the permissive
        # direction means ringing somebody at 3am.
        logger.warning("quiet-hours config unreadable; using the safe defaults")
    return start, end, enabled


def save_config(quiet_from: str, quiet_until: str, enabled: bool = True) -> None:
    payload = {
        "_comment": "Hours when Citra will not place calls. Overnight "
                    "windows (22:00 -> 07:00) are expected and handled.",
        "quiet_from": quiet_from,
        "quiet_until": quiet_until,
        "enabled": enabled,
    }
    temp = CONFIG_PATH + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp, CONFIG_PATH)


def is_quiet(now: datetime | None = None) -> bool:
    """Is `now` inside the quiet window?"""
    start, end, enabled = load_config()
    if not enabled:
        return False
    moment = (now or datetime.now()).time()

    if start == end:
        return False
    if start < end:                     # a same-day window, e.g. 01:00-06:00
        return start <= moment < end
    # An OVERNIGHT window, e.g. 22:00 -> 07:00. This wraps midnight, so
    # the test is "after the start OR before the end" - the case a naive
    # start <= t < end check gets exactly backwards, allowing calls all
    # night and blocking them all day.
    return moment >= start or moment < end


def check(now: datetime | None = None, allow_quiet: bool = False) -> Verdict:
    """
    May Citra place a call right now?

    allow_quiet=True is the narrow, deliberate exception - the scheduled
    wake-up call. It is logged, so an override always leaves a trace.
    """
    start, end, enabled = load_config()
    if not enabled:
        return Verdict(True, "quiet hours are switched off")

    moment = now or datetime.now()
    if not is_quiet(moment):
        return Verdict(True, "outside quiet hours")

    window = "%s-%s" % (start.strftime("%H:%M"), end.strftime("%H:%M"))
    if allow_quiet:
        logger.warning("QUIET HOURS OVERRIDDEN at %s (window %s)",
                       moment.strftime("%H:%M"), window)
        return Verdict(True, f"quiet hours ({window}) overridden on purpose",
                       overridden=True)

    return Verdict(False,
                   f"it's {moment.strftime('%H:%M')} - inside quiet hours "
                   f"({window}). I'm not going to ring anyone right now.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "set":
        save_config(sys.argv[2], sys.argv[3])
        print("quiet hours: %s -> %s" % (sys.argv[2], sys.argv[3]))
    start, end, enabled = load_config()
    print("window  : %s -> %s  (%s)" % (start.strftime("%H:%M"),
                                        end.strftime("%H:%M"),
                                        "on" if enabled else "off"))
    print("now     : %s" % datetime.now().strftime("%H:%M"))
    verdict = check()
    print("calling : %s - %s" % ("ALLOWED" if verdict else "BLOCKED",
                                 verdict.reason))
