"""
=============================================================================
JARVIS REMINDERS — voice-controlled one-shot timers and reminders
=============================================================================
Runs on: the same process as jarvis_voice_assistant.py.
Talks to: nothing external — pure in-process scheduling (threading.Timer),
          no network, no hardware board.

WHY THIS EXISTS
---------------
A real, missing capability: there was no way to say "remind me in 20
minutes to take the laundry out" and have anything happen. This is one of
the most standard things a voice assistant does, and this project had
never built it.

IN-MEMORY ONLY, ON PURPOSE:
A reminder set now does NOT survive this process restarting — a
threading.Timer is a scheduled in-process callback, nothing written to
disk. That's a deliberate scope limit, not an oversight: persisting
reminders across restarts needs a real store (at minimum a JSON file) plus
rehydration logic on startup, which is meaningfully more than "set a
timer." This covers the actual common case — you're already running,
you want a nudge in N minutes — without that added complexity. If
surviving a restart becomes a real need, that's the point to add
persistence, not before.

WHY speak_fn/notify_fn ARE INJECTED, NOT IMPORTED:
speak_async() and citra_ui_bridge live in/are used by
jarvis_voice_assistant.py, which is what CONSTRUCTS a ReminderController.
Importing them directly here would mean jarvis_voice_assistant.py
importing this file, which imports back toward jarvis_voice_assistant.py
— a real import cycle. Accepting them as constructor callables instead
mirrors the exact dependency-injection shape jarvis_router.py's
JarvisRouter already uses for its own controllers (accept a pre-built
instance or construct a default), and happens to make this trivially
testable with a mock speak function too.
=============================================================================
"""

import datetime
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jarvis_reminders")


@dataclass
class ReminderActionResult:
    """Same shape as PCActionResult/VisionActionResult/HardwareResult —
    see PCActionResult's own docstring for why each subsystem keeps its
    own copy rather than sharing one class across unrelated subsystems."""
    success: bool
    action: str
    message: str
    data: Optional[dict] = field(default=None)

    def to_dict(self) -> dict:
        return {"success": self.success, "action": self.action, "message": self.message, "data": self.data}


@dataclass
class _PendingReminder:
    id: str
    message: str
    fire_at: float  # time.monotonic() timestamp
    timer: threading.Timer
    # None for a plain spoken reminder; the command text to execute for a
    # scheduled ACTION ("turn off the ac"). The two share this one queue
    # and one timer mechanism deliberately -- they differ only in what
    # happens at fire time, so splitting them would duplicate the
    # scheduling, listing and cancelling logic for no gain.
    command: Optional[str] = None


MAX_REMINDER_MINUTES = 24 * 60  # 24 hours — a sane ceiling for an
                                    # in-memory timer that doesn't survive
                                    # a restart anyway (see module
                                    # docstring); rejecting anything
                                    # longer is honest about that limit
                                    # rather than silently accepting a
                                    # reminder unlikely to still be alive
                                    # when it matters.
MIN_REMINDER_SECONDS = 5  # below this, "remind me" is almost certainly a
                                    # misheard "remember" or similar —
                                    # reject rather than fire something
                                    # near-instantly with no real gap.


def _minutes_until_clock_time(at_time: str) -> Optional[float]:
    """Minutes from now until the next occurrence of a wall-clock time
    like "02:00", "2 am", "14:30". Returns None if unparseable.

    Rolls to TOMORROW when the time has already passed today, which is the
    only sane reading of "turn off the AC at 2 am" said at 11pm — the
    common case for this feature, not an edge case.

    Absolute times are resolved HERE rather than asking the model to
    compute a delay: the LLM has no reliable notion of the current time
    (nothing in the prompt tells it), so "at 2 am" would otherwise be a
    guess at how many minutes away that is.
    """
    text = at_time.strip().lower().replace(".", "")
    match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)

    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None

    now = datetime.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds() / 60.0


class ReminderController:
    """
    Owns every pending reminder for the process's lifetime. Thread-safe:
    _fire() runs on whatever thread threading.Timer schedules it on
    (never the main/consumer thread), while set_reminder/list_reminders/
    cancel_reminder are called from the Smart Path's tool-dispatch thread
    — self._lock guards the shared _pending dict against both sides
    touching it at once.
    """

    def __init__(
        self,
        speak_fn: Optional[Callable[[str], object]] = None,
        notify_fn: Optional[Callable[[str], None]] = None,
        action_fn: Optional[Callable[[str], object]] = None,
    ):
        # action_fn runs a scheduled command when its timer fires, by
        # routing the text exactly as if it had just been spoken. Injected
        # for the same reason speak_fn is: the router lives upstream of
        # this file, and importing it here would create a cycle.
        self._action_fn = action_fn
        # Defaults to a log-only fallback, not a required argument — same
        # zero-args-works pattern every other controller in this project
        # follows (SmartRoomController(), PCController(), VisionController()
        # all construct with no arguments). Matters concretely: this
        # file's own __main__ interactive test loop, and jarvis_router.py's,
        # construct these controllers with no real TTS available at all.
        # jarvis_voice_assistant.py passes its REAL speak_async in — this
        # fallback only ever fires when nothing else was provided.
        self._speak_fn = speak_fn or (lambda text: logger.info("[no speak_fn configured] Would have said: %s", text))
        self._notify_fn = notify_fn
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingReminder] = {}

    def _fire(self, reminder_id: str) -> None:
        with self._lock:
            pending = self._pending.pop(reminder_id, None)
        if pending is None:
            return  # cancelled between being scheduled and firing

        if pending.command is not None:
            # Scheduled ACTION — do the thing, SILENTLY. Deliberately no
            # speech here: the headline use for this is "turn off the AC
            # at 2 am", and announcing it out loud would wake the room it
            # was meant to keep comfortable. The action is logged instead,
            # so there's still a record of what ran and whether it worked.
            logger.info("Scheduled action firing: %r", pending.command)
            if self._action_fn is None:
                logger.error("No action_fn configured — cannot run scheduled command %r.", pending.command)
                return
            try:
                self._action_fn(pending.command)
            except Exception as exc:
                logger.error("Scheduled action %r failed: %s", pending.command, exc)
            return

        logger.info("Reminder firing: '%s'", pending.message)
        spoken = f"Reminder: {pending.message}"
        self._speak_fn(spoken)
        if self._notify_fn is not None:
            self._notify_fn(spoken)

    def schedule_action(
        self,
        command: str,
        at_time: Optional[str] = None,
        minutes: Optional[float] = None,
    ) -> ReminderActionResult:
        """Runs `command` later, as if it had been spoken then. Takes
        either a wall-clock time ("2 am") or a relative delay."""
        command = (command or "").strip()
        if not command:
            return ReminderActionResult(success=False, action="schedule_action", message="What should I do?")

        if at_time:
            resolved = _minutes_until_clock_time(at_time)
            if resolved is None:
                return ReminderActionResult(
                    success=False, action="schedule_action",
                    message=f"I couldn't read '{at_time}' as a time.",
                )
            minutes = resolved
            when_text = f"at {at_time}"
        elif minutes is not None:
            try:
                minutes = float(minutes)
            except (TypeError, ValueError):
                return ReminderActionResult(success=False, action="schedule_action", message="I need a time or a number of minutes.")
            when_text = (
                f"in {int(round(minutes * 60))} seconds" if minutes < 1
                else f"in {int(round(minutes))} minutes"
            )
        else:
            return ReminderActionResult(success=False, action="schedule_action", message="When should I do that?")

        if minutes * 60 < MIN_REMINDER_SECONDS:
            return ReminderActionResult(
                success=False, action="schedule_action",
                message="That's too soon to schedule — I'd just do it now.",
            )
        if minutes > MAX_REMINDER_MINUTES:
            return ReminderActionResult(
                success=False, action="schedule_action",
                message=f"I can only schedule up to {MAX_REMINDER_MINUTES // 60} hours ahead.",
            )

        action_id = str(uuid.uuid4())
        timer = threading.Timer(minutes * 60, self._fire, args=(action_id,))
        timer.daemon = True
        with self._lock:
            self._pending[action_id] = _PendingReminder(
                id=action_id, message=command,
                fire_at=time.monotonic() + minutes * 60,
                timer=timer, command=command,
            )
        timer.start()

        logger.info("Scheduled action %r %s (%.0f min away).", command, when_text, minutes)
        return ReminderActionResult(
            success=True, action="schedule_action",
            message=f"Done — I'll {command} {when_text}.",
            data={"id": action_id, "minutes": minutes, "command": command},
        )

    def set_reminder(self, minutes: float, message: str) -> ReminderActionResult:
        message = (message or "").strip()
        if not message:
            return ReminderActionResult(success=False, action="set_reminder", message="What should I remind you about?")

        try:
            minutes = float(minutes)
        except (TypeError, ValueError):
            return ReminderActionResult(success=False, action="set_reminder", message="I need a number of minutes for that.")

        seconds = minutes * 60
        if seconds < MIN_REMINDER_SECONDS:
            return ReminderActionResult(
                success=False, action="set_reminder",
                message=f"That's too soon for me to set reliably — give me at least {MIN_REMINDER_SECONDS} seconds.",
            )
        if minutes > MAX_REMINDER_MINUTES:
            return ReminderActionResult(
                success=False, action="set_reminder",
                message=f"That's further out than I can reliably hold, sir — I only keep reminders in memory, so anything past {MAX_REMINDER_MINUTES // 60} hours won't survive a restart.",
            )

        reminder_id = str(uuid.uuid4())
        timer = threading.Timer(seconds, self._fire, args=(reminder_id,))
        timer.daemon = True  # never blocks process shutdown

        with self._lock:
            self._pending[reminder_id] = _PendingReminder(
                id=reminder_id, message=message,
                fire_at=time.monotonic() + seconds, timer=timer,
            )
        timer.start()

        if minutes < 1:
            when_text = f"{int(round(seconds))} seconds"
        elif minutes == int(minutes):
            when_text = f"{int(minutes)} minute{'s' if minutes != 1 else ''}"
        else:
            when_text = f"{minutes:.1f} minutes"

        logger.info("Reminder set (%s from now): '%s'", when_text, message)
        return ReminderActionResult(
            success=True, action="set_reminder",
            message=f"Got it — I'll remind you in {when_text}.",
            data={"id": reminder_id, "minutes": minutes},
        )

    def list_reminders(self) -> ReminderActionResult:
        with self._lock:
            items: List[_PendingReminder] = list(self._pending.values())

        if not items:
            return ReminderActionResult(success=True, action="list_reminders", message="You don't have any reminders set right now.")

        now = time.monotonic()
        parts = []
        for item in sorted(items, key=lambda r: r.fire_at):
            remaining = max(0, int(item.fire_at - now))
            minutes, seconds = divmod(remaining, 60)
            when = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
            parts.append(f"'{item.message}' in {when}")

        return ReminderActionResult(
            success=True, action="list_reminders",
            message=f"You have {len(items)} reminder(s): " + "; ".join(parts) + ".",
        )

    def cancel_reminder(self, message_contains: str) -> ReminderActionResult:
        query = (message_contains or "").strip().lower()

        with self._lock:
            if not query or query in ("all", "everything"):
                matches = list(self._pending.values())
            else:
                matches = [r for r in self._pending.values() if query in r.message.lower()]
            for match in matches:
                match.timer.cancel()
                self._pending.pop(match.id, None)

        if not matches:
            return ReminderActionResult(
                success=False, action="cancel_reminder",
                message=f"I don't have a reminder matching '{message_contains}'.",
            )

        logger.info("Cancelled %d reminder(s) matching %r", len(matches), message_contains)
        return ReminderActionResult(
            success=True, action="cancel_reminder",
            message=f"Cancelled {len(matches)} reminder{'s' if len(matches) != 1 else ''}.",
        )


# -----------------------------------------------------------------------------
# TOOL SCHEMA — same OpenAI-style shape as jarvis_hardware_api.JARVIS_TOOL_SCHEMA
# -----------------------------------------------------------------------------
REMINDER_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": (
                "Set a one-time reminder that speaks a message aloud after a "
                "delay. Use for 'remind me in X minutes/hours to...', 'set a "
                "timer for X'. Convert the user's phrasing into minutes "
                "yourself (e.g. 'half an hour' = 30, 'an hour and a half' = "
                "90, '2 hours' = 120, '90 seconds' = 1.5)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "number",
                        "description": "How many minutes from now to fire the reminder (can be fractional for under a minute).",
                    },
                    "message": {
                        "type": "string",
                        "description": "What to remind the user about.",
                    },
                },
                "required": ["minutes", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_action",
            "description": (
                "Schedule a room command to run automatically later — e.g. "
                "'turn off the AC at 2am', 'switch off the lights in an "
                "hour'. Use this whenever the user wants something DONE at "
                "a later time, as opposed to set_reminder which only speaks "
                "a message to remind THEM to do it. Pass a wall-clock time "
                "in at_time when the user names one ('2am', '6:30'); pass "
                "minutes only for a relative delay. It runs silently, so "
                "it's safe to schedule overnight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run later, phrased as the user would say it now, e.g. 'turn off the ac'.",
                    },
                    "at_time": {
                        "type": "string",
                        "description": "Wall-clock time like '2am', '02:00', '14:30'. Rolls to tomorrow if already past today.",
                    },
                    "minutes": {
                        "type": "number",
                        "description": "Alternative to at_time: run this many minutes from now.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List every currently pending reminder and how long until each fires. Use for 'what are my reminders', 'do I have any timers set'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancel a pending reminder. Use for 'cancel my reminder about X', 'cancel the timer'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_contains": {
                        "type": "string",
                        "description": "Text to match against pending reminders' messages, or 'all' to cancel every pending reminder.",
                    }
                },
                "required": ["message_contains"],
            },
        },
    },
]


if __name__ == "__main__":
    import json
    print("\n=== Reminder Tool Schema ===\n")
    print(json.dumps(REMINDER_TOOL_SCHEMA, indent=2))
