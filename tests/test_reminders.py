"""
Reminders and scheduled actions ("turn off the AC at 2am"). Timers are
never allowed to actually wait in here: _fire is called directly, so the
suite stays instant and deterministic while still proving that a fired
reminder speaks and a fired action routes silently.
"""
import datetime

import pytest

import jarvis_reminders
from jarvis_reminders import (
    MAX_REMINDER_MINUTES,
    MIN_REMINDER_SECONDS,
    ReminderController,
    _minutes_until_clock_time,
)


@pytest.fixture(autouse=True)
def cancel_leftover_timers():
    """No test may leave a live threading.Timer behind."""
    created = []
    real_controller_init = ReminderController.__init__

    def tracking_init(self, *args, **kwargs):
        real_controller_init(self, *args, **kwargs)
        created.append(self)

    ReminderController.__init__ = tracking_init
    try:
        yield
    finally:
        ReminderController.__init__ = real_controller_init
        for controller in created:
            controller.cancel_reminder("all")


@pytest.fixture
def spoken():
    return []


@pytest.fixture
def routed():
    return []


@pytest.fixture
def controller(spoken, routed) -> ReminderController:
    return ReminderController(
        speak_fn=spoken.append,
        action_fn=routed.append,
    )


# --------------------------------------------------------------------------
# _minutes_until_clock_time
# --------------------------------------------------------------------------
@pytest.fixture
def frozen_now(monkeypatch):
    """Pins 'now' to 23:00 so 'at 2 am' is unambiguous."""
    fixed = datetime.datetime(2026, 1, 15, 23, 0, 0)

    class FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(jarvis_reminders.datetime, "datetime", FrozenDateTime)
    return fixed


@pytest.mark.parametrize("text, expected_minutes", [
    ("2 am", 180.0),          # rolls to tomorrow
    ("2am", 180.0),
    ("2:30 am", 210.0),
    ("02:00", 180.0),
    ("14:30", 15.5 * 60),     # already passed today -> tomorrow
    ("11 pm", 24 * 60),       # exactly now -> tomorrow, not zero
    ("11:30 pm", 30.0),
    ("12 am", 60.0),          # midnight, not noon
    ("12 pm", 13 * 60),       # noon
    ("7 p.m.", 20 * 60),      # "p.m." with dots, as STT sometimes writes it
])
def test_clock_times_resolve_to_the_next_occurrence(frozen_now, text, expected_minutes):
    assert _minutes_until_clock_time(text) == pytest.approx(expected_minutes)


@pytest.mark.parametrize("text", ["soon", "25:00", "12:75", "", "two am"])
def test_unreadable_clock_times_return_none(frozen_now, text):
    assert _minutes_until_clock_time(text) is None


# --------------------------------------------------------------------------
# set_reminder
# --------------------------------------------------------------------------
def test_set_reminder_registers_and_lists(controller):
    result = controller.set_reminder(10, "drink water")

    assert result.success is True
    assert "10 minutes" in result.message
    assert result.data["minutes"] == 10.0

    listed = controller.list_reminders()
    assert listed.success is True
    assert "'drink water'" in listed.message
    assert "1 reminder(s)" in listed.message


def test_set_reminder_needs_a_message(controller):
    result = controller.set_reminder(10, "   ")
    assert result.success is False
    assert "remind you about" in result.message


def test_set_reminder_needs_a_number(controller):
    result = controller.set_reminder("ten", "x")
    assert result.success is False


def test_set_reminder_rejects_too_soon_and_too_far(controller):
    too_soon = controller.set_reminder((MIN_REMINDER_SECONDS - 1) / 60, "x")
    assert too_soon.success is False
    assert "too soon" in too_soon.message

    too_far = controller.set_reminder(MAX_REMINDER_MINUTES + 1, "x")
    assert too_far.success is False
    assert "24 hours" in too_far.message


def test_sub_minute_reminders_are_described_in_seconds(controller):
    assert "30 seconds" in controller.set_reminder(0.5, "x").message
    assert "1 minute." in controller.set_reminder(1, "y").message
    assert "2.5 minutes" in controller.set_reminder(2.5, "z").message


def test_a_fired_reminder_is_spoken(controller, spoken, routed):
    result = controller.set_reminder(10, "call mum")
    controller._fire(result.data["id"])

    assert spoken == ["Reminder: call mum"]
    assert routed == []
    assert "don't have any reminders" in controller.list_reminders().message


def test_a_fired_reminder_also_notifies(spoken):
    notified = []
    controller = ReminderController(speak_fn=spoken.append, notify_fn=notified.append)
    controller._fire(controller.set_reminder(10, "x").data["id"])
    assert notified == ["Reminder: x"]


def test_firing_a_cancelled_reminder_is_a_no_op(controller, spoken):
    result = controller.set_reminder(10, "x")
    controller.cancel_reminder("x")
    controller._fire(result.data["id"])
    assert spoken == []


# --------------------------------------------------------------------------
# cancel_reminder
# --------------------------------------------------------------------------
def test_cancel_matches_by_substring(controller):
    controller.set_reminder(10, "drink water")
    controller.set_reminder(20, "water the plants")
    controller.set_reminder(30, "call mum")

    result = controller.cancel_reminder("water")
    assert result.success is True
    assert "Cancelled 2 reminders" in result.message
    assert "'call mum'" in controller.list_reminders().message


def test_cancel_all(controller):
    controller.set_reminder(10, "a")
    controller.set_reminder(10, "b")
    assert "Cancelled 2" in controller.cancel_reminder("everything").message
    assert "don't have any" in controller.list_reminders().message


def test_cancel_with_no_match_says_so(controller):
    result = controller.cancel_reminder("nothing")
    assert result.success is False
    assert "don't have a reminder matching" in result.message


# --------------------------------------------------------------------------
# schedule_action
# --------------------------------------------------------------------------
def test_schedule_action_at_a_clock_time(controller, frozen_now):
    result = controller.schedule_action("turn off the ac", at_time="2 am")

    assert result.success is True
    assert result.message == "Done — I'll turn off the ac at 2 am."
    assert result.data["minutes"] == pytest.approx(180.0)


def test_schedule_action_in_minutes(controller):
    result = controller.schedule_action("turn on light 1", minutes=5)
    assert result.success is True
    assert "in 5 minutes" in result.message


def test_schedule_action_validation(controller):
    assert controller.schedule_action("", minutes=5).success is False
    assert "When" in controller.schedule_action("x").message
    assert "couldn't read" in controller.schedule_action("x", at_time="never").message
    assert "too soon" in controller.schedule_action("x", minutes=0.01).message
    assert "24 hours" in controller.schedule_action("x", minutes=MAX_REMINDER_MINUTES + 1).message


def test_a_fired_action_is_routed_silently(controller, spoken, routed):
    result = controller.schedule_action("turn off the ac", minutes=5)
    controller._fire(result.data["id"])

    assert routed == ["turn off the ac"]
    assert spoken == []   # 2am AC-off must not wake the room


def test_a_fired_action_with_no_router_does_not_raise(spoken):
    controller = ReminderController(speak_fn=spoken.append)
    controller._fire(controller.schedule_action("x", minutes=5).data["id"])
    assert spoken == []


def test_an_action_that_raises_is_logged_not_propagated(spoken):
    def bad_router(command):
        raise RuntimeError("board offline")

    controller = ReminderController(speak_fn=spoken.append, action_fn=bad_router)
    controller._fire(controller.schedule_action("x", minutes=5).data["id"])
