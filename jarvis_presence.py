"""
PRESENCE — keeping Citra company with you while she works.

THE PROBLEM THIS SOLVES, precisely:

Every path through this assistant except one already talks the moment it
understands you. The protocol handlers in jarvis_voice_assistant.py speak
and dispatch hardware on separate threads — "Turning on light 2, sir."
leaves the speaker while the relay is still clicking. That is why lights
and AC feel instant even though they aren't quite.

The Smart Path does not do this. jarvis_router.JarvisRouter.route() blocks
for 1.3 seconds or more while Gemini thinks, and for that entire time the
room is silent. The user has finished speaking, heard nothing back, and
has no way to know whether they were heard at all. That is the moment
people repeat themselves — and a repeat lands as a second wake, a second
recording, and a genuinely confused assistant.

Silence is the bug. Not the latency.

WHY THE LINES HERE ARE TRUE ONES:

The tempting version of this feature is to invent impressive-sounding
progress ("searching the web...", "querying the database...") to make the
wait feel earned. Every line in this module is instead something that is
actually happening, or an honest readback of what was actually asked.

Two practical reasons, over and above not wanting to lie to somebody in
their own home:

  1. Invented status breaks in public. "Searching the web" spoken while
     the building's internet is down is a claim the listener can falsify
     standing in their own hall — and in this product the listener is a
     neighbour who knows which flat to knock on.

  2. It contradicts the pitch. Citra is sold on being the honest option:
     no cloud, no data leaving the flat, "if it isn't right for your home
     I'll tell you so." An assistant that embellishes what it is doing,
     however harmlessly, is a small crack in exactly the thing that makes
     the product worth buying.

The honest lines are also simply better, because the useful information
in a progress line is not "I am busy" — it is READBACK. "Let me check the
weather for Sector 9" tells you she got the topic and the place right.
That is the same reason air traffic control reads instructions back
rather than saying "acknowledged": confirmation beats reassurance.

WHY THE TIMER, AND NOT JUST SPEAKING IMMEDIATELY:

Speech costs time. A two-second spoken line in front of an operation that
would have finished in 400ms makes the assistant five times slower while
appearing more responsive — the worst possible trade. So nothing here
speaks until the work has ALREADY proven itself slow (FIRST_LINE_DELAY_
SECONDS). Fast work is never narrated, is never delayed, and never even
constructs a line: start() arms timers and returns immediately.
"""

import random
import threading
from collections.abc import Callable, Sequence

# =============================================================================
# TIMING
# =============================================================================
# Measured context for these two numbers (see the latency pass logged in
# jarvis_voice_assistant.py): Whisper tiny transcribes in 377-511ms, the
# semantic router decides in ~9ms, and a Gemini round trip with the tool
# schema attached costs ~1.3s and up. Anything routed semantically is
# therefore finished and speaking well before the first threshold here and
# will never reach this module's timers at all — which is the intent.
FIRST_LINE_DELAY_SECONDS = 0.65

# Deliberately far out. A second line exists for genuinely long work — a
# tool call that hits the network, a vision pass over a screenshot — and
# not to fill a normal Gemini wait, which the first line already covers.
# Two lines in quick succession reads as nervous, not attentive.
SECOND_LINE_DELAY_SECONDS = 4.5

# How long the answer is willing to wait for an in-flight progress line to
# finish before it starts talking anyway. Lines here are short (roughly a
# second spoken), so this is generous; it exists to stop two Piper streams
# opening the same output device at once, not to enforce politeness.
LINE_DRAIN_TIMEOUT_SECONDS = 2.5


# =============================================================================
# WHAT SHE SAYS
# =============================================================================
# TOPIC LINES: matched against the transcript, so they read back what the
# user actually asked rather than asserting anything about internal state.
# "Let me check the weather" is a statement of intent about a request that
# genuinely was about weather; it stays true whether Gemini ends up calling
# the weather tool, answering from context, or failing.
#
# Ordering matters: the first tuple whose keywords appear wins, so the more
# specific topics are listed above the broader ones.
_TOPIC_LINES: Sequence[tuple[tuple[str, ...], tuple[str, ...]]] = (
    (
        ("weather", "rain", "raining", "umbrella", "forecast", "barish", "baarish", "garmi", "thand"),
        ("Let me check the weather, sir.",
         "One moment — checking the forecast."),
    ),
    (
        ("screen", "my display", "what am i looking", "on my laptop", "this window"),
        ("Taking a look at your screen, sir.",
         "One moment — reading your screen."),
    ),
    (
        ("remind", "reminder", "yaad dila", "wake me", "alarm"),
        ("Setting that up now, sir.",
         "One moment — putting that on the list."),
    ),
    (
        ("schedule", "at 2 am", "later tonight", "tomorrow morning", "every day"),
        ("Scheduling that now, sir.",
         "One moment — setting the timer."),
    ),
    (
        ("play", "song", "music", "gaana", "volume"),
        ("Getting that playing, sir.",
         "One moment — starting the music."),
    ),
    (
        ("open", "launch", "close", "quit", "shut down"),
        ("Opening that now, sir.",
         "One moment — starting it up."),
    ),
    (
        ("complaint", "plumber", "electrician", "society office", "guard"),
        ("Writing that down for the office, sir.",
         "One moment — logging that complaint."),
    ),
    (
        ("who", "what is", "what's the", "how many", "when did", "why is", "tell me about"),
        ("Let me look that up, sir.",
         "One moment — finding that out."),
    ),
)

# SMALL TALK: questions ABOUT HER, not about the world or a task. These
# never get a progress line, of any kind — not a topic line, not a
# generic one. Two reasons, and the second is the one that actually
# forced this:
#
#   1. There is nothing to report. "What are you doing" isn't waiting on
#      a lookup or a dispatch; it's waiting on Gemini forming a sentence.
#      A filler here isn't buying the user information, just noise.
#
#   2. Casual chit-chat never matches a fast or semantic protocol, so it
#      ALWAYS falls through to the Smart Path — and the Smart Path is
#      measured well past FIRST_LINE_DELAY_SECONDS on a typical call.
#      That means the generic line fired on *every single* "how are you"
#      or "what are you doing", not occasionally: a real, systematic
#      pattern, not a one-off. Worse for exactly this category, because
#      "working on it, sir" sitting where an answer to "what are you
#      doing" should be reads as a non-answer to the very question asked
#      — the recursion is what makes it sound broken rather than merely
#      unnecessary.
#
# Deliberately multi-word phrases, matching this file's existing
# preference for narrow-but-safe substrings over single common words —
# "what are you doing" cannot false-positive inside a real command the
# way a bare "are" or "you" would.
_SMALL_TALK_PHRASES: tuple[str, ...] = (
    "what are you doing", "what are you up to",
    "how are you", "how's it going", "hows it going", "how are things",
    "who are you", "what are you",
    "are you there", "are you awake", "are you okay",
    "what's up", "whats up",
    "can you hear me",
)

# GENERIC LINES: used when the transcript matches no topic above. These say
# only that she is working, which is unambiguously true — she is, on the
# calling thread, right now. Kept short on purpose: this line is competing
# with the answer that is about to arrive behind it.
_GENERIC_FIRST_LINES: tuple[str, ...] = (
    "One moment, sir.",
    "Let me work that out.",
    "Thinking about that, sir.",
    "Just a moment.",
    "Working on it, sir.",
)

# SECOND LINES: only ever reached when work has run past
# SECOND_LINE_DELAY_SECONDS, by which point the user has already heard a
# first line and is waiting on a promise. These acknowledge the wait
# without claiming progress that isn't measurable from here.
_SECOND_LINES: tuple[str, ...] = (
    "Still working on it, sir.",
    "Bear with me, almost there.",
    "Nearly there, sir.",
)


def _pick(pool: Sequence[str], avoid: str | None) -> str:
    """
    Choose a line, never repeating the one just used.

    WHY NOT PLAIN random.choice: with a five-line pool, plain random
    selection repeats back-to-back about one time in five. A voice
    assistant that says "One moment, sir" twice running does not sound
    varied — it sounds stuck, which is the precise impression this whole
    module exists to prevent. Excluding the previous line costs nothing
    and removes the only failure mode anyone would actually notice.
    """
    candidates = [line for line in pool if line != avoid] or list(pool)
    return random.choice(candidates)


def _lines_for(transcribed_text: str) -> tuple[Sequence[str], Sequence[str]]:
    """
    Returns (first_line_pool, second_line_pool) for this request.

    An empty first pool is a real, meaningful return value — it means
    "say nothing, however long this takes." Checked first and separately
    from _TOPIC_LINES: small talk about her wouldn't match any topic
    anyway, but being explicit here means it can never accidentally start
    matching one later just because a future topic's keywords happen to
    overlap.
    """
    lowered = transcribed_text.lower()
    if any(phrase in lowered for phrase in _SMALL_TALK_PHRASES):
        return (), ()
    for keywords, lines in _TOPIC_LINES:
        if any(keyword in lowered for keyword in keywords):
            return lines, _SECOND_LINES
    return _GENERIC_FIRST_LINES, _SECOND_LINES


# =============================================================================
# THE NARRATOR
# =============================================================================
class ProgressNarrator:
    """
    Speaks a truthful progress line if — and only if — the work it is
    wrapping has already proven slow.

    LIFECYCLE, and why it is start/finish rather than a context manager:
    _route_and_respond's Smart Path branch is a single blocking call with
    an error branch after it, and a `with` block around it would put the
    narrator's cleanup inside the same scope as the answer's speak_async()
    call — which is exactly the ordering that must not happen, since
    finish() has to complete BEFORE the answer starts talking. Explicit
    start()/finish() makes that ordering visible at the call site instead
    of implicit in a __exit__.

    THREAD SAFETY: timers fire on their own threads and can race finish()
    — the whole point is that nobody knows in advance which happens first.
    Every mutation of the timer list, the done flag, and the in-flight
    thread handle is under a single lock, and the timer callback re-checks
    the done flag INSIDE that lock before speaking. That closes the window
    where a timer that had already begun executing when finish() ran could
    still get a line out after the answer started.
    """

    def __init__(
        self,
        speak_fn: Callable[[str], threading.Thread],
        first_delay_seconds: float = FIRST_LINE_DELAY_SECONDS,
        second_delay_seconds: float = SECOND_LINE_DELAY_SECONDS,
    ) -> None:
        self._speak_fn = speak_fn
        self._first_delay = first_delay_seconds
        self._second_delay = second_delay_seconds

        self._lock = threading.Lock()
        self._timers: list[threading.Timer] = []
        self._done = True
        self._in_flight: threading.Thread | None = None
        self._last_line: str | None = None
        self._spoke_anything = False

    # -- public -----------------------------------------------------------
    def start(self, transcribed_text: str) -> None:
        """
        Arm the timers and return immediately.

        Costs a lock acquisition and two Timer constructions — microseconds,
        on the thread that is about to block on the network anyway. Nothing
        is synthesized, nothing is spoken, and no audio device is touched
        unless a timer actually fires.
        """
        first_pool, second_pool = _lines_for(transcribed_text)
        if not first_pool:
            # Small talk (see _lines_for) — nothing to say, however long
            # this takes. No Timer objects, no lock beyond marking done
            # so a stray finish() call afterward is a safe no-op rather
            # than operating on stale state from the PREVIOUS request.
            with self._lock:
                self._done = True
                self._timers = []
            return

        first_line = _pick(first_pool, avoid=self._last_line)
        second_line = _pick(second_pool, avoid=first_line)

        with self._lock:
            self._done = False
            self._in_flight = None
            self._spoke_anything = False
            self._timers = [
                threading.Timer(self._first_delay, self._say, args=(first_line,)),
                threading.Timer(self._second_delay, self._say, args=(second_line,)),
            ]
            timers = list(self._timers)

        for timer in timers:
            timer.daemon = True
            timer.start()

    def finish(self, drain_timeout_seconds: float = LINE_DRAIN_TIMEOUT_SECONDS) -> bool:
        """
        Stop narrating and wait for any line already talking to finish.

        Call this the instant the wrapped work returns and BEFORE speaking
        the answer. Returns whether anything was actually said, which the
        caller can use for logging.

        THE JOIN IS THE POINT: Piper playback opens an output stream on the
        speaker device. If the answer's speak_async() ran while a progress
        line was still playing, two streams would contend for one device —
        at best the lines overlap into mush, at worst sounddevice raises.
        Joining the in-flight line first serializes them. In the common
        case the line finished long before the Gemini call returned and
        this join is an immediate no-op.
        """
        with self._lock:
            self._done = True
            timers, self._timers = self._timers, []
            in_flight = self._in_flight
            spoke = self._spoke_anything

        for timer in timers:
            timer.cancel()

        if in_flight is not None and in_flight.is_alive():
            in_flight.join(drain_timeout_seconds)

        return spoke

    # -- internal ---------------------------------------------------------
    def _say(self, line: str) -> None:
        """Timer callback. Runs on the Timer's own thread."""
        with self._lock:
            # finish() may have run between this timer firing and this
            # lock being acquired — in which case the answer is already
            # on its way to the speaker and this line must be dropped.
            if self._done:
                return
            self._last_line = line
            self._spoke_anything = True
            self._in_flight = self._speak_fn(line)
