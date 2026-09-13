"""
=============================================================================
WATCHDOG — keeps a Citra process alive across an unattended run
=============================================================================
jarvis_voice_assistant.py's own top-level entry point only catches
KeyboardInterrupt around its main loop (see the bottom of that file) —
deliberately, since swallowing arbitrary exceptions deep inside the state
machine risks silently masking a real bug rather than surfacing it. But
that means an unexpected exception (a transient network blip that manages
to propagate further than expected, a Windows audio-device hiccup, etc.)
takes the whole process down with nothing to bring it back, which is a real
problem specifically for an overnight unattended run with nobody at the
keyboard to notice and restart it.

This script is the answer to that gap, kept DELIBERATELY separate from the
application code rather than adding generic exception recovery inside the
state machine: an external supervisor that restarts a crashed process is a
well-understood, low-risk pattern (it can't introduce a new bug into
working application logic, because it never touches that logic — it only
watches a subprocess's exit code). If the app's own logic is buggy, that
bug still shows up in the log exactly as before; this only stops one crash
from being the end of the whole night.

Restart backoff is escalating (2s, 4s, 8s... capped at 60s) specifically
to prevent a crash-LOOP: a process that fails immediately on every launch
(e.g. a real startup bug, a port already in use) would otherwise be
relaunched in a tight loop, burning CPU and spamming the log instead of
actually giving anyone a chance to see what's wrong. A crash after running
successfully for a while resets the backoff back to the short end, since
that's evidence the process CAN run fine and this was likely transient.

Usage:
    python watchdog.py "<command>" [log_file_path]

Runs `<command>` (a single shell string) via the same shell the parent
process uses, restarting it whenever it exits, until Ctrl+C.

log_file_path (OPTIONAL — added for running headless via pythonw.exe, see
AUTOSTART.md): when given, the supervised process's stdout/stderr are
redirected there (append mode) instead of inherited from this process's
own console. This matters specifically because pythonw.exe has NO
console at all — without a log file, a process launched that way produces
zero visible output anywhere, not even an error if it fails to start.
Rotated at MAX_LOG_BYTES (renamed to <path>.old, previous .old discarded)
so a process meant to run for weeks unattended doesn't grow an unbounded
log file. Watchdog's OWN status messages (this file's own `logger` calls)
are unaffected by this argument and still go to whatever console is
present — there usually isn't one when this argument is used, so those
specific messages are simply not visible in that case, which was judged
an acceptable loss: they're operational noise ("restarting in Ns"), not
the application's actual output, which IS captured.
"""
import logging
import os
import subprocess
import sys
import time
from collections import deque

import citra_logging

citra_logging.configure()
logger = logging.getLogger("watchdog")

MIN_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
# A crash within this long after a successful start doesn't reset the
# backoff to the minimum — it's still treated as part of the same
# escalating sequence, so a process that crashes every ~10s (too fast to
# ever look "stable" but not instant enough to look like a hard failure)
# still gets backed off rather than hammered.
STABLE_RUNTIME_SECONDS = 120.0
# Hard stop if restarts happen faster than this, REGARDLESS of backoff —
# a final safety net in case the backoff math itself has a bug. This is
# what actually prevents an infinite crash-loop from running all night.
MAX_RESTARTS_PER_HOUR = 30
MAX_LOG_BYTES = 20 * 1024 * 1024  # 20MB — see log_file_path's comment above


def _rotate_log_if_needed(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > MAX_LOG_BYTES:
        rotated = path + ".old"
        if os.path.exists(rotated):
            os.remove(rotated)
        os.replace(path, rotated)


def run_supervised(command: str, log_file_path: str | None = None) -> None:
    backoff = MIN_BACKOFF_SECONDS
    restart_times: deque[float] = deque()

    logger.info("Starting supervised process: %s", command)
    if log_file_path:
        logger.info("Supervised process output redirected to: %s", log_file_path)

    while True:
        restart_times.append(time.monotonic())
        while restart_times and time.monotonic() - restart_times[0] > 3600:
            restart_times.popleft()
        if len(restart_times) > MAX_RESTARTS_PER_HOUR:
            logger.error(
                "Restarted %d times in the last hour (limit %d) — stopping. "
                "This is almost certainly a real startup bug, not a transient "
                "failure; check the process's own log output above.",
                len(restart_times), MAX_RESTARTS_PER_HOUR,
            )
            return

        start_time = time.monotonic()

        log_handle = None
        if log_file_path:
            _rotate_log_if_needed(log_file_path)
            # line-buffered (buffering=1) so a tail -f-style view stays
            # live rather than waiting on a full buffer to flush.
            log_handle = open(log_file_path, "a", buffering=1, encoding="utf-8", errors="replace")

        try:
            # shell=True: `command` is a single string the caller controls
            # directly (passed on the command line at watchdog launch time,
            # not derived from any untrusted input), the same trust boundary
            # as any other local shell script.
            process = subprocess.Popen(command, shell=True, stdout=log_handle, stderr=log_handle)

            try:
                exit_code = process.wait()
            except KeyboardInterrupt:
                logger.info("Watchdog interrupted — terminating supervised process.")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return
        finally:
            if log_handle is not None:
                log_handle.close()

        runtime = time.monotonic() - start_time

        if exit_code == 0:
            logger.info("Process exited cleanly (code 0) after %.0fs — not restarting.", runtime)
            return

        logger.warning("Process exited with code %s after %.0fs.", exit_code, runtime)

        if runtime >= STABLE_RUNTIME_SECONDS:
            # Ran long enough to be considered genuinely stable before
            # this crash — treat it as a fresh problem, not a continuation
            # of a previous crash-loop.
            backoff = MIN_BACKOFF_SECONDS

        logger.info("Restarting in %.0fs...", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python watchdog.py "<command>" [log_file_path]')
        sys.exit(1)
    run_supervised(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
