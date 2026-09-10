"""
=============================================================================
CITRA SMART ROOM SYSTEM — V2 VOICE ASSISTANT
=============================================================================
Runs on: Windows 11 Pro laptop, alongside jarvis_hardware_api.py and (for
the Smart Path fallback) LM Studio running locally.

WHY THIS FILE EXISTS
---------------------
jarvis_router.py (V1) is a text-in, text-out dual-path router: type a
command, get a result. This file is the voice layer on top of it — it adds
ears (wake word + speech-to-text), a mouth (offline TTS), and swaps regex
matching for SEMANTIC matching, because "andhera ho raha hai" (Hindi for
"it's getting dark") has no keyword in common with "turn on the lights" —
only a shared MEANING, which is exactly what sentence embeddings capture
and regex cannot.

THE FOUR STAGES, IN ORDER:
  1. EAR & TRANSLATOR    — openWakeWord listens for the wake word
                            continuously; on trigger, records until
                            silence, then faster-whisper transcribes to
                            text (English, Hindi, or Hinglish).
  2. SEMANTIC ROUTING    — sentence-transformers embeds the transcribed
                            text and compares it against a bank of example
                            phrases per protocol (LIGHTING_PROTOCOL,
                            COOLING_PROTOCOL, ...). Highest similarity
                            above a threshold wins; below threshold falls
                            through to the Smart Path.
  3. STATE-AWARE HANDLER — the matched protocol's handler function checks
                            CURRENT hardware state via SmartRoomController,
                            picks the right one of several possible spoken
                            responses based on that state, THEN fires the
                            hardware command and the speech in parallel
                            threads.
  4. SMART PATH FALLBACK — anything below the similarity threshold gets
                            forwarded to jarvis_router.py's LM Studio
                            integration, and the LLM's text answer is
                            spoken back through the same TTS path.

!!! IMPORTANT — READ BEFORE YOU ASSUME "EVERYTHING IS INSTANT" !!!
-----------------------------------------------------------------------
This file implements REAL, GENUINE parallelism between speaking and
hardware dispatch — once the handler has decided what to say, the TTS
thread and the hardware-command thread are started back-to-back and run
CONCURRENTLY, so your lights/AC genuinely start switching while Jarvis is
still mid-sentence. That part is real.

What is NOT parallel, and cannot be made parallel, is the STATE CHECK that
runs before the handler decides what to say. To know whether to say
"turning on the AC" vs "the cooling system is already running," the code
must first ask the NodeMCUs what their current state is — a real HTTP
round-trip over Wi-Fi, typically single-digit-to-tens of milliseconds, but
never zero, and not something any amount of threading can make simultaneous
with a sentence whose WORDING depends on that round-trip's answer. This is
a causality constraint, not a performance bug: you cannot speak a sentence
that depends on information you don't have yet.

The honest timeline for a typical command is:
    [state check: ~10-50ms, sequential]  ->  [speak + dispatch: parallel]
This is called out explicitly in COOLING_PROTOCOL's handler below, since
that's the handler with the most branching state logic.

REQUIRED PIP INSTALLS (all offline/local once models are downloaded —
model weights ARE downloaded from the internet on first run, but nothing
after that requires a network connection for these libraries specifically):

    pip install openwakeword pyaudio faster-whisper sentence-transformers piper-tts sounddevice numpy
    python -m piper.download_voices en_US-ryan-medium

Platform notes for Windows 11:
  - pyaudio: if `pip install pyaudio` fails to build, install the
    prebuilt wheel instead: `pip install pipwin && pipwin install pyaudio`
    (a common Windows-specific gotcha — pyaudio's C extension often has no
    prebuilt wheel for the newest Python versions on PyPI directly).
  - piper-tts + sounddevice replace pyttsx3 for TTS — piper-tts is a
    pure ONNX Runtime package (no separate native binary to manage) and
    installs cleanly via pip on Windows. sounddevice wraps PortAudio,
    same as pyaudio, so if pyaudio's build worked, sounddevice's
    prebuilt wheels should too.
  - faster-whisper and sentence-transformers will pull in PyTorch, which
    is a large download (~2-3GB) the first time — this only happens once.

This file imports SmartRoomController from jarvis_hardware_api.py and
JarvisRouter/LM_STUDIO_* from jarvis_router.py — both must be in the same
directory or importable on your PYTHONPATH.
=============================================================================
"""

import io
import os
import re
import json
import threading
import time
import wave
import logging
import queue
import datetime
import difflib
import http.server
from collections import deque

# -----------------------------------------------------------------------------
# HUGGING FACE OFFLINE MODE — set BEFORE any HF-backed library is imported
# -----------------------------------------------------------------------------
# Measured from a real startup log: 36 network round-trips to huggingface.co
# on EVERY launch, purely to re-check whether already-cached models had
# changed. They never change. That check dominated startup time —
# Whisper took 23.5-33.5s and the sentence-transformers model 16-30.5s,
# versus 3.4s and 3.7s respectively with HF_HUB_OFFLINE=1 and identical
# cached weights. Total measured startup was 51-108s; this is the single
# biggest contributor to it.
#
# Set CONDITIONALLY, not unconditionally: offline mode makes a cache MISS
# a hard failure instead of a download, so forcing it on would break the
# very first run on a fresh machine — which matters directly for the
# "install this in other households" goal, where a first run with an
# empty cache is the NORMAL case, not an edge case. Detecting whether the
# cache is already populated means an established install gets the fast
# path automatically while a fresh one still downloads normally, with no
# per-machine configuration step to remember.
#
# faster_whisper and sentence_transformers are both imported lazily
# (inside run()/SemanticRouter.__init__, not at module scope), which is
# what makes setting this here — at import time of THIS file — early
# enough to take effect.
_HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
if os.environ.get("HF_HUB_OFFLINE") is None and os.path.isdir(_HF_CACHE_DIR):
    if any(name.startswith("models--") for name in os.listdir(_HF_CACHE_DIR)):
        os.environ["HF_HUB_OFFLINE"] = "1"

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

import numpy as np
import requests

from jarvis_hardware_api import SmartRoomController, HardwareResult
from jarvis_router import JarvisRouter
from jarvis_reminders import ReminderController
from jarvis_presence import ProgressNarrator
import citra_mute
import citra_ui_bridge


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("jarvis_voice")


# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# OPENWAKEWORD (WAKE WORD)
# -----------------------------------------------------------------------------
# Replaces pvporcupine — Picovoice's console now gates free signup behind a
# company email and only offers a 7-day Enterprise trial, which doesn't fit
# a permanent personal project. openWakeWord is fully open-source (no
# account, no key, no expiry) and, notably, already ships an official
# "hey_jarvis" pre-trained model — so unlike the Porcupine setup, you don't
# need to train anything custom to get the exact wake word you want.
#
# Pre-trained model names ship as part of the package (downloaded once via
# download_models() the first time this runs, then cached locally — fully
# offline after that). "hey_jarvis", "alexa", and "hey_mycroft" remain
# available as built-in fallbacks if the custom model below ever needs
# to be swapped out.
#
# !!! CUSTOM "HEY CITRA" MODEL — TRAINED, NOT BUILT-IN !!!
# openWakeWord ships no "hey_citra" model, so one was trained from scratch
# using openWakeWord's own automated training pipeline. THIS IS ROUND 3.
#
#   Round 1 (~10k/~10k synthetic-only samples): ~77% accuracy / ~55%
#   recall / ~0.88 FP/hour. Real-world testing showed what that FP
#   number meant in practice: Citra kept re-triggering on her own "Yes
#   sir?" TTS echo, sometimes at HIGHER confidence (0.91) than a genuine
#   "hey citra" utterance — raising the threshold alone couldn't fix it
#   since the score ranges genuinely overlapped.
#
#   Round 2 (~30k/~30k, still synthetic-only): added Citra's own spoken
#   phrases ("yes sir", "citra online", etc.) as explicit negative
#   training examples. Fixed the echo problem completely, but recall
#   dropped to ~41% on (synthetic) validation.
#
#   Round 3 (this one, ~100k/~100k): added REAL recorded audio -- ~100
#   actual "hey citra" utterances and ~90 real background/other-speech
#   clips, recorded via record_my_voice.py in the actual deployment
#   room through the actual mic, oversampled (duplicated before
#   augmentation) so they carry real weight against the much larger
#   synthetic set. Also bumped layer_size 32 -> 64. Measured against
#   REAL held-out recordings (not synthetic TTS -- the 20 real "hey
#   citra" clips and 13 real negative clips set aside as validation,
#   never touched by training): a clean separation between the two --
#   every real negative sample scored below 0.0016, while genuine "hey
#   citra" utterances ranged much higher. That gap directly set
#   WAKE_WORD_DETECTION_THRESHOLD below at 0.06 (a ~37x margin over the
#   worst real negative observed), which real-data testing measured at
#   a 94% catch rate on genuine attempts with ZERO false positives
#   across every real negative clip tested.
#
# Net effect: this should feel meaningfully more reliable than round 2
# on the very first "hey citra" attempt, specifically because it was
# tuned against real speech in the real room, not just synthetic TTS
# validation data. If real-world use still disagrees with this
# benchmark, the next lever is more real recordings (record_my_voice.py
# again, then re-fold via fold_in_real_voice.py in ~/citra_wakeword),
# not just nudging the threshold further -- the full pipeline (config,
# scripts, downloaded datasets) is preserved in the WSL Ubuntu
# environment's ~/citra_wakeword directory for exactly that.
WAKE_WORD_MODEL_NAME = "hey_citra.onnx"  # path to the custom-trained
                                  # model (same directory as this file).
                                  # openWakeWord accepts either a bare
                                  # official model NAME (resolved via its
                                  # own download cache, e.g. "hey_jarvis")
                                  # or a literal path to an .onnx file
                                  # like this one — both go in the same
                                  # wakeword_models=[...] list in
                                  # _create_openwakeword_model() below.
                                  # Revert to "hey_jarvis" here (one-line
                                  # change) if the false-positive rate
                                  # above proves impractical day to day.
WAKE_WORD_INFERENCE_FRAMEWORK = "onnx"  # "onnx" works on both Windows and
                                  # Linux; "tflite" is Linux/Arm64-only
                                  # and not supported on modern Windows —
                                  # "onnx" is the correct choice for your
                                  # Windows 11 laptop.
WAKE_WORD_DETECTION_THRESHOLD = 0.5  # 0.0-1.0. RETRAINED MODEL (see
                                  # hey_citra.yaml) — this is a normal,
                                  # unremarkable value again, which is
                                  # itself the signal that the model is
                                  # now actually converged. The previous
                                  # model (max_negative_weight=2000,
                                  # auto-escalated to 8000x by openWake-
                                  # Word's own training loop) never
                                  # learned to separate the classes in any
                                  # normal probability range, which is why
                                  # this constant previously had to sit at
                                  # 0.003 just to be usable at all — that
                                  # was a workaround for a broken model,
                                  # not a real operating point. Fixed by
                                  # lowering max_negative_weight to 25 and
                                  # retraining (reused the already-cached
                                  # augmented features, so no re-synthesis
                                  # needed) rather than tuning around it
                                  # again.
                                  #
                                  # Measured the same way as before —
                                  # streaming the 100 real "hey citra" /
                                  # 90 real room-noise recordings through
                                  # this exact model, 80ms frames in
                                  # order, 2s of real-noise warmup first —
                                  # but the result this time is a properly
                                  # separated model, not a crutch:
                                  #     genuine "hey citra" peak scores:
                                  #         p10 0.908, median 0.989, max 0.991
                                  #     real room-noise peak scores:
                                  #         median 0.003, p90 0.004, max 0.0099
                                  # Recall was measured (with
                                  # WAKE_MIN_SPEECH_RMS's energy gate also
                                  # applied, matching production exactly)
                                  # at every threshold from 0.02 to 0.6:
                                  # it is FLAT at 96-97% across that
                                  # entire range, with ZERO false-positive
                                  # clips at every single one. 0.5 is
                                  # chosen from deliberately deep inside
                                  # that plateau — 50x above the worst
                                  # real noise ever observed — rather than
                                  # from the edge of it, because the 90
                                  # negative recordings are a small,
                                  # narrow sample (one room, one session)
                                  # that can't characterize every noise
                                  # type this will ever meet, and margin
                                  # against the unknown matters more here
                                  # than the <1 percentage point of recall
                                  # separating 0.5 from the plateau's edge.
                                  #
                                  # CAUTION FOR FUTURE VALIDATION: an
                                  # onnxruntime InferenceSession loads a
                                  # model's large layers from a companion
                                  # hey_citra.onnx.data file, referenced by
                                  # that EXACT literal filename baked into
                                  # the graph — NOT derived from whatever
                                  # you name the .onnx file. Renaming
                                  # "hey_citra.onnx" to compare two model
                                  # versions side by side (e.g. to
                                  # "hey_citra_new.onnx") while an OLD
                                  # "hey_citra.onnx.data" is still sitting
                                  # in the same directory silently loads
                                  # the NEW graph with the OLD weights —
                                  # onnxruntime does not error, and the
                                  # resulting Frankenstein model can
                                  # produce plausible-looking scores. This
                                  # happened once during this retrain's
                                  # own validation and was only caught
                                  # because the NEXT step (overwriting the
                                  # live file) made the mismatch loud.
                                  # Always compare model versions from
                                  # SEPARATE directories, each using the
                                  # model's real, unrenamed filenames.

# -----------------------------------------------------------------------------
# INDIVIDUAL LIGHT NAMING (used by _extract_target_relays, near
# _handle_lighting_on_protocol / _handle_lighting_off_protocol)
# -----------------------------------------------------------------------------
LIGHT_NAME_ALIASES = {
    # Relay 1 = Warm (brightest), Relay 2-4 = White groups — per the
    # confirmed switchboard wiring. Update this table directly if the
    # physical wiring or your preferred names for each light change;
    # nothing else needs to change to support new names.
    "warm": {1},
    "bright": {1},
    "brightest": {1},
    "white": {2, 3, 4},
    "dim": {2, 3, 4},
    "dimmest": {4},
}

_LIGHT_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}

# -----------------------------------------------------------------------------
# AUDIO RECORDING (SILENCE-DETECTION PARAMETERS)
# -----------------------------------------------------------------------------
# After wake-word detection, we record until the user stops talking. This
# is a simple energy-based silence detector, not a full VAD (Voice Activity
# Detection) model — good enough for a quiet room, and avoids adding
# another model dependency for something this constrained use case doesn't
# strictly need.
# -----------------------------------------------------------------------------
# NEURAL VOICE ACTIVITY DETECTION (Silero VAD)
# -----------------------------------------------------------------------------
# Replaces the RMS-threshold end-of-speech detection below, which was
# measurably broken in this specific room rather than merely imprecise.
#
# THE MEASUREMENT THAT FORCED THIS: with the AC running (this room's
# normal state — its own background hum sits around RMS 700, ABOVE
# SILENCE_THRESHOLD's 500), the RMS detector classified 62 of 62 frames
# of pure room noise as "speech" with nobody talking at all. Silero
# classified 0 of 62 the same way. A detector that never sees silence
# never ends a recording, so every command ran to MAX_RECORDING_SECONDS.
# The reverse failure showed up too: real logs contain recordings ending
# at 0.96s — shorter than it takes to say "turn on the lights" — which
# truncate the command mid-word and produce the garbled transcriptions
# seen in those same logs. One fixed loudness threshold cannot separate
# "quiet speech" from "loud room", so tuning it only ever traded one of
# those failures for the other. That's a mechanism problem, not a
# constant-tuning problem, which is why this is a new mechanism.
#
# Cost, measured: 1.3s one-time load (on the background thread with the
# other models, so it's off the startup path), then 0.25ms per 32ms of
# audio — roughly 0.8% of one core, i.e. free relative to what it fixes.
#
# Silero's native frame unit is 512 samples @16kHz; this pipeline's is
# 1280 (80ms), which is not a multiple of 512. _SpeechDetector buffers
# across calls and evaluates whole 512-sample chunks as they become
# available, rather than resampling or zero-padding either side to force
# a fit (both of which would feed the model audio it wasn't trained on).
VAD_SPEECH_PROBABILITY_THRESHOLD = 0.5  # silero's own documented default.
                                # Deliberately NOT tuned away from it
                                # without evidence: the model outputs
                                # well-separated probabilities (measured
                                # 0.00-0.01 on room noise, 0.86-1.00 on
                                # speech), so there's a wide margin here
                                # and nothing to gain from moving it.


class _SpeechDetector:
    """
    Neural VAD wrapper. `is_speech()` returns True/False per pipeline
    frame, or None if the model isn't available — that None is the signal
    for callers to fall back to the old RMS threshold, so a missing or
    broken silero install degrades to previous behavior instead of
    breaking speech detection outright.
    """

    def __init__(self) -> None:
        self._model = None
        self._unavailable = False
        self._buffer = np.zeros(0, dtype=np.float32)

    def load(self) -> None:
        """Called from the background model-loading thread; safe to call
        more than once."""
        if self._model is not None or self._unavailable:
            return
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
            logger.info("Silero VAD loaded.")
        except Exception as exc:
            self._unavailable = True
            logger.warning(
                "Silero VAD unavailable (%s) — falling back to the RMS "
                "threshold. Speech detection will be less reliable in a "
                "noisy room; `pip install silero-vad` to fix.", exc,
            )

    def reset(self) -> None:
        """Clears buffered audio AND the model's own internal recurrent
        state between recordings. Silero is stateful across calls — a
        fresh utterance must not inherit the previous one's context, the
        same class of stale-state bug already documented for
        openWakeWord's feature buffer in _consume_frame_cooldown."""
        self._buffer = np.zeros(0, dtype=np.float32)
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    def is_speech(self, raw_frame: bytes) -> Optional[bool]:
        if self._model is None:
            return None
        try:
            import torch
            samples = np.frombuffer(raw_frame, dtype=np.int16).astype(np.float32) / 32768.0
            self._buffer = np.concatenate((self._buffer, samples))

            speech = False
            evaluated_any = False
            while len(self._buffer) >= 512:
                chunk, self._buffer = self._buffer[:512], self._buffer[512:]
                prob = self._model(torch.from_numpy(chunk), 16000).item()
                evaluated_any = True
                # ANY sub-chunk counting as speech makes the whole frame
                # speech: this feeds a "have we been silent long enough"
                # counter, so being conservative here means erring toward
                # "still talking" — i.e. toward letting someone finish,
                # which is the failure direction that matters.
                if prob >= VAD_SPEECH_PROBABILITY_THRESHOLD:
                    speech = True

            if not evaluated_any:
                # Not enough buffered audio for a full chunk yet — report
                # the previous verdict's safest interpretation rather than
                # inventing one. Treated as "not silence" so a partially
                # filled buffer can never itself advance the silence
                # counter toward ending a recording early.
                return True
            return speech
        except Exception as exc:
            logger.warning("Silero VAD inference failed (%s) — falling back to RMS.", exc)
            self._model = None
            self._unavailable = True
            return None


SILENCE_THRESHOLD = 500       # RMS amplitude below which audio is
                                # considered "silence". 16-bit PCM audio
                                # ranges roughly -32768 to 32767; typical
                                # room background noise sits well under
                                # this value, typical speech well over it.
                                # Tune this up if your room is noisy and
                                # recording never stops; tune it down if
                                # recording cuts off mid-sentence.
SILENCE_DURATION_SECONDS = 0.8  # how long the user must stay quiet before
                                  # we consider them done talking. Lowered
                                  # from 1.5s during a latency pass — this
                                  # is DEAD AIR added to every single
                                  # command, and at 1.5s it was actually
                                  # the single biggest contributor in the
                                  # whole pipeline, bigger than Whisper
                                  # transcription itself (measured ~0.5-
                                  # 0.9s on this machine). 1.0s is a
                                  # common baseline for voice assistants
                                  # and still comfortably longer than a
                                  # natural mid-sentence breath pause.
                                  #
                                  # Then 1.0 -> 0.8 chasing a 1-second
                                  # target. 0.8s is at the aggressive end
                                  # of comfortable: it's still longer than
                                  # a typical breath pause, but this is
                                  # now the FIRST thing to raise back up
                                  # if she starts cutting you off before
                                  # you've finished a sentence. That
                                  # failure mode is far more annoying than
                                  # 200ms of extra wait, so treat this as
                                  # the tuned floor, not a target to keep
                                  # pushing down.
NO_SPEECH_TIMEOUT_SECONDS = 4.0  # how long to wait for the command to
                                  # actually START after the wake word
                                  # before giving up and discarding.
                                  # Distinct from SILENCE_DURATION_SECONDS,
                                  # which only governs the gap AFTER speech
                                  # has begun — see _consume_frame_listening
                                  # for why conflating the two ended every
                                  # recording at exactly 0.8s. Generous on
                                  # purpose: this is the "user said the
                                  # wake word then paused to think" case,
                                  # and cutting that off is worse than
                                  # waiting a beat longer on the rare
                                  # genuine false wake.
MAX_RECORDING_SECONDS = 10.0    # hard ceiling so a stuck/failed silence
                                  # detection can't record forever

# -----------------------------------------------------------------------------
# BARGE-IN (LAYER 2) PARAMETERS
# -----------------------------------------------------------------------------
# Reuses the SAME RMS-energy approach as SILENCE_THRESHOLD above, applied
# in reverse: instead of "how quiet before we consider the user done
# talking", this asks "how loud, while Jarvis is speaking, before we
# consider the user trying to interrupt him". A dedicated neural VAD
# (e.g. Silero) would be more accurate, but this project already has a
# working energy-threshold primitive and introducing a new model
# dependency for this specific feature wasn't asked for — if false
# barge-ins become a real problem in practice, swapping this check for a
# proper VAD is a contained change (only _reader_thread_main's SPEAKING
# branch would need to change).
BARGE_IN_ENERGY_THRESHOLD = 3000  # RMS amplitude while Jarvis is
                                    # speaking, above which we treat it
                                    # as the user trying to interrupt.
                                    # Deliberately HIGHER than
                                    # SILENCE_THRESHOLD (500) — Jarvis's
                                    # own voice is coming out of the
                                    # speaker and may leak into the mic
                                    # at a nontrivial level (acoustic
                                    # echo), so this threshold needs to
                                    # sit above typical echo level to
                                    # avoid Jarvis interrupting himself.
                                    # If you notice Jarvis stopping
                                    # mid-sentence with no one talking,
                                    # raise this. If you can't interrupt
                                    # him at all, lower it.
BARGE_IN_CONSECUTIVE_FRAMES_REQUIRED = 8  # how many consecutive 80ms
                                    # frames must exceed the energy
                                    # threshold before triggering a
                                    # barge-in, rather than acting on a
                                    # single loud frame (a door slam, a
                                    # cough). 3 frames = ~240ms of
                                    # sustained loud audio — long enough
                                    # to filter out a single transient
                                    # sound, short enough that a real
                                    # interruption still feels instant.
COOLDOWN_SECONDS = 5.0  # time spent in the COOLDOWN state after speech
                                    # ends (whether it finished naturally
                                    # or was interrupted) before
                                    # returning to IDLE and re-arming
                                    # wake-word detection. This exists
                                    # SPECIFICALLY to close the original
                                    # TTS-echo hallucination loop: without
                                    # it, Citra's own trailing audio
                                    # (room reverb, speaker/mic bleed)
                                    # could immediately re-trigger the
                                    # wake word the instant she stops
                                    # talking. Raised 1.5s -> 3.0s -> 5.0s
                                    # across repeated real-world rounds —
                                    # 3.0s STILL wasn't enough margin
                                    # against real room reverb tail, per
                                    # a real session log showing the
                                    # "yes sir -> re-trigger -> yes sir"
                                    # loop continuing even at 3.0s + a
                                    # 0.5 threshold. This alone is a
                                    # mitigation, not a full fix — see
                                    # WAKE_WORD_CONSECUTIVE_FRAMES_REQUIRED
                                    # below for the other, structurally
                                    # different lever added alongside it.
WAKE_MIN_SPEECH_RMS = 300  # a wake is REJECTED unless real sound was heard
                                    # in the last WAKE_ENERGY_WINDOW_FRAMES.
                                    #
                                    # This exists because the wake-word model
                                    # demonstrably fires on SILENCE. Measured
                                    # from the live room (AC running, nobody
                                    # speaking): RMS median 0, max 12 across
                                    # 6 seconds — acoustically nothing. Over
                                    # the same conditions the model was
                                    # emitting wake scores of 0.004 up to
                                    # 0.129, i.e. hallucinating the wake word
                                    # out of an empty room, and doing it on a
                                    # perfect 5.1s cycle (COOLDOWN_SECONDS +
                                    # one frame) because it re-fired the
                                    # instant it re-armed.
                                    #
                                    # Crucially those spurious scores OVERLAP
                                    # the genuine ones from the same session
                                    # (real commands woke at 0.009, 0.011,
                                    # 0.015, 0.017, 0.054), so no threshold
                                    # can separate them — the model simply
                                    # cannot tell the difference, and tuning
                                    # WAKE_WORD_DETECTION_THRESHOLD can only
                                    # trade one failure for the other. That is
                                    # exactly the loop this project kept going
                                    # around: too sensitive, then deaf, then
                                    # too sensitive again.
                                    #
                                    # Audio ENERGY separates them completely
                                    # where the model's own score cannot:
                                    # silence peaks at 12, speech sits near
                                    # 2878 — a ~240x gap with nothing in
                                    # between. 300 sits 25x above the loudest
                                    # observed silence and ~10x below normal
                                    # speech, so it rejects empty-room wakes
                                    # outright while leaving even a quiet or
                                    # distant "hey citra" comfortably through.
                                    # Deliberately BELOW SILENCE_THRESHOLD
                                    # (500, used for end-of-speech detection)
                                    # so a soft wake word is never dropped by
                                    # this gate.
                                    #
                                    # This is a guard around a weak model, not
                                    # a substitute for retraining it.
WAKE_ENERGY_WINDOW_FRAMES = 12  # ~1s of 80ms frames. The model usually fires
                                    # a frame or two AFTER the phrase, so the
                                    # energy check looks at a short window of
                                    # recent history rather than only the
                                    # triggering frame — which on its own can
                                    # legitimately be the quiet tail of the
                                    # utterance.

WAKE_WORD_CONSECUTIVE_FRAMES_REQUIRED = 1  # how many consecutive 80ms frames
                                    # must EACH score above the threshold
                                    # before a wake fires. 1 = no debounce.
                                    #
                                    # This was set to 3 on the reasoning
                                    # that a genuine "hey citra" spans
                                    # 6-12 frames so requiring 3 in a row
                                    # "costs essentially nothing". That
                                    # reasoning was wrong, and measurement
                                    # showed by how much: the model only
                                    # scores above threshold on the one or
                                    # two frames where the phrase actually
                                    # lands, not across the whole
                                    # utterance. Requiring 3 consecutive
                                    # frames cost roughly a THIRD of all
                                    # recall on real recordings (at
                                    # threshold 0.002: 98% with 1 frame,
                                    # 92% with 2, 44% with 3).
                                    #
                                    # It also wasn't buying anything: the
                                    # measured false-positive rate is 0.00
                                    # per minute at 1 frame anyway, across
                                    # every threshold tested, because real
                                    # room noise never gets close to even
                                    # the lowest useful threshold. A
                                    # debounce is the right tool when
                                    # noise spikes cross the bar; here
                                    # nothing crosses it, so it only threw
                                    # away genuine wakes.
                                    #
                                    # Kept as a constant rather than
                                    # deleted because it becomes useful
                                    # again after a retrain, when positive
                                    # scores are high enough to run a
                                    # threshold where noise can compete.

# -----------------------------------------------------------------------------
# CONTINUOUS LISTENING MODE (toggle-able alternate to wake-word spotting)
# -----------------------------------------------------------------------------
# Default mode stays wake-word-first (self._continuous_mode = False) — this
# is an opt-in ALTERNATE mode, switched into/out of by voice command (see
# CONTINUOUS_MODE_ON_PHRASES/OFF_PHRASES and the check in
# _route_and_respond). While off, _consume_frame_idle behaves exactly as
# before; while on, it runs _consume_frame_idle_continuous instead.
#
# WHY THIS NEEDS A DIFFERENT PIPELINE, NOT JUST A LOOSER WAKE PHRASE:
# The trained wake-word model (hey_citra.onnx) only knows how to recognize
# "hey citra" as a specific acoustic pattern — it has no way to recognize
# "citra" said in the middle of an arbitrary sentence like "alright citra,
# do me a favor...". Continuous mode instead transcribes every utterance
# (energy-triggered start, same SILENCE_THRESHOLD/DURATION used for
# end-of-speech already) and searches the TEXT for the name, which is what
# actually makes free-form mid-sentence addressing possible. The real cost
# is that every utterance gets fully transcribed by Whisper, not just ones
# following a recognized wake phrase — meaningfully heavier than the
# lightweight always-on classifier the default mode uses, which is exactly
# why this stays opt-in rather than becoming the default.
CONTINUOUS_MODE_ON_PHRASES = ("continuous mode", "always listen", "listen continuously", "keep listening")
CONTINUOUS_MODE_OFF_PHRASES = ("wake word mode", "stop continuous", "normal mode", "stop always listening")
CONTINUOUS_MODE_NAME = "citra"
CONTINUOUS_MODE_NAME_MATCH_THRESHOLD = 0.70  # FUZZY match, not an exact
                                    # substring check — real usage showed
                                    # Whisper transcribing "Citra" (said in
                                    # an Indian accent) inconsistently:
                                    # "sitra" sometimes, something else
                                    # other times, not one fixed
                                    # mis-spelling a hardcoded list could
                                    # ever fully cover. difflib.
                                    # SequenceMatcher's similarity ratio
                                    # against every word in the transcript
                                    # generalizes to variants nobody
                                    # explicitly listed, rather than
                                    # playing whack-a-mole with individual
                                    # spellings.
                                    #
                                    # 0.70 chosen by measuring, not
                                    # guessing: scored plausible
                                    # mis-transcriptions (sitra 0.80,
                                    # chitra 0.91, citara 0.91, cithra
                                    # 0.91, chithra 0.83, sitara 0.73,
                                    # citta 0.80) against a list of
                                    # ordinary conversational words of
                                    # similar length (water, letter,
                                    # winter, center, sister, picture,
                                    # nature, city, ...) — the highest any
                                    # of those scored was "city" at 0.667,
                                    # comfortably below every real variant
                                    # tested. 0.70 sits in that gap.
CONTINUOUS_MODE_VAD_CONSECUTIVE_FRAMES_REQUIRED = 3  # ~240ms. NOT the same
                                    # tradeoff as
                                    # WAKE_WORD_CONSECUTIVE_FRAMES_REQUIRED
                                    # being set to 1 above — that case was
                                    # a NARROW classifier score that only
                                    # spikes on the 1-2 frames where the
                                    # trained phrase acoustically lands, so
                                    # debouncing cost real recall. This is
                                    # raw RMS energy instead: genuine
                                    # speech sustains above SILENCE_THRESHOLD
                                    # for many consecutive frames (a spoken
                                    # sentence is not a 1-frame blip), so a
                                    # short debounce costs it essentially
                                    # nothing while filtering a single
                                    # transient (a cough, a door) that a
                                    # bare energy threshold can't otherwise
                                    # distinguish from speech onset.
CONTINUOUS_MODE_PREROLL_FRAMES = 3  # ~240ms of audio immediately BEFORE
                                    # the VAD trigger fires, prepended to
                                    # the recording. Needed because
                                    # crossing the debounce takes those
                                    # same few frames — without a pre-roll,
                                    # the very start of what was said would
                                    # be clipped every single time.


def _extract_command_without_name(text: str) -> Optional[str]:
    """
    If `text` mentions Citra's name (fuzzy-matched word-by-word — see
    CONTINUOUS_MODE_NAME_MATCH_THRESHOLD's comment for how that bar was
    calibrated), returns the text with that word removed. Returns None if
    the name isn't mentioned at all (the caller's signal to discard the
    segment).

    WHY THIS REMOVES THE WORD, NOT JUST DETECTS IT:
    Found via a real query: continuous mode passes the WHOLE transcribed
    utterance to semantic routing, unlike wake-word mode, where the
    trigger phrase is detected acoustically and never becomes part of the
    transcribed text at all. Leaving the name in measurably hurts every
    protocol match, not just an edge case — measured directly: "what's
    the weather outside, citra" scores 0.816 against WEATHER_PROTOCOL's
    phrases, versus 0.976 for the exact same question with "citra"
    removed. A real garbled transcript ("the weather outside Sitra.")
    scored 0.662 with the name in, 0.875 with it stripped. The extra,
    protocol-irrelevant proper noun measurably drags the embedding away
    from what the sentence is actually asking, in every single case
    tested — stripping it isn't a nice-to-have, it's what makes
    continuous-mode routing quality match wake-word mode's at all.
    """
    tokens = text.split()
    kept = []
    found_name = False
    for token in tokens:
        bare = re.sub(r"[^a-zA-Z]", "", token).lower()
        if len(bare) >= 3 and difflib.SequenceMatcher(None, bare, CONTINUOUS_MODE_NAME).ratio() >= CONTINUOUS_MODE_NAME_MATCH_THRESHOLD:
            found_name = True
            continue  # drop this token (and any attached punctuation) entirely
        kept.append(token)

    if not found_name:
        return None

    cleaned = " ".join(kept).strip()
    # If the name was literally the whole utterance ("Citra?" with
    # nothing else), fall back to the original text rather than routing
    # an empty string — _process_recording's existing empty-transcript
    # guard already handles "nothing meaningful was said" correctly.
    return cleaned or text


class AssistantState(Enum):
    """
    The five states the assistant cycles through, verified via isolated
    unit tests before being wired into real audio I/O below (see project
    notes). Exactly one thread — the main loop in _consumer_loop — ever
    changes self._state; the reader thread (_reader_thread_main) only
    READS self._state to decide what to do with each frame, never writes
    it. This single-writer discipline is what keeps the state machine
    safe to share between the two threads without needing a lock around
    every access.
    """
    IDLE = auto()        # waiting for wake word
    LISTENING = auto()   # wake word heard, recording the command
    PROCESSING = auto()  # transcribing + routing (no live audio decisions)
    SPEAKING = auto()    # Jarvis is talking; barge-in energy check active
    COOLDOWN = auto()    # brief pause after speech, before returning to IDLE

# -----------------------------------------------------------------------------
# FASTER-WHISPER (SPEECH-TO-TEXT)
# -----------------------------------------------------------------------------
# "base" is a genuine multilingual checkpoint (NOT "base.en", which is
# English-only) — this matters directly for your Hindi/Hinglish
# requirement. Do not append ".en" to this name.
# MEASURED on this machine (20 logical cores, tiny/int8, a 4.5s recording
# with the 0.8s of trailing silence endpointing actually produces):
#
#   beam_size=5, autodetect, default threads .... 602ms   <- what we shipped
#   + cpu_threads=20 + vad_filter + beam_size=1   422ms
#   + language forced ........................... 244ms
#
# Every one of those numbers came from running it, not from a datasheet.
# Two things people expect to matter did NOT: beam_size=1 on its own was
# 689ms vs 657ms (i.e. no better, and the transcribed words are byte-
# identical on every test phrase), and the trailing silence costs 5ms,
# not the hundreds you would guess. The whole gap is language detection.
WHISPER_CPU_THREADS = os.cpu_count() or 0
                              # 0 = let ctranslate2 decide. Set to the core
                              # count to force it. Measured here: default
                              # 654ms, 8 threads 526ms, 20 threads 475ms -
                              # so on THIS cpu more is better, and the
                              # common advice about E-cores hurting did not
                              # reproduce. Re-measure on the Orange Pi
                              # before assuming it holds there.
WHISPER_BEAM_SIZE = 5         # REVERTED from 1 after accuracy testing.
                              # Verified against the 190 real recordings in
                              # my_voice_positive/ and my_voice_negative/:
                              # beam_size=1 changed the transcript on a
                              # meaningful fraction of them. It was worth
                              # 10ms (235 vs 245) once the language cache
                              # was in, which is inside measurement noise.
                              # Ten milliseconds is not worth changing what
                              # Citra hears.
WHISPER_USE_VAD_FILTER = False
                              # REVERTED from True. It trims what it judges
                              # to be non-speech BEFORE the encoder sees it,
                              # and on the real corpus seven recordings went
                              # from having a transcript to having none at
                              # all. On those files (background noise) that
                              # is arguably correct - but the corpus contains
                              # no quiet or accented COMMANDS, so there is no
                              # evidence it would not do the same to one, and
                              # a dropped command is a listening regression.
                              # It also measured SLOWER at beam_size=5
                              # (294ms vs 245ms), so it costs accuracy risk
                              # AND speed.

# LANGUAGE DETECTION IS THE ENTIRE 350ms. Whisper checks ~99 languages on
# every single utterance, and this project only ever accepts two (see
# ALLOWED_TRANSCRIPTION_LANGUAGES). Forcing language="en" would fix the
# speed and break the Hindi requirement outright, which is not a trade
# worth making on a product sold with "AC chala do" on the page.
#
# So: detect ONCE (192ms), remember it, and force it on every utterance
# after that - 247ms each instead of 602ms, with Hindi intact. The cache
# expires so that a household switching language mid-evening is picked up
# without a restart; the cost of being wrong is one re-detect, not a
# wrong transcription, because a forced language still transcribes
# code-mixed speech, just with a prior.
LANGUAGE_CACHE_SECONDS = 1800.0   # 30 minutes

WHISPER_MODEL_SIZE = "tiny"   # "tiny" | "base" | "small".
                                #
                                # SWITCHED base -> tiny for latency, after
                                # the goal became "reply in 1 second".
                                # Measured on this machine, same phrase,
                                # identical transcription output: tiny
                                # 511ms vs base 920ms. That ~400ms is a
                                # large fraction of a 1s budget.
                                #
                                # "base" was originally chosen over
                                # "base.en" specifically for Hindi/Hinglish
                                # support, and tiny IS the weaker model
                                # there -- but the user reported base
                                # wasn't actually understanding their Hindi
                                # or Hinglish either, so that accuracy
                                # premium was being paid without being
                                # realized. If Hindi accuracy is fixed
                                # later and tiny proves to be the limit,
                                # this is a one-word revert back to "base".
                                # Do NOT append ".en" to any of these --
                                # that picks the English-only checkpoint
                                # and would remove Hindi support outright.
                                #
                                # Previous note, still true:
                                # "small" is more accurate but slower —
                                # if transcription feels sluggish on your
                                # hardware, drop to "base" first before
                                # reaching for a smaller architecture
                                # change elsewhere in the pipeline.
WHISPER_DEVICE = "cpu"        # change to "cuda" if you have a supported
                                # NVIDIA GPU + the matching CUDA/cuDNN
                                # libraries installed; "cpu" is the safe
                                # default that works everywhere.
WHISPER_COMPUTE_TYPE = "int8"  # int8 quantization keeps CPU inference
                                 # fast with a small accuracy trade-off —
                                 # a good default for real-time voice
                                 # commands on a laptop CPU. Use "float16"
                                 # instead if WHISPER_DEVICE="cuda".

# transcribe() is never given a `language=` argument (see the call site in
# _process_recording) because forcing a single language would break the
# Hindi/Hinglish requirement above — but that means it runs FULL language
# auto-detection (~99 languages) on every single recording, including ones
# that are mostly silence or room noise. Real-world testing caught this
# misfiring: a near-silent recording got auto-detected as Greek at just
# 19% confidence and was passed to the router anyway, which then made an
# unnecessary Smart Path call for what was actually nothing. Rather than
# hardcode a language (breaking Hindi), ALLOWED_TRANSCRIPTION_LANGUAGES +
# MIN_LANGUAGE_PROBABILITY post-filter what auto-detect already returned —
# see the check in _process_recording, right after transcription.
ALLOWED_TRANSCRIPTION_LANGUAGES = frozenset({"en", "hi"})  # English and
                                  # Hindi — covers the Hinglish case too,
                                  # since code-mixed speech typically gets
                                  # classified as one or the other rather
                                  # than a dedicated "hinglish" code (which
                                  # doesn't exist in Whisper's language set).
MIN_LANGUAGE_PROBABILITY = 0.3  # below this, auto-detect itself isn't
                                  # confident in ANY language — a strong
                                  # signal the audio was noise/silence/an
                                  # echo fragment rather than real allowed-
                                  # language speech, regardless of which
                                  # language technically won the vote.
                                  #
                                  # Lowered from 0.5 after a real session
                                  # log showed it rejecting GENUINE
                                  # commands: four separate "How are
                                  # you?" attempts, each correctly
                                  # transcribed and correctly detected as
                                  # English, discarded anyway at
                                  # confidences of 0.35, 0.38, 0.39, and
                                  # 0.43 — every single rejection that
                                  # session was valid English, not noise.
                                  # This makes sense in hindsight: a short
                                  # phrase gives Whisper's language-ID
                                  # head less signal to work with, so
                                  # genuine short commands land at
                                  # naturally lower confidence than a full
                                  # sentence would, even when the
                                  # detected language is exactly right.
                                  # From the user's side this looked
                                  # identical to the wake word not
                                  # working — a valid command silently
                                  # discarded meant repeating it, which is
                                  # indistinguishable from "not
                                  # listening" even though the wake
                                  # detection itself was correct every
                                  # time. 0.3 sits just below every
                                  # genuine value observed (0.35-0.64)
                                  # while still catching confidence near
                                  # zero, which is what the original
                                  # incident (silence detected as Greek
                                  # at 19%) actually looked like. Note
                                  # that incident is ALSO independently
                                  # caught by the language-membership
                                  # check above (Greek was never in
                                  # ALLOWED_TRANSCRIPTION_LANGUAGES to
                                  # begin with) — this constant exists
                                  # for the narrower case of noise
                                  # misdetected AS en/hi specifically, and
                                  # should stay well below any real
                                  # short-command confidence you observe,
                                  # not at a "sounds reasonable" round
                                  # number.

# -----------------------------------------------------------------------------
# SENTENCE-TRANSFORMERS (SEMANTIC INTENT ROUTING)
# -----------------------------------------------------------------------------
SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # per your
                                # instruction — this specific model is
                                # trained on 50+ languages including
                                # Hindi, and handles code-mixed
                                # (Hinglish) input reasonably well since
                                # its training data includes parallel
                                # multilingual paraphrase pairs.
SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # cosine similarity floor for a
                                # protocol to be considered "matched"
                                # rather than falling through to the
                                # Smart Path. Raised from 0.55 after
                                # observing short Whisper hallucinations
                                # ("you", "thank you") scoring high
                                # enough against protocol example phrases
                                # to falsely trigger real hardware. This
                                # is a STARTING POINT at the new value,
                                # not a final tuned constant — see the
                                # tuning note in SemanticRouter.route()
                                # below for how to adjust it further
                                # based on your own false-positive/
                                # false-negative rate.

# -----------------------------------------------------------------------------
# WHISPER HALLUCINATION FILTER (fixed phrase list)
# -----------------------------------------------------------------------------
# faster-whisper, like all Whisper variants, is trained on audio that
# always contains SOME speech — it never saw pure silence during training.
# Fed silence or ambiguous room noise, it doesn't reliably output nothing;
# it tends to hallucinate a small, repeating set of short filler phrases
# instead, apparently patterns absorbed from how its training transcripts
# handled trailing silence in real recordings (video sign-offs, etc).
#
# This list is necessarily NOT exhaustive — new hallucinated phrases can
# surface on different hardware, mic gain levels, or background noise
# profiles. If you notice a new recurring hallucination in your logs
# (_process_recording logs every discarded string), add its normalized form
# here. Entries are lowercase with no trailing punctuation, since
# _process_recording normalizes transcribed text the same way before comparing.
_WHISPER_HALLUCINATIONS = frozenset({
    "you",
    "thank you",
    "bye",
    "thanks for watching",
    "thank you for watching",
    "see you next time",
    "subscribe",
    # --- NEW: Ignore Jarvis's own "Yes sir?" acknowledgment ---
    "yes sir",
    "yes, sir",
    "yes sir?",
    "yes",
    "sir",
    "yeah",
})


# =============================================================================
# TTS: PIPER (NEURAL, LOCAL) — CHUNKED, INTERRUPTIBLE PLAYBACK
# =============================================================================
# Replaces pyttsx3. pyttsx3 wraps the OS's built-in SAPI5 voices on
# Windows — robotic-sounding by construction, since it's a formant/
# concatenative synthesizer, not a neural model. Piper is a real neural
# TTS engine (VITS architecture exported to ONNX) that runs entirely
# offline and, per its own maintainers, synthesizes roughly an order of
# magnitude faster than real-time on a modern desktop CPU — no GPU
# required, which matters for "ultra-low latency" on a laptop that isn't
# guaranteed to have a strong GPU.
#
# WHY PIPER OVER KOKORO/XTTSv2 FOR THIS PROJECT SPECIFICALLY:
# XTTSv2 produces excellent, highly expressive voices but is a much
# larger model — real-time CPU inference is not its design point, and it
# pulls in a heavier dependency stack. Kokoro is fast and good, but
# Piper's Windows story (a pip-installable ONNX Runtime package, no
# separate native binary to manage, no CUDA requirement) is the most
# friction-free path to "just works offline on Windows 11 Pro" that
# still sounds meaningfully more natural than pyttsx3 — which is the
# actual bar you're trying to clear. If you later want XTTSv2's extra
# expressiveness and have GPU headroom to spare, the swap point is
# _synthesize_to_pcm() below; everything downstream (chunked playback,
# barge-in) is engine-agnostic.
#
# TERMINAL SETUP (run once):
#   pip install piper-tts sounddevice numpy
#   python -m piper.download_voices en_US-amy-medium
#
# en_US-amy-medium was chosen as the default voice for a natural, warm
# female voice suited to Citra's persona (swapped from en_US-ryan-medium,
# a male voice, when the assistant was renamed from Jarvis to Citra).
# Piper ships many other voices (browse: https://github.com/rhasspy/piper/
# blob/master/VOICES.md) — swapping is a one-line change to
# PIPER_VOICE_NAME below plus re-running download_voices for the new
# name; no code logic changes required. Other well-regarded female
# en_US voices worth trying if Amy isn't to your taste: "en_US-kathleen-low"
# (lighter/faster, lower quality tier) or "en_US-ljspeech-medium".
#
# !!! ACTION NEEDED: run `python -m piper.download_voices en_US-amy-medium`
# in your project's venv before starting the assistant — the .onnx/.onnx.json
# pair isn't in this repo (same as ryan-medium wasn't, originally) and
# PiperVoice.load() will fail until it's downloaded once.

LAPTOP_SPEAKERS_DEVICE_NAME = "Speakers (Realtek(R) Audio)"  # this
                                    # machine's actual built-in speaker
                                    # endpoint, verified live via
                                    # sounddevice.query_devices(). Citra's
                                    # own voice is pinned to this SPECIFIC
                                    # device (see _laptop_speakers_device_
                                    # index() and its use in
                                    # _play_pcm_interruptible/_play_earcon)
                                    # rather than "whatever the current
                                    # default output device is" -- by
                                    # request, so that switching the
                                    # system's default output (e.g. to a
                                    # Bluetooth speaker for music, see
                                    # jarvis_pc_control.py's
                                    # play_apple_music) never routes
                                    # Citra's own speech somewhere
                                    # unexpected. Update this constant if
                                    # the audio hardware/driver ever
                                    # changes.

MICROPHONE_DEVICE_NAME_MATCH = "microphone array"  # substring (lowercased)
                                    # identifying this machine's REAL
                                    # built-in mic, "Microphone Array
                                    # (Intel(R) Smart Sound Technology...)".
                                    # A substring rather than the full name
                                    # on purpose: PyAudio's MME backend
                                    # truncates device names to 31
                                    # characters ("Microphone Array (Intel®
                                    # Smart"), so an exact-match constant
                                    # would silently never match.
                                    #
                                    # WHY THE MIC IS PINNED BY NAME AT ALL
                                    # -- a real, measured failure, not a
                                    # hypothetical: both pyaudio.open()
                                    # calls below previously passed no
                                    # input_device_index, i.e. "use
                                    # Windows' default recording device".
                                    # On this machine that default is
                                    # "Microphone (Steam Streaming
                                    # Microphone)" -- a VIRTUAL device
                                    # Steam installs, which emits pure
                                    # silence when Steam isn't streaming.
                                    # Measured directly: 2s off that device
                                    # peaked at RMS 0.5, while the real
                                    # Intel array peaked at 73.5 on nothing
                                    # but room noise. Since WAKE_MIN_SPEECH_
                                    # RMS below gates every wake at 300,
                                    # the wake word could NEVER fire -- the
                                    # "hey citra isn't listening" symptom,
                                    # with no error anywhere, because
                                    # opening a silent device succeeds
                                    # perfectly. Pinning by name makes this
                                    # independent of whatever Windows (or
                                    # Steam, or a newly paired Bluetooth
                                    # headset) sets as the default.
_STEAM_VIRTUAL_MIC_MARKER = "steam"  # never select one of Steam's virtual
                                    # capture devices, whatever it's called.

PIPER_VOICE_NAME = "en_US-amy-medium.onnx"  # swap here for a different voice;
                                          # re-run
                                          # `python -m piper.download_voices <name>`
                                          # for whichever name you choose.
PIPER_LENGTH_SCALE = 0.70  # Piper's speaking-rate control: a duration
                                          # multiplier where 1.0 is the voice's
                                          # natural pace, LOWER is faster
                                          # (0.70 = ~30% faster), HIGHER is
                                          # slower. 0.85 (the first speedup)
                                          # still read as too slow in
                                          # real-world use. 0.70 is close to
                                          # the floor before VITS-based
                                          # voices like Piper's start losing
                                          # intelligibility (word boundaries
                                          # blur) — if speech starts sounding
                                          # rushed/unclear rather than just
                                          # brisk, that's the sign to bring
                                          # this back up, not push it lower.
PIPER_SAMPLE_RATE_FALLBACK = 22050  # Piper voices are commonly 22050Hz;
                                      # the ACTUAL rate is read from the
                                      # loaded voice's config at runtime
                                      # (see _get_piper_voice below) — this
                                      # constant is only a documentation
                                      # aid, never actually used for
                                      # playback math, so it can't cause a
                                      # sample-rate mismatch even if a
                                      # future voice uses a different rate.
PIPER_PLAYBACK_CHUNK_FRAMES = 1024  # audio frames written to the output
                                      # stream per loop iteration during
                                      # playback. SMALLER means the
                                      # barge-in stop-check (see
                                      # _play_pcm_interruptible below)
                                      # runs more often, so an interrupt
                                      # lands faster — but too small adds
                                      # per-chunk call overhead and risks
                                      # audio underruns/crackling on some
                                      # Windows audio backends. 1024
                                      # frames at a typical 22050Hz Piper
                                      # voice is ~46ms per chunk — the
                                      # worst-case delay between "barge-in
                                      # signaled" and "audio actually
                                      # stops" is roughly one chunk's
                                      # duration, so ~46ms here, which is
                                      # imperceptibly fast to a person.

# -----------------------------------------------------------------------------
# WAKE ACKNOWLEDGMENT EARCON
# -----------------------------------------------------------------------------
# A short two-tone chime played on wake INSTEAD of speaking "Yes sir?" —
# see play_earcon_async() below for the full reasoning on why that swap is
# the real fix for the self-triggering loop rather than a cosmetic change.
EARCON_FREQUENCIES_HZ = (880.0, 1174.7)  # A5 then D6 — an ascending perfect
                                    # fourth, which reads as "listening,
                                    # go ahead" rather than the descending
                                    # interval most systems use for
                                    # "finished/dismissed". Deliberately
                                    # well above the ~85-255Hz fundamental
                                    # range of human speech so it occupies
                                    # a different part of the spectrum than
                                    # anything the wake-word model was
                                    # trained to respond to.
EARCON_TONE_SECONDS = 0.075         # per tone, so ~150ms total. Short
                                    # enough to be over before you've
                                    # started talking, so it barely
                                    # overlaps the recording at all.
EARCON_FADE_SECONDS = 0.012         # attack/decay ramp — see _play_earcon
                                    # on why a hard edge would be worse
                                    # than useless here.
EARCON_AMPLITUDE = 0.18             # fraction of full scale. Clearly
                                    # audible across a room without being
                                    # startling; raise if you can't hear it
                                    # over ambient noise.

# -----------------------------------------------------------------------------
# "GOT IT" EARCON - played the moment recording STOPS
# -----------------------------------------------------------------------------
# The wake chime above answers "are you listening?". This one answers the
# question that comes next and used to go unanswered: "did you get that?"
#
# Between the user finishing their sentence and Citra saying anything, this
# assistant spends ~450ms transcribing and, on the Smart Path, another 1.3s
# or more waiting on Gemini. That silence is where people repeat themselves
# - and a repeat is heard as a fresh wake, so the cost of saying nothing is
# not just an awkward pause, it is a second recording and a confused turn.
#
# A tone rather than speech, for the same three reasons as the wake chime
# (see play_earcon_async): Whisper cannot transcribe it into a phantom
# command, it cannot re-trigger the wake model, and - the deciding factor
# here - it needs no Piper synthesis pass, so it lands in under 50ms
# instead of several hundred. Speech at this moment would be slower than
# the silence it is trying to fill.
WORKING_EARCON_FREQUENCIES_HZ = (659.3,)  # a single E5. Deliberately NOT a
                                    # two-tone interval: the ascending pair
                                    # on wake means "go ahead", and a
                                    # descending pair would read as
                                    # "dismissed/failed". One flat mid tone
                                    # carries neither meaning - it is a
                                    # receipt, not a verdict. Still well
                                    # clear of speech fundamentals.
WORKING_EARCON_TONE_SECONDS = 0.055  # ~55ms, shorter than the wake chime.
                                    # This one fires on every single
                                    # utterance, so it has to be something
                                    # you stop noticing by day two.
WORKING_EARCON_AMPLITUDE = 0.11     # quieter than the wake chime, for the
                                    # same reason. It should register at
                                    # the edge of attention, not command it.

# -----------------------------------------------------------------------------
# BARGE-IN STOP SIGNAL
# -----------------------------------------------------------------------------
# A single, module-level threading.Event shared between whatever plays
# audio (_play_pcm_interruptible) and whatever detects an interrupt
# condition (the wake-word listener, once Layer 2's concurrent-listening
# restructure is in place — see the note at the bottom of this section).
# Setting this event mid-playback causes the chunk-writing loop to stop
# within one PIPER_PLAYBACK_CHUNK_FRAMES-sized chunk, rather than only
# after the full sentence finishes.
#
# WHY A MODULE-LEVEL EVENT AND NOT AN INSTANCE ATTRIBUTE ON
# JarvisVoiceAssistant: speak_async() is called as a free function
# throughout this file (protocol handlers, _process_recording, etc.), not as a
# method on the assistant instance — keeping the stop-signal at the same
# scope as speak_async() itself avoids threading a self reference through
# every call site just to reach one flag.
_tts_interrupt_event = threading.Event()


def interrupt_speech() -> None:
    """
    Call this to immediately stop whatever Jarvis is currently saying.
    Safe to call even if nothing is playing (the flag is simply a no-op
    in that case) — callers don't need to check "is Jarvis speaking"
    first.
    """
    _tts_interrupt_event.set()


# -----------------------------------------------------------------------------
# "IS PLAYBACK ACTIVE RIGHT NOW" TRACKER (added for Layer 2)
# -----------------------------------------------------------------------------
# Separate from _tts_interrupt_event on purpose: that event means "stop
# playing", this one means "playback is currently in progress". They are
# not the same fact — interrupt_speech() can be called (setting the
# interrupt event) a moment BEFORE playback has actually halted, and the
# SPEAKING state's exit condition (see JarvisVoiceAssistant's
# _consume_frame_speaking) needs to know the latter specifically:
# whether audio is genuinely still coming out of the speaker right now,
# regardless of whether an interrupt was requested. Set True right
# before the output stream starts writing chunks, cleared in a finally
# block so it's reliably reset even if playback raises partway through.
_tts_playing_event = threading.Event()


def _tts_is_currently_playing() -> bool:
    """Used by the Layer 2 state machine's SPEAKING state to detect when
    playback has genuinely ended (naturally or via interrupt) and it's
    time to transition to COOLDOWN."""
    return _tts_playing_event.is_set()


# -----------------------------------------------------------------------------
# PIPER VOICE: LOADED ONCE, REUSED (unlike pyttsx3's per-call engine)
# -----------------------------------------------------------------------------
# WHY THIS DIFFERS FROM THE OLD PYTTSX3 PATTERN: pyttsx3's engine had to
# be constructed fresh per utterance, on the calling thread, due to its
# own documented threading/reuse bugs (see the removed comment block
# above, preserved in version history). Piper's PiperVoice has no such
# constraint — it's a stateless ONNX inference wrapper around a loaded
# model, safe to construct once and call .synthesize_wav() on repeatedly
# from different threads. Loading the ONNX model is the slow part
# (disk read + ONNX Runtime session setup); reloading it per-utterance
# would reintroduce exactly the kind of latency this upgrade is meant to
# remove. A simple lazy-init module-level cache is enough here — this
# project has one voice loaded for the assistant's whole lifetime, not a
# multi-voice pool needing more elaborate lifecycle management.
_piper_voice = None
_piper_voice_lock = threading.Lock()


def _get_piper_voice():
    """Lazily loads and caches the PiperVoice instance. Thread-safe."""
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice

    with _piper_voice_lock:
        # Double-checked locking: another thread may have finished
        # loading while this one was waiting for the lock.
        if _piper_voice is not None:
            return _piper_voice

        import os
        from piper import PiperVoice

        # PiperVoice.load() looks for "<name>.onnx" plus a matching
        # "<name>.onnx.json" config in the same directory. Both files live
        # in this project's own root (fetched once by
        # `python -m piper.download_voices <name>`), not in some
        # Piper-managed cache dir -- so a bare name resolves against the
        # process's CURRENT WORKING DIRECTORY, not necessarily this
        # project. That broke in practice: launching this script from a
        # freshly opened terminal that hadn't cd'd into the project first
        # produced "No such file or directory: 'en_US-amy-medium.onnx.json'"
        # even though the file was sitting right there in the repo.
        # Resolved relative to THIS script's directory instead, same
        # pattern as WAKE_WORD_MODEL_NAME above, so it loads correctly no
        # matter where the process is launched from.
        voice_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PIPER_VOICE_NAME)
        logger.info("Loading Piper voice '%s'...", PIPER_VOICE_NAME)
        _piper_voice = PiperVoice.load(voice_path)
        logger.info("Piper voice loaded.")
        return _piper_voice


def _synthesize_to_pcm(text: str) -> "tuple[np.ndarray, int]":
    """
    Runs Piper synthesis and returns (samples, sample_rate) as a ready-
    to-play int16 numpy array, decoded from Piper's WAV output.

    WHY VIA AN IN-MEMORY WAV BUFFER (io.BytesIO) RATHER THAN A TEMP FILE
    ON DISK: Piper's documented Python API (synthesize_wav) writes into
    a wave.Wave_write object — that's the real, current API shape,
    verified against the official PiperVoice source and docs rather than
    assumed. Wrapping an io.BytesIO in wave.open() gives that same
    Wave_write interface Piper expects, entirely in memory, avoiding a
    disk write+read round trip for every single utterance — meaningful
    for the "ultra-low latency" requirement, since disk I/O is
    unpredictable relative to CPU-bound synthesis time.
    """
    import io
    import wave as wave_module  # aliased: this file's own module-level
                                  # `import wave` (top of file) already
                                  # exists for a different purpose
                                  # elsewhere; aliasing here avoids any
                                  # ambiguity about which `wave` is in
                                  # scope at this point in the file.

    voice = _get_piper_voice()

    from piper.voice import SynthesisConfig
    syn_config = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)

    buffer = io.BytesIO()
    with wave_module.open(buffer, "wb") as wav_writer:
        voice.synthesize_wav(text, wav_writer, syn_config=syn_config)

    buffer.seek(0)
    with wave_module.open(buffer, "rb") as wav_reader:
        sample_rate = wav_reader.getframerate()
        n_frames = wav_reader.getnframes()
        raw_bytes = wav_reader.readframes(n_frames)

    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    return samples, sample_rate


_laptop_speakers_device_index_cache: "int | None | object" = "unresolved"  # tri-
                                    # state cache: "unresolved" (not looked
                                    # up yet), an int (found), or None
                                    # (looked up, genuinely not found --
                                    # cached too, so a missing device
                                    # doesn't re-scan on every single
                                    # utterance).


def _laptop_speakers_device_index():
    """
    Resolves LAPTOP_SPEAKERS_DEVICE_NAME to a sounddevice device index by
    NAME, not a hardcoded index -- device indices can and do shift as
    other audio devices are plugged in or out (verified live: this
    machine's own list changes order depending on what's connected).

    Matches the MME host API specifically, not the DirectSound/WASAPI
    entries sounddevice also lists for the exact same physical device --
    MME is what this machine's system default already used before any of
    this (sd.default.device pointed at the MME entry), so pinning to
    that specific entry keeps behavior identical to "however it already
    sounded," just no longer dependent on staying the Windows default.

    Returns None (logged once) if the named device can't be found, so
    callers can fall back to the default device rather than crashing --
    a laptop's built-in speakers are about the least likely audio
    device to ever actually disappear, but a failed lookup should
    degrade gracefully, not silently break all of Citra's speech.
    """
    global _laptop_speakers_device_index_cache
    if _laptop_speakers_device_index_cache != "unresolved":
        return _laptop_speakers_device_index_cache

    import sounddevice as sd

    found = None
    try:
        hostapis = sd.query_hostapis()
        for index, device in enumerate(sd.query_devices()):
            if device["max_output_channels"] < 1:
                continue
            if device["name"] != LAPTOP_SPEAKERS_DEVICE_NAME:
                continue
            if hostapis[device["hostapi"]]["name"] != "MME":
                continue
            found = index
            break
    except Exception as exc:
        logger.error("Couldn't look up the laptop speakers device: %s", exc)

    if found is None:
        logger.warning(
            "Couldn't find output device %r (MME) -- falling back to the "
            "system default output device for Citra's own speech.",
            LAPTOP_SPEAKERS_DEVICE_NAME,
        )
    _laptop_speakers_device_index_cache = found
    return found


def _play_pcm_interruptible(samples: "np.ndarray", sample_rate: int) -> None:
    import sounddevice as sd

    stream = sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype="int16",
        device=_laptop_speakers_device_index(),
    )
    stream.start()
    # _tts_playing_event.set()  <-- REMOVE OR COMMENT OUT THIS LINE
    try:
        total_frames = len(samples)
        position = 0
        while position < total_frames:
            if _tts_interrupt_event.is_set():
                logger.info("Speech interrupted (barge-in).")
                break
            chunk = samples[position : position + PIPER_PLAYBACK_CHUNK_FRAMES]
            stream.write(chunk.reshape(-1, 1))
            position += PIPER_PLAYBACK_CHUNK_FRAMES
    finally:
        _tts_playing_event.clear()
        stream.stop()
        stream.close()


def _speak_worker(text: str) -> None:
    """Runs entirely on its own thread: synthesize, then play in
    interruptible chunks."""
    try:
        samples, sample_rate = _synthesize_to_pcm(text)
        _play_pcm_interruptible(samples, sample_rate)
    except Exception as exc:
        logger.error("TTS failed to speak: %s", exc)
    finally:
        # CRITICAL FIX: Ensure this is cleared even if synthesis fails,
        # otherwise the state machine gets stuck in SPEAKING forever.
        _tts_playing_event.clear()

def _play_earcon(
    frequencies=EARCON_FREQUENCIES_HZ,
    tone_seconds=EARCON_TONE_SECONDS,
    amplitude=EARCON_AMPLITUDE,
) -> None:
    """
    Synthesizes and plays a short acknowledgment chime.

    Parameterized rather than hardcoded because there are now two of these
    - the ascending wake chime and the single-tone "got it" receipt - and
    they differ only in frequencies, length and level. The envelope, device
    selection and stream handling below are identical for both and are the
    parts that were fiddly to get right, so they are shared rather than
    copied.
    """
    import sounddevice as sd

    sample_rate = 22050
    tone_frames = int(tone_seconds * sample_rate)
    t = np.arange(tone_frames) / sample_rate

    # Short linear attack + decay envelope. Without it, a tone that starts
    # and ends at nonzero amplitude produces an audible click at each edge
    # (a step discontinuity in the waveform) — which is exactly the kind of
    # broadband transient that a wake-word model might latch onto.
    envelope = np.minimum(np.minimum(t, tone_seconds - t) / EARCON_FADE_SECONDS, 1.0)
    envelope = np.clip(envelope, 0.0, 1.0)

    chime = np.concatenate([
        np.sin(2 * np.pi * freq * t) * envelope for freq in frequencies
    ])
    samples = (chime * amplitude * 32767).astype(np.int16)

    stream = sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype="int16",
        device=_laptop_speakers_device_index(),
    )
    stream.start()
    try:
        stream.write(samples.reshape(-1, 1))
    finally:
        stream.stop()
        stream.close()


def play_earcon_async() -> None:
    """
    Plays a short non-speech acknowledgment chime, on its own thread.

    THIS REPLACES SPEAKING "Yes sir?" ON WAKE, and that swap is the actual
    structural fix for the self-triggering loop — not another threshold
    tweak. The old behavior had a genuinely bad failure mode: any spurious
    wake (ambient noise crossing threshold) made Citra SAY a full sentence,
    whose own audio then leaked back into the mic, was transcribed as text,
    and could itself re-trigger the wake word — a feedback loop where one
    false positive became "yes sir, yes sir, yes sir" indefinitely.

    A pure-tone chime breaks that loop at three separate points at once:
      1. It isn't speech, so Whisper can't transcribe it into a phantom
         command the router then tries to answer.
      2. It doesn't resemble "hey citra" acoustically, so it can't
         re-trigger the wake-word model the way spoken words can.
      3. It's ~150ms instead of ~1s of speech, so even the raw acoustic
         echo it puts into the room is far smaller and decays sooner.

    It also fixes the UX complaint directly: a false wake is now a quiet
    blip you can ignore, not Citra loudly talking to herself. And it's
    genuinely FASTER feedback than before — no Piper synthesis pass (a few
    hundred ms) between detecting the wake and acknowledging it.

    Deliberately does NOT set _tts_playing_event: that event means "Citra
    is speaking a response", which drives the SPEAKING state. A wake
    acknowledgment isn't a response — the state machine should be in
    LISTENING here, recording what you say next.
    """
    if citra_mute.is_muted():
        return
    threading.Thread(target=_play_earcon, daemon=True).start()


def play_working_earcon_async() -> None:
    """
    Plays the single-tone "got it, working on it" receipt, on its own
    thread, the instant recording stops.

    See WORKING_EARCON_FREQUENCIES_HZ above for why this is a tone and not
    a spoken acknowledgment. Like play_earcon_async it deliberately does
    NOT set _tts_playing_event: the state machine is still in its
    processing path here, not SPEAKING, and claiming otherwise would make
    _consume_frame_speaking transition to COOLDOWN before the real answer
    had even been synthesized.
    """
    if citra_mute.is_muted():
        return
    threading.Thread(
        target=_play_earcon,
        kwargs={
            "frequencies": WORKING_EARCON_FREQUENCIES_HZ,
            "tone_seconds": WORKING_EARCON_TONE_SECONDS,
            "amplitude": WORKING_EARCON_AMPLITUDE,
        },
        daemon=True,
    ).start()


# -----------------------------------------------------------------------------
# WAKE-TRIGGERED VOLUME DUCKING
# -----------------------------------------------------------------------------
# On "hey citra", every OTHER app's audio session gets turned down for the
# duration of listening + responding, then restored — so Citra's own voice
# (and whatever song was playing) aren't competing with each other. Uses
# pycaw's per-SESSION volume control (Windows Core Audio), not the system
# master volume: Citra's own speech is pinned to a specific OUTPUT DEVICE
# (LAPTOP_SPEAKERS_DEVICE_NAME, see above), not necessarily the same device
# whatever's ducked is playing through (e.g. music routed to a Bluetooth
# speaker, see jarvis_pc_control.py's play_apple_music) — a single master-
# volume knob doesn't reliably map onto "duck everything except Citra" once
# two different devices are involved, but per-session volume does regardless
# of which device each session is on.
_DUCK_EXCLUDE_PROCESS_NAMES = {"pythonw.exe", "python.exe"}  # Citra's own
                                    # process — this file's own TTS/earcon
                                    # playback shows up as a session under
                                    # whichever of these launched it, and
                                    # ducking that too would quiet Citra's
                                    # own voice, defeating the point.
WAKE_DUCK_VOLUME_FRACTION = 0.15  # other audio drops to 15% of whatever it
                                    # was already at (not fully muted, so
                                    # something urgent in the background
                                    # isn't rendered completely inaudible)
                                    # while Citra is listening/responding.

_ducked_session_original_volumes: dict = {}  # pid -> original volume
                                    # (0.0-1.0), populated by
                                    # _duck_other_audio() and consumed by
                                    # _restore_other_audio() so each
                                    # session goes back to EXACTLY what it
                                    # was, not some assumed default.
                                    # Module-level for the same reason
                                    # _tts_interrupt_event is: both
                                    # functions are called as free
                                    # functions from different points in
                                    # JarvisVoiceAssistant's frame-driven
                                    # state machine, not as methods.


def _duck_other_audio() -> None:
    """Turns down every other app's Core Audio session (see this
    section's module comment for why per-session, not master volume).
    Called once, right at wake-word detection — see
    _consume_frame_idle/_consume_frame_idle_continuous."""
    try:
        from pycaw.pycaw import AudioUtilities

        _ducked_session_original_volumes.clear()
        for session in AudioUtilities.GetAllSessions():
            process = session.Process
            if process is None or process.name().lower() in _DUCK_EXCLUDE_PROCESS_NAMES:
                continue
            volume_control = session.SimpleAudioVolume
            if volume_control is None:
                continue
            current = volume_control.GetMasterVolume()
            if current <= 0.0:
                continue  # already silent — nothing to duck or restore
            _ducked_session_original_volumes[process.pid] = current
            volume_control.SetMasterVolume(current * WAKE_DUCK_VOLUME_FRACTION, None)
        if _ducked_session_original_volumes:
            logger.info("Ducked %d other audio session(s).", len(_ducked_session_original_volumes))
    except Exception as exc:
        logger.warning("Couldn't duck background audio: %s", exc)


def _restore_other_audio() -> None:
    """Undoes _duck_other_audio() — restores each session to its exact
    original volume. Called once, at the COOLDOWN -> IDLE transition
    (see _consume_frame_cooldown) rather than at every individual point
    _process_recording can finish, so it fires reliably regardless of
    which path a given wake-word cycle took (a real spoken response, a
    spurious wake with nothing transcribed, a discarded low-confidence
    transcription, etc.) — every one of those paths reaches COOLDOWN,
    which always reaches IDLE."""
    if not _ducked_session_original_volumes:
        return
    try:
        from pycaw.pycaw import AudioUtilities

        for session in AudioUtilities.GetAllSessions():
            process = session.Process
            if process is None or process.pid not in _ducked_session_original_volumes:
                continue
            volume_control = session.SimpleAudioVolume
            if volume_control is None:
                continue
            volume_control.SetMasterVolume(_ducked_session_original_volumes[process.pid], None)
    except Exception as exc:
        logger.warning("Couldn't restore background audio: %s", exc)
    finally:
        _ducked_session_original_volumes.clear()


def speak_async(text: str) -> threading.Thread:
    # MUTED MEANS MUTED, NOT OFF. The caption still goes to the dashboard
    # and the caller still gets a Thread back, so every state machine and
    # every .join() upstream behaves exactly as it does when speaking -
    # only the audio is dropped. Doing this here rather than at each of
    # the ~30 call sites means a future handler cannot forget to check.
    if citra_mute.is_muted():
        citra_ui_bridge.notify_caption(text)
        logger.info("[muted] would have said: %s", text)
        return threading.Thread(target=lambda: None)

    # CRITICAL FIX: Set the playing event IMMEDIATELY.
    # Piper synthesis takes a few hundred milliseconds. If we don't signal
    # the state machine now, it will think speech is already finished,
    # transition to IDLE, and the mic will hear Jarvis when he finally speaks.
    _tts_playing_event.set()
    _tts_interrupt_event.clear()
    citra_ui_bridge.notify_caption(text)

    thread = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
    thread.start()
    return thread


# =============================================================================
# INTENT REGISTRY — EXTENSIBLE PATTERN (per your request)
# =============================================================================
@dataclass
class ProtocolIntent:
    """
    One entry in the intent registry: a protocol name, a bank of example
    phrases (in English/Hindi/Hinglish) that represent what this protocol
    means, and the handler function that executes it.

    WHY THIS MIRRORS HardwareIntent FROM jarvis_router.py:
    Your V1 router already established the pattern of "(matching logic,
    handler function) pairs in a list, extend by appending" — this keeps
    that same shape for semantic matching instead of regex matching, so
    the two files feel like one coherent system rather than two different
    architectural styles bolted together. Adding a new protocol later
    (your "Extra Plug" on Relay 4) means writing one handler function and
    appending one ProtocolIntent to PROTOCOL_REGISTRY below — nothing else
    in this file needs to change.

    `handler` receives the SmartRoomController and the raw transcribed
    text (in case a handler wants to extract something from it, e.g. a
    temperature value spoken in a sentence — none of the two protocols
    below need this, but the signature leaves room for a future one that
    does), and returns the sentence Jarvis should speak.

    `example_phrases` are embedded ONCE at startup (see SemanticRouter),
    not per-request — same "compile once, match many times" principle as
    the regex patterns in jarvis_router.py's INTENTS table.
    """
    name: str
    example_phrases: List[str]
    handler: Callable[[SmartRoomController, str], str]


# -----------------------------------------------------------------------------
# PROTOCOL HANDLERS — STATE-AWARE, SEMANTIC DIRECTION
# -----------------------------------------------------------------------------
# Direction (on vs off) is now determined by WHICH protocol matched, 
# not by brittle keyword parsing. The handler just checks hardware state 
# to generate the correct spoken response and avoid redundant commands.

# -----------------------------------------------------------------------------
# INDIVIDUAL LIGHT TARGETING
# -----------------------------------------------------------------------------
# Extracts WHICH specific relay(s) a lighting command refers to, if any.
# This is deliberately a REGEX pass over the raw transcribed text, run
# BEFORE the handler touches hardware state — not something the semantic
# router does. SemanticRouter's job is classifying an utterance into one
# of the 4 protocol buckets (lighting-on/off, cooling-on/off) by overall
# MEANING via embedding similarity; it was never designed to and doesn't
# extract entities like "which specific light number was mentioned" from
# within an utterance it already classified. Regex is the right tool for
# that different, narrower job — pulling explicit numbers and known
# names out of text — the same way jarvis_router.py's Fast Path already
# uses regex for structured extraction (temperature values, mode names)
# rather than asking an embedding model to do it.
#
# Empty return means "no specific light was named" — callers fall back
# to acting on all 4 relays, which is the exact behavior this file had
# before individual targeting existed. That fallback is preserved
# EXACTLY, so "turn on the lights" / "andhera ho raha hai" / any ambient
# phrase with no light name in it behaves identically to before this was
# added.


def _join_with_and(numbers: list) -> str:
    """
    Formats a list of relay numbers as natural spoken English: "2" for
    one, "2 and 3" for two, "2, 3, and 4" for three or more. Used
    wherever a handler needs to speak multiple light numbers in one
    sentence, so "turning on lights 2, 3, sir" (grammatically off)
    reads as "turning on lights 2 and 3, sir" instead.
    """
    strs = [str(n) for n in numbers]
    if len(strs) == 1:
        return strs[0]
    if len(strs) == 2:
        return f"{strs[0]} and {strs[1]}"
    return f"{', '.join(strs[:-1])}, and {strs[-1]}"


def _extract_target_relays(transcribed_text: str) -> set:
    """
    Returns the set of relay numbers (1-4) explicitly referenced in
    transcribed_text — via digit ("light 2"), number word ("light two"),
    or descriptive name ("the warm light", "the white lights"). Handles
    multiple targets in one sentence ("lights 2 and 3", "the warm light
    and light 3"). Returns an empty set if no specific light is named,
    which callers treat as "act on all 4" — the pre-existing whole-room
    behavior, unchanged.
    """
    text_lower = transcribed_text.lower()
    targets = set()

    for match in re.finditer(r'\b(?:light|relay)s?\s+(\d)\b', text_lower):
        n = int(match.group(1))
        if 1 <= n <= 4:
            targets.add(n)

    for match in re.finditer(r'\band\s+(\d)\b|,\s*(\d)\b', text_lower):
        n = int(match.group(1) or match.group(2))
        if 1 <= n <= 4:
            targets.add(n)

    for word, n in _LIGHT_NUMBER_WORDS.items():
        if re.search(rf'\b(?:light|relay)s?\s+{word}\b', text_lower):
            targets.add(n)
        if re.search(rf'\band\s+{word}\b', text_lower):
            targets.add(n)

    for name, relay_set in LIGHT_NAME_ALIASES.items():
        if re.search(rf'\b{name}\b', text_lower):
            targets |= relay_set

    return targets


def _handle_lighting_on_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    target_relays = _extract_target_relays(transcribed_text)
    relay_numbers = sorted(target_relays) if target_relays else [1, 2, 3, 4]
    is_whole_room = not target_relays

    statuses = {i: controller.get_relay_status(i) for i in relay_numbers}
    if any(not s.success for s in statuses.values()):
        response = "Sorry sir, I couldn't reach the relay board. Please check its connection."
        speak_async(response)
        return response

    already_on = [i for i in relay_numbers if statuses[i].data and statuses[i].data.get("state") == "on"]
    needs_on = [i for i in relay_numbers if i not in already_on]

    if not needs_on:
        response = ("All the lights are already on, sir." if is_whole_room
                    else f"That light is already on, sir." if len(relay_numbers) == 1
                    else "Those lights are already on, sir.")
    else:
        if is_whole_room:
            response = "Right away sir, turning on the lights."
        elif len(needs_on) == 1:
            response = f"Turning on light {needs_on[0]}, sir."
        else:
            response = f"Turning on lights {_join_with_and(needs_on)}, sir."

        def _dispatch():
            for i in needs_on:
                controller.turn_on_relay(i)
        threading.Thread(target=_dispatch, daemon=True).start()

    speak_async(response)
    return response

def _handle_lighting_off_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    target_relays = _extract_target_relays(transcribed_text)
    relay_numbers = sorted(target_relays) if target_relays else [1, 2, 3, 4]
    is_whole_room = not target_relays

    statuses = {i: controller.get_relay_status(i) for i in relay_numbers}
    if any(not s.success for s in statuses.values()):
        response = "Sorry sir, I couldn't reach the relay board. Please check its connection."
        speak_async(response)
        return response

    currently_on = [i for i in relay_numbers if statuses[i].data and statuses[i].data.get("state") == "on"]

    if not currently_on:
        response = ("All the lights are already off, sir." if is_whole_room
                    else "That light is already off, sir." if len(relay_numbers) == 1
                    else "Those lights are already off, sir.")
    else:
        if is_whole_room:
            response = "Turning off all the lights, sir."
        elif len(currently_on) == 1:
            response = f"Turning off light {currently_on[0]}, sir."
        else:
            response = f"Turning off lights {_join_with_and(currently_on)}, sir."

        def _dispatch():
            for i in currently_on:
                controller.turn_off_relay(i)
        threading.Thread(target=_dispatch, daemon=True).start()

    speak_async(response)
    return response

def _handle_cooling_on_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    ac_status = controller.get_ac_status()
    if not ac_status.success:
        response = "Sorry sir, I couldn't check the AC status."
        speak_async(response)
        return response

    ac_on = ac_status.data and ac_status.data.get("power") == "on"
    if ac_on:
        response = "The AC is already running, sir."
    else:
        response = "Sure sir, turning on the AC."
        def _dispatch():
            controller.set_ac_power(True)
        threading.Thread(target=_dispatch, daemon=True).start()

    speak_async(response)
    return response

def _handle_cooling_off_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    ac_status = controller.get_ac_status()
    if not ac_status.success:
        response = "Sorry sir, I couldn't check the AC status."
        speak_async(response)
        return response

    ac_on = ac_status.data and ac_status.data.get("power") == "on"
    if not ac_on:
        response = "The AC is already off, sir."
    else:
        response = "Turning off the AC, sir."
        def _dispatch():
            controller.set_ac_power(False)
        threading.Thread(target=_dispatch, daemon=True).start()

    speak_async(response)
    return response


# -----------------------------------------------------------------------------
# TIME & WEATHER — Siri-style ambient queries, no LLM round-trip needed
# -----------------------------------------------------------------------------
# Both answers are fully deterministic (a clock read or a live API call),
# so — same reasoning as lighting/cooling above — these are Fast Path
# protocols with direct handlers, not questions handed to the Smart
# Path's LLM. That means no hallucination risk on "what time is it" (an
# LLM has no way to actually know that without a tool call anyway) and
# no per-query latency or cost hit for something this simple.

_WEATHER_CODE_DESCRIPTIONS = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with frost",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with light hail",
    99: "thunderstorms with heavy hail",
}  # WMO weather interpretation codes, per Open-Meteo's own published
   # table (open-meteo.com/en/docs) — not invented here, so any code the
   # API actually returns should already have a matching entry above.

# Hardcoded to this specific installation rather than auto-detected via
# IP geolocation. That approach (this project's first version) had two
# real problems, not just theoretical ones: (1) a laptop's IP-based
# location is only as reliable as its ISP's registration data, which can
# resolve to the wrong city entirely, and (2) it was one more network
# call — and therefore one more way for weather to fail — that a fixed,
# already-known installation address doesn't need at all. Update these
# three values directly if this build of Citra is ever installed
# somewhere else; nothing else in the weather handler needs to change.
#
# Coordinates verified against OpenStreetMap's Nominatim geocoder for
# "<your area>, <your city>" (not guessed or recalled from memory),
# and cross-checked by confirming Open-Meteo's own forecast endpoint
# resolves them to the same metro Delhi timezone/region.
WEATHER_DEFAULT_LOCATION = (28.6139, 77.2090, "New Delhi")


def _handle_time_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    now = datetime.datetime.now()
    text_lower = transcribed_text.lower()

    if "date" in text_lower or "day" in text_lower or "today" in text_lower:
        # %-d (no leading zero) isn't portable to Windows' strftime, so
        # the day-of-month is formatted manually rather than relying on
        # a platform-specific format code.
        response = f"It's {now.strftime('%A')}, {now.strftime('%B')} {now.day}, sir."
    else:
        # Same portability issue as above applies to %-I for the hour —
        # strip a leading zero manually instead of using a Linux-only
        # strftime extension.
        hour_12 = now.strftime("%I").lstrip("0") or "12"
        response = f"It's {hour_12}:{now.strftime('%M %p')}, sir."

    speak_async(response)
    return response


WEATHER_FORECAST_DAYS = 4  # today + 3 more. A real "full forecast" request
                                    # (as opposed to "what's the weather
                                    # right now") reasonably means several
                                    # upcoming days, but a genuine 7-day
                                    # readout is a lot to sit through
                                    # spoken aloud one day at a time — 4
                                    # days covers "the next few days" /
                                    # "this week" well enough for a spoken
                                    # answer without turning into a
                                    # monologue. Bump this if you want more
                                    # and don't mind a longer answer.


def _handle_weather_protocol(controller: SmartRoomController, transcribed_text: str) -> str:
    lat, lon, city = WEATHER_DEFAULT_LOCATION
    text_lower = transcribed_text.lower()

    # Checked before asked_about_rain: "will it rain this week" or
    # "weather forecast, is rain coming" should get the multi-day answer,
    # not the single-day one — forecast-scope words win when both are
    # present, since they're the more specific signal of what's actually
    # being asked.
    asked_for_forecast = any(
        w in text_lower for w in (
            "forecast", "this week", "next few days", "coming days",
            "upcoming days", "tomorrow", "next couple",
        )
    )
    asked_about_rain = any(
        w in text_lower for w in ("rain", "umbrella", "wet", "shower", "drizzle", "storm")
    )

    try:
        api_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,apparent_temperature",
                # "current" ONLY reflects this exact instant — the bug the
                # rain-question fix addressed: "will it rain today" got
                # answered from a snapshot that only knows whether it's
                # raining THIS SECOND, so on a merely overcast moment with
                # real rain forecast for later the same day it never
                # actually addressed rain at all. The daily block below is
                # a real FORECAST, not a snapshot, for exactly that reason
                # — and now covers WEATHER_FORECAST_DAYS days, not just
                # today, so "what's the forecast" has real multi-day data
                # to answer from instead of only ever being able to talk
                # about right now.
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": WEATHER_FORECAST_DAYS,
            },
            timeout=8,
        )
        api_response.raise_for_status()
        payload = api_response.json()
        current = payload["current"]
        daily = payload["daily"]
        rain_chance_today = round(daily["precipitation_probability_max"][0])
    except (requests.exceptions.RequestException, KeyError, ValueError, IndexError):
        response = "Sorry sir, I couldn't reach the weather service right now."
        speak_async(response)
        return response

    temp = round(current["temperature_2m"])
    feels_like = round(current["apparent_temperature"])
    description = _WEATHER_CODE_DESCRIPTIONS.get(current["weather_code"], "clear")

    # "Feels like" is only worth saying when it actually diverges from the
    # air temperature — in humid climates it routinely runs several
    # degrees hotter, which is genuinely useful; when it's within a
    # couple of degrees, saying both is just redundant noise.
    feels_like_clause = f" — feels like {feels_like}" if abs(feels_like - temp) >= 3 else ""

    if asked_for_forecast:
        day_names = [datetime.date.fromisoformat(d).strftime("%A") for d in daily["time"]]
        day_parts = []
        for i in range(len(daily["time"])):
            label = "Today" if i == 0 else "Tomorrow" if i == 1 else day_names[i]
            day_desc = _WEATHER_CODE_DESCRIPTIONS.get(daily["weather_code"][i], "clear")
            hi = round(daily["temperature_2m_max"][i])
            lo = round(daily["temperature_2m_min"][i])
            rain = round(daily["precipitation_probability_max"][i])
            day_parts.append(f"{label}, {day_desc}, high {hi}, low {lo}, {rain} percent chance of rain")
        response = f"Here's the forecast for {city}, sir. " + ". ".join(day_parts) + "."

    elif asked_about_rain:
        # Directly answer the yes/no that was actually asked, THEN give
        # temperature as supporting context — not the other way around,
        # which is exactly the ordering that made the old response feel
        # like it never answered the question at all.
        if rain_chance_today >= 60:
            response = (
                f"Yes sir, good chance of rain today — about {rain_chance_today} percent. "
                f"I'd bring an umbrella. It's {temp} degrees and {description} right now."
            )
        elif rain_chance_today >= 30:
            response = (
                f"Maybe, sir — about a {rain_chance_today} percent chance of rain today. "
                f"It's {temp} degrees and {description} right now."
            )
        else:
            response = (
                f"No rain expected today, sir — only about {rain_chance_today} percent. "
                f"It's {temp} degrees and {description} right now."
            )
    else:
        response = f"It's {temp} degrees and {description} in {city} right now, sir{feels_like_clause}."
        # Volunteered even when not asked directly, the way a real weather
        # app leads with rain when it matters — matches how Siri/Google
        # Assistant answer a plain "what's the weather" query.
        if rain_chance_today >= 40:
            response += f" There's a {rain_chance_today} percent chance of rain later today."

    speak_async(response)
    return response


# -----------------------------------------------------------------------------
# THE REGISTRY ITSELF
# -----------------------------------------------------------------------------
PROTOCOL_REGISTRY: List[ProtocolIntent] = [
    ProtocolIntent(
        name="LIGHTING_ON_PROTOCOL",
        example_phrases=[
            # English
            "turn on the lights", "switch on the lights", "turn the lights on",
            "put the lights on", "lights on", "illuminate the room",
            "brighten the room", "let there be light", "i can't see anything",
            "it's too dark in here", "i need some light", "enable lights",
            # Hindi/Hinglish
            "light jalao", "lights on karo", "roshni chahiye", "andhera ho raha hai",
            "andhera hai kamre mein", "bulb jalao", "light chalu karo", "ujala karo",
            "kamre ki light on karo", "light on kar do", "roshni karo",
            # Individual/named light targeting — these teach the router
            # that a light-specific command is still LIGHTING_ON_PROTOCOL;
            # WHICH light is then extracted separately by
            # _extract_target_relays(), not by this classification step.
            "turn on light 2", "turn on light one", "switch on light 3",
            "turn on the warm light", "turn on the white lights",
            "turn on lights 2 and 3", "light 4 on please",
            "can you turn on the bright light", "turn on relay 2",
            # Compound "reason + request" phrasing — a real gap found by
            # testing, not a hypothetical: a sentence embedding for a
            # two-clause utterance sits meaningfully further from EITHER
            # clause's own short example than either clause's similarity
            # to the OTHER clause's example, so "it's dark in here, turn
            # the lights on" can score below SEMANTIC_SIMILARITY_THRESHOLD
            # even though "it's dark in here" alone and "turn the lights
            # on" alone both score ~1.0. A few real compound examples
            # closes that gap directly rather than hoping the model
            # generalizes.
            "it's dark in here can you turn on the lights",
            "i can't see anything turn the lights on please",
        ],
        handler=_handle_lighting_on_protocol,
    ),
    ProtocolIntent(
        name="LIGHTING_OFF_PROTOCOL",
        example_phrases=[
            # English
            "turn off the lights", "switch off the lights", "turn the lights off",
            "kill the lights", "put the lights out", "lights off",
            "shut off the lights", "disable lights", "i want it dark",
            "make it dark", "turn off all the lights",
            # Hindi/Hinglish
            "light band karo", "lights off karo", "bulb band kar do",
            "light band kar do", "kamre ki light off karo", "light bujha do",
            "roshni band karo", "ujala band karo",
            # Individual/named light targeting — same reasoning as the ON
            # protocol above.
            "turn off light 2", "turn off light three", "switch off light 4",
            "turn off the warm light", "turn off the white lights",
            "turn off lights 2 and 3", "light 1 off please",
            "can you turn off the dim lights", "turn off relay 3",
            # Compound phrasing — see LIGHTING_ON_PROTOCOL's comment above
            # for why this is needed, not just belt-and-suspenders.
            "it's too bright in here turn off the lights",
            "i'm heading to bed turn off the lights please",
        ],
        handler=_handle_lighting_off_protocol,
    ),
    ProtocolIntent(
        name="COOLING_ON_PROTOCOL",
        example_phrases=[
            # English
            "turn on the ac", "switch on the ac", "start the ac", "power on the ac",
            "ac on", "turn the air conditioner on", "it's really hot in here",
            "i'm sweating", "cool the room down", "make it cooler", "i need some cooling",
            "turn on the cooling", "activate the ac",
            # Hindi/Hinglish
            "ac chalao", "ac on karo", "ac chalu karo", "ac on kar do",
            "garmi lag rahi hai", "garmi lagri hai yaar", "bahut garam hai",
            "thanda karo", "thand chahiye", "cooling on karo", "ac start karo",
            # Compound phrasing — measured directly: "im sweating turn on
            # the ac" alone scored 0.69 against the single-clause examples
            # above, below the 0.75 match threshold, despite each clause
            # individually scoring near 1.0. Real compound examples fix
            # that rather than relying on the model to generalize across
            # clause boundaries it wasn't shown.
            "it's really hot in here turn on the ac",
            "i'm sweating can you turn on the ac please",
        ],
        handler=_handle_cooling_on_protocol,
    ),
    ProtocolIntent(
        name="COOLING_OFF_PROTOCOL",
        example_phrases=[
            # English
            "turn off the ac", "switch off the ac", "turn the ac off",
            "stop the ac", "power off the ac", "turn the air conditioner off",
            "ac off", "shut down the ac", "deactivate the ac", "turn off the cooling",
            "stop the cooling", "i'm cold turn off the ac",
            # Hindi/Hinglish
            "ac band karo", "ac off karo", "ac band kar do", "cooling band karo",
            "thanda band karo", "ac stop karo", "ac band kar do please",
            # Compound phrasing — see COOLING_ON_PROTOCOL's comment above.
            "i'm cold now turn off the ac",
            "it's freezing in here turn the ac off",
        ],
        handler=_handle_cooling_off_protocol,
    ),
    ProtocolIntent(
        name="TIME_PROTOCOL",
        example_phrases=[
            "what time is it", "what's the time", "tell me the time",
            "do you know the time", "what time is it right now",
            "what's the current time", "what's today's date",
            "what's the date today", "what day is it today",
            "what's the date", "tell me today's date",
            "kitne baje hain", "time kya hua hai", "time batao",
            "aaj kya date hai", "aaj konsa din hai",
        ],
        handler=_handle_time_protocol,
    ),
    ProtocolIntent(
        name="WEATHER_PROTOCOL",
        example_phrases=[
            "what's the weather like", "what's the weather outside",
            "how's the weather today", "is it hot outside",
            "is it cold outside", "what's the temperature outside",
            "what's the temperature right now", "will it rain today",
            "is it going to rain", "how hot is it outside",
            "do i need an umbrella today", "what's it like outside",
            # Longer, more conversational umbrella phrasings — added after
            # a real query ("Do I have to take umbrella with me when I go
            # out?") scored 0.745 against the phrases above, JUST under
            # SEMANTIC_SIMILARITY_THRESHOLD (0.75), and fell through to
            # the Smart Path, which has no real weather data and gave a
            # deflecting non-answer. Measured directly, not guessed: these
            # five score 0.80-0.94 against that same real query (vs 0.745
            # for the shorter phrases above), comfortably clearing the
            # threshold — the gap was phrasing LENGTH/STYLE, not topic.
            "do i have to take an umbrella with me",
            "do i need to carry an umbrella when i go out",
            "will i need an umbrella if i go outside",
            "do i need to bring an umbrella",
            "should i take an umbrella with me today",
            "bahar mausam kaisa hai", "bahar garmi hai kya",
            "bahar thand hai kya", "temperature kitna hai",
            # Forecast-scope phrasing — routes to the same handler, which
            # detects the "forecast" intent itself (see
            # asked_for_forecast in _handle_weather_protocol) and answers
            # with several days instead of just right now.
            "what's the weather forecast", "give me the weather forecast",
            "what's the forecast for this week", "what's the weather like this week",
            "how's the weather looking the next few days",
            "what will the weather be like tomorrow",
            "what's tomorrow's weather", "this week ka mausam kaisa rahega",
        ],
        handler=_handle_weather_protocol,
    ),
]

    # -------------------------------------------------------------------
    # TEMPLATE FOR RELAY 4 ("EXTRA PLUG") — NOT YET ACTIVE
    # -------------------------------------------------------------------
    # This is intentionally commented out, not a real registered
    # protocol. What's plugged into Relay 4 hasn't been decided yet, so
    # there's no real device, phrase set, or handler behavior to encode
    # — inventing one here would mean guessing what a real device should
    # do, which risks producing a protocol that looks legitimate but
    # controls hardware based on a fabricated assumption.
    #
    # To activate this once you know what Relay 4 controls:
    #   1. Write a handler function above (near _handle_lighting_protocol
    #      and _handle_cooling_protocol) with the same signature and the
    #      same "check state (sequential) -> decide sentence ->
    #      speak_async() + dispatch thread (parallel)" shape.
    #   2. Uncomment the block below, rename it, fill in real example
    #      phrases (5-10, covering English/Hindi/Hinglish the way the
    #      two protocols above do), and point `handler` at your new
    #      function.
    #   3. That's it — SemanticRouter discovers and embeds every entry
    #      in PROTOCOL_REGISTRY automatically at startup. Nothing else
    #      in this file needs to change.
    #
    # ProtocolIntent(
    #     name="EXTRA_PLUG_PROTOCOL",
    #     example_phrases=[
    #         "turn on the <device>",
    #         "turn off the <device>",
    #         # ... add real phrases once the device is known
    #     ],
    #     handler=_handle_extra_plug_protocol,  # write this function first
    # ),



# =============================================================================
# SEMANTIC ROUTER
# =============================================================================
class SemanticRouter:
    """
    Embeds every protocol's example phrases once at construction time,
    then for each new transcribed utterance, embeds it and finds the
    highest-similarity protocol. If that similarity clears
    SEMANTIC_SIMILARITY_THRESHOLD, that protocol's handler runs; otherwise
    the caller should fall through to the Smart Path (LM Studio).

    WHY "highest similarity across ALL example phrases of ALL protocols",
    NOT "average similarity per protocol": a single very-close-matching
    example phrase is a stronger signal than a protocol's overall average
    — someone might phrase a lighting request in a way that's extremely
    close to ONE of your ten example phrases while being fairly distant
    from the other nine (different phrasing styles, different languages).
    Taking the max within each protocol, then the max ACROSS protocols,
    respects that a single strong match is meaningful evidence even if
    the rest of that protocol's phrase bank isn't a great fit for this
    particular utterance.
    """

    def __init__(self, registry: List[ProtocolIntent], model_name: str = SEMANTIC_MODEL_NAME):
        # Imported inside __init__, not at module level, so importing
        # jarvis_voice_assistant.py for its dataclasses/constants doesn't
        # require sentence-transformers (and the multi-GB PyTorch it pulls
        # in) to already be installed — only actually constructing a
        # SemanticRouter does.
        from sentence_transformers import SentenceTransformer, util
        self._util = util

        logger.info("Loading semantic model '%s' (first run downloads weights)...", model_name)
        self.model = SentenceTransformer(model_name)

        self.registry = registry

        # Pre-compute embeddings for every protocol's example phrases,
        # ONCE, at startup — not per-utterance. This is the direct
        # semantic-routing analog of jarvis_router.py's "compile regexes
        # once at import time" principle: the expensive part (running the
        # model) happens up front, so runtime matching is just a handful
        # of cosine similarity computations against already-computed
        # vectors.
        self._protocol_embeddings = []  # list of (protocol_index, tensor)
        for idx, intent in enumerate(registry):
            embeddings = self.model.encode(intent.example_phrases, convert_to_tensor=True)
            self._protocol_embeddings.append((idx, embeddings))
        logger.info("Semantic router ready with %d protocol(s).", len(registry))

    def route(self, transcribed_text: str) -> Optional[ProtocolIntent]:
        """
        Returns the best-matching ProtocolIntent if its similarity clears
        the threshold, else None (signal to fall through to the Smart
        Path — same None-means-"no match" convention as
        jarvis_router.py's _try_fast_path).

        TUNING NOTE: SEMANTIC_SIMILARITY_THRESHOLD = 0.75 is a reasonable
        starting point for this multilingual MiniLM model, but the RIGHT
        value depends on your actual usage — run this with logging at
        INFO level for a few days, note the similarity scores for
        utterances that SHOULD have matched a protocol but didn't (raise
        the threshold... no, LOWER it to catch them) versus utterances
        that matched a protocol but shouldn't have (RAISE the threshold
        to be more conservative). This is a one-line constant to adjust,
        not something requiring code changes.
        """
        if not transcribed_text or not transcribed_text.strip():
            return None

        query_embedding = self.model.encode(transcribed_text, convert_to_tensor=True)

        best_score = -1.0
        best_protocol_idx: Optional[int] = None

        for protocol_idx, phrase_embeddings in self._protocol_embeddings:
            # cos_sim(query, all_phrases_for_this_protocol) -> a 1xN
            # tensor of similarity scores; .max() gives this protocol's
            # single best-matching example phrase's score.
            scores = self._util.cos_sim(query_embedding, phrase_embeddings)
            protocol_best = scores.max().item()

            if protocol_best > best_score:
                best_score = protocol_best
                best_protocol_idx = protocol_idx

        best_matched = best_protocol_idx is not None and best_score >= SEMANTIC_SIMILARITY_THRESHOLD
        logger.info(
            "Semantic match: '%s' -> %s (score=%.3f, threshold=%.2f, matched=%s)",
            transcribed_text,
            self.registry[best_protocol_idx].name if best_protocol_idx is not None else "NONE",
            best_score,
            SEMANTIC_SIMILARITY_THRESHOLD,
            best_matched,
        )

        if best_matched:
            return self.registry[best_protocol_idx]
        return None


# =============================================================================
# WAKE WORD + AUDIO CAPTURE
# =============================================================================
def _create_openwakeword_model():
    """
    Constructs the openWakeWord model, downloading the pre-trained ONNX
    weights on first run if they aren't already cached locally.

    UNLIKE PORCUPINE: openWakeWord ships its models as separate downloads
    fetched via download_models() rather than bundling them in the pip
    package itself — this is a real, required setup step (confirmed
    against the library's own README), not optional. It's a one-time
    network fetch; every run after the first uses the local cache and
    needs no internet connection, keeping this consistent with the
    fully-offline design of the rest of the project. This download is
    STILL needed even though WAKE_WORD_MODEL_NAME now points at our own
    custom-trained hey_citra.onnx, not a built-in name: the shared
    melspectrogram/embedding feature-extraction models it fetches are
    infrastructure every wake-word classifier sits on top of, custom or
    official.

    CUSTOM MODEL PATH RESOLUTION: when WAKE_WORD_MODEL_NAME is a bare
    official name (e.g. "hey_jarvis"), openWakeWord resolves it itself
    against its own download cache. When it's a file path (our
    "hey_citra.onnx"), it's resolved relative to THIS script's directory
    rather than the process's current working directory, so the wake
    word still loads correctly no matter where you launch this script
    from.
    """
    import os
    import openwakeword
    from openwakeword.model import Model

    logger.info("Ensuring openWakeWord pre-trained models are downloaded...")
    openwakeword.utils.download_models()

    wakeword_model_path = WAKE_WORD_MODEL_NAME
    if wakeword_model_path.endswith((".onnx", ".tflite")):
        wakeword_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), wakeword_model_path)

    return Model(
        wakeword_models=[wakeword_model_path],
        inference_framework=WAKE_WORD_INFERENCE_FRAMEWORK,
    )


_microphone_device_index_cache: "int | None | object" = "unresolved"  # same
                                    # tri-state cache shape as
                                    # _laptop_speakers_device_index_cache
                                    # above, for the same reason.


def _microphone_device_index(pyaudio_instance):
    """
    Resolves MICROPHONE_DEVICE_NAME_MATCH to a PyAudio input-device index
    -- see that constant's comment for the measured failure this exists to
    prevent (Windows' default recording device being Steam's silent
    virtual mic, which made the wake word impossible to trigger).

    Resolved by NAME rather than a hardcoded index because PyAudio's
    indices shift as audio devices come and go, and returns None (logged)
    if nothing matches, so the caller falls back to PyAudio's default
    device -- degrading to the old behavior rather than failing to open a
    microphone at all.

    NOTE these are PyAudio indices, which are NOT interchangeable with the
    sounddevice indices _laptop_speakers_device_index() returns: the two
    libraries enumerate devices separately. That's why this is its own
    function rather than a shared lookup.
    """
    global _microphone_device_index_cache
    if _microphone_device_index_cache != "unresolved":
        return _microphone_device_index_cache

    found = None
    try:
        for index in range(pyaudio_instance.get_device_count()):
            info = pyaudio_instance.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) < 1:
                continue
            name = str(info.get("name", "")).lower()
            if _STEAM_VIRTUAL_MIC_MARKER in name:
                continue
            if MICROPHONE_DEVICE_NAME_MATCH in name:
                found = index
                logger.info("Using microphone device %d: %s", index, info.get("name"))
                break
    except Exception as exc:
        logger.error("Couldn't look up the microphone device: %s", exc)

    if found is None:
        logger.warning(
            "No input device matching %r -- falling back to the system "
            "default recording device, which may be silent.",
            MICROPHONE_DEVICE_NAME_MATCH,
        )
    _microphone_device_index_cache = found
    return found


def _record_until_silence(pyaudio_instance, sample_rate: int) -> bytes:
    """
    Records audio from the default microphone starting immediately (the
    wake word was just detected, so the user is presumably about to
    speak), and stops once SILENCE_DURATION_SECONDS of below-threshold
    audio has been seen, or MAX_RECORDING_SECONDS is hit as a hard
    ceiling. Returns raw 16-bit PCM bytes.
    """
    import pyaudio

    CHUNK = 1024
    stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=_microphone_device_index(pyaudio_instance),
        frames_per_buffer=CHUNK,
    )

    frames: List[bytes] = []
    silence_chunks_needed = int(SILENCE_DURATION_SECONDS * sample_rate / CHUNK)
    silent_chunk_count = 0
    max_chunks = int(MAX_RECORDING_SECONDS * sample_rate / CHUNK)

    try:
        for _ in range(max_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Simple RMS-based silence check.
            samples = np.frombuffer(data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

            if rms < SILENCE_THRESHOLD:
                silent_chunk_count += 1
                if silent_chunk_count >= silence_chunks_needed:
                    break
            else:
                silent_chunk_count = 0
    finally:
        stream.stop_stream()
        stream.close()

    return b"".join(frames)


def _pcm_bytes_to_numpy(pcm_bytes: bytes) -> np.ndarray:
    """
    faster-whisper's transcribe() accepts a numpy float32 array directly
    (in addition to file paths) — converting in-memory avoids writing a
    temporary .wav file to disk for every single utterance, which keeps
    the wake-word-to-transcription latency lower than a disk round-trip
    would.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    # faster-whisper expects float32 samples normalized to [-1.0, 1.0],
    # matching how it internally treats decoded audio.
    return samples.astype(np.float32) / 32768.0


# =============================================================================
# AUDIO INGEST — accepts recordings submitted from the web UI's mic button
# =============================================================================
# Lets a phone (or any device on the LAN) talk to Citra using its OWN
# microphone instead of this laptop's — citra_ui/app.js records a clip via
# getUserMedia + MediaRecorder, citra_ui_server.py forwards the raw bytes
# here (see that file's voice_handler/VOICE_INGEST_URL), and this decodes
# + routes it through the EXACT SAME pipeline the physical mic uses:
# _process_recording() — same transcription, same language/hallucination
# filtering, same semantic routing, same Smart Path fallback, same spoken
# response through THIS machine's speakers. No parallel/duplicate logic —
# a phone-submitted command is indistinguishable from a locally-spoken one
# past this one entry point.
AUDIO_INGEST_PORT = 8766  # LOCALHOST ONLY, deliberately never exposed to
                                    # the LAN directly — citra_ui_server.py
                                    # (which IS LAN-exposed) is the sole
                                    # intended caller, already having done
                                    # its own job of being reachable from
                                    # other devices. This process doesn't
                                    # need a second network-facing surface
                                    # when one already exists for exactly
                                    # this purpose.


def _decode_audio_to_pcm16_16k_mono(audio_bytes: bytes) -> bytes:
    """
    Decodes an arbitrary-format audio blob — webm/opus from Chrome/
    Android's MediaRecorder, mp4/aac from iOS Safari's, or anything else a
    browser might produce — into 16-bit PCM at 16kHz mono: the exact
    format _process_recording() and the rest of this pipeline already
    expect from the physical microphone, so a phone-submitted recording
    needs no special-casing beyond this one conversion.

    Uses PyAV (already an installed dependency), which bundles its own
    FFmpeg libraries — no system ffmpeg binary needs to be installed or on
    PATH for this to work. Verified against real encode/decode round-trips
    of BOTH webm/opus and mp4/aac before this was wired in, not assumed to
    work from documentation alone — this is exactly the kind of
    "should work in theory" conversion worth actually confirming.
    """
    import av

    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    try:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
    finally:
        container.close()

    if not chunks:
        return b""
    pcm = np.concatenate(chunks, axis=-1).reshape(-1).astype(np.int16)
    return pcm.tobytes()


class _AudioIngestHandler(http.server.BaseHTTPRequestHandler):
    """
    Handles POST /ingest from citra_ui_server.py. `assistant` is a class
    attribute set once (to the single running JarvisVoiceAssistant
    instance) before the server starts, since http.server.HTTPServer
    constructs a fresh handler instance per request with no built-in way
    to pass extra constructor arguments through.
    """
    assistant: "Optional[JarvisVoiceAssistant]" = None

    def do_POST(self) -> None:
        if self.path != "/ingest":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        audio_bytes = self.rfile.read(content_length) if content_length else b""

        if not audio_bytes:
            self._respond(400, {"success": False, "message": "No audio received."})
            return

        # A quick check-then-set on self.assistant._state, not guarded by
        # a lock — matches the level of synchronization the REST of this
        # state machine already uses (a background thread setting
        # self._state = COOLDOWN at the end of _process_recording has
        # never been lock-guarded either). This means a genuinely
        # simultaneous local wake-word detection and phone submission
        # (both landing in the same handful of milliseconds) could in
        # theory both see IDLE and both proceed. Accepted as a known,
        # very-low-probability edge case rather than adding a lock that
        # would only protect THIS entry point and not the wake-word one
        # too — a partial fix is not a real fix here, and wrapping the
        # existing, heavily-tested wake-word path in a new lock is a
        # bigger, riskier change than this rare race justifies.
        if self.assistant._state != AssistantState.IDLE:
            self._respond(409, {
                "success": False,
                "message": "Citra is busy right now — try again in a moment.",
            })
            return

        try:
            pcm_bytes = _decode_audio_to_pcm16_16k_mono(audio_bytes)
        except Exception as exc:
            logger.error("Failed to decode phone-submitted audio: %s", exc)
            self._respond(400, {"success": False, "message": f"Couldn't decode that audio: {exc}"})
            return

        if not pcm_bytes:
            self._respond(400, {"success": False, "message": "That recording had no audio in it."})
            return

        # Mirrors _consume_frame_listening's own IDLE/LISTENING ->
        # PROCESSING transition exactly — same state, same UI
        # notification, same "hand off to a background thread so this
        # doesn't block" reasoning (transcription + routing is genuinely
        # slow, and here it would otherwise block the HTTP response).
        self.assistant._state = AssistantState.PROCESSING
        citra_ui_bridge.notify_state("PROCESSING")
        processing_thread = threading.Thread(
            target=self.assistant._process_recording,
            args=(pcm_bytes,),
            daemon=True,
        )
        processing_thread.start()

        self._respond(202, {"success": True, "message": "Got it, processing now."})

    def _respond(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args) -> None:
        # BaseHTTPRequestHandler logs every request to stderr by default,
        # in its own format — this project already has its own logger
        # used consistently everywhere else, so that would just be
        # redundant, differently-formatted noise rather than useful signal.
        pass


# =============================================================================
# THE VOICE ASSISTANT — TOP-LEVEL ORCHESTRATION
# =============================================================================
class JarvisVoiceAssistant:
    """
    Owns all four pipeline stages and runs the main listen loop. One
    instance per process — construct it once, call .run().
    """

    def __init__(self):
        # Stage 1 components (constructed lazily in run(), since they
        # require real audio hardware / downloaded models — see run()'s
        # docstring for why construction is deferred rather than done
        # here in __init__).
        self._openwakeword_model = None
        self._pyaudio_instance = None
        self._whisper_model = None
        self._input_stream = None  # the SINGLE continuous PyAudio input
                                     # stream — see run()'s docstring for
                                     # why there is exactly one, never two.

        # Stage 2: semantic router over the protocol registry.
        self.semantic_router: Optional[SemanticRouter] = None  # built in run()

        # Gate for run()'s background model loading (see _load_heavy_models).
        # Created here, not only in run(), so a JarvisVoiceAssistant built
        # directly for a unit test — which this file's design deliberately
        # supports, see run()'s docstring on why heavy init lives there —
        # can still call _process_recording with an injected mock model
        # without tripping over an attribute that only exists post-run().
        # Pre-set for exactly that case: nothing to wait for when the test
        # supplied the model itself.
        self._heavy_models_ready = threading.Event()
        self._heavy_models_ready.set()

        # Neural end-of-speech detection — see _SpeechDetector. Constructed
        # here (cheap, no model loaded yet); the actual model load happens
        # on the background thread in _load_heavy_models.
        self._speech_detector = _SpeechDetector()

        # Stage 3: hardware controller, shared across all handlers.
        # AC-ONLY TESTING NOTE: SmartRoomController() with no arguments
        # picks up jarvis_hardware_api.py's own defaults — currently
        # ac_host="192.168.0.7" (confirmed working) and
        # relay_host="192.168.0.X" (placeholder, relay board not yet
        # flashed/IP-confirmed). LIGHTING_PROTOCOL (relay-based) will
        # fail against the placeholder until that's set; COOLING_PROTOCOL
        # (AC + relay 3 fan) will succeed on the AC portion and fail the
        # same way on the fan relay call. Update relay_host in
        # jarvis_hardware_api.py's SmartRoomController.__init__ default
        # once the relay board has a real IP — nothing in this file needs
        # to change.
        self.controller = SmartRoomController()

        # Reminders need to actually SPEAK when they fire, on whatever
        # thread threading.Timer schedules them on (never the main
        # consumer thread) — passing this process's real speak_async in
        # explicitly is what turns ReminderController's log-only default
        # (see its own docstring) into reminders you actually hear.
        # notify_fn is optional and best-effort: a live caption on the
        # dashboard when a reminder fires is a nice-to-have, not
        # something a missing/offline UI server should ever block on —
        # citra_ui_bridge.notify_caption already swallows its own
        # connection errors for exactly that reason.
        self.reminder_controller = ReminderController(
            speak_fn=speak_async,
            notify_fn=citra_ui_bridge.notify_caption,
            action_fn=self._run_scheduled_command,
        )

        # Stage 4: reuse jarvis_router.py's LM Studio integration directly
        # rather than reimplementing HTTP-to-LLM logic a second time —
        # single source of truth for "how do we talk to LM Studio" stays
        # in jarvis_router.py.
        self.text_router = JarvisRouter(controller=self.controller, reminder_controller=self.reminder_controller)

        # Speaks a truthful progress line if - and only if - the Smart
        # Path proves slow. Constructed once and reused: it remembers the
        # last line it used so it can avoid repeating itself on the next
        # request (see ProgressNarrator._pick), which only means anything
        # if the instance persists across calls.
        self._narrator = ProgressNarrator(speak_async)

        # Whose language, and when we worked it out. See
        # LANGUAGE_CACHE_SECONDS.
        self._cached_language: Optional[str] = None
        self._cached_language_at: float = 0.0

        # -------------------------------------------------------------
        # LAYER 2: STATE MACHINE + CONCURRENCY PRIMITIVES
        # -------------------------------------------------------------
        # self._state is read by the reader thread (to decide what to do
        # with each incoming frame) and written ONLY by the main thread
        # (inside the consumer loop). This single-writer discipline —
        # confirmed safe in isolated testing before being wired in here
        # — is what allows sharing state between the two threads without
        # a lock around every single access. Python's GIL makes a single
        # attribute assignment atomic enough for this pattern; the
        # reader thread reading a slightly-stale state value for one
        # frame (a few milliseconds) has no meaningful consequence here,
        # unlike e.g. a financial transaction would.
        self._state = AssistantState.IDLE

        # The queue every incoming audio frame flows through, regardless
        # of current state. Decoupling "reading from the mic" from
        # "deciding what a frame means" via a queue is what lets exactly
        # ONE reader thread own the PyAudio stream for the assistant's
        # entire lifetime, while the CONSUMER side (running on the main
        # thread) freely changes its own behavior between states without
        # ever touching the stream or reader thread itself.
        self._frame_queue: "queue.Queue[bytes]" = queue.Queue()

        # Signals the reader thread to stop; checked in its loop and set
        # by run()'s shutdown path.
        self._reader_stop_event = threading.Event()

        # Accumulates frames during LISTENING (recording a command) —
        # populated by the consumer loop, not the reader thread, keeping
        # with the single-writer-per-piece-of-state discipline.
        self._recording_frames: List[bytes] = []
        self._recording_silent_frame_count = 0
        self._recording_speech_started = False

        # Accumulates consecutive loud frames during SPEAKING, for the
        # BARGE_IN_CONSECUTIVE_FRAMES_REQUIRED debounce.
        self._barge_in_loud_frame_count = 0

        # Accumulates consecutive above-threshold wake-word frames during
        # IDLE, for the WAKE_WORD_CONSECUTIVE_FRAMES_REQUIRED debounce.
        self._wake_word_consecutive_frame_count = 0

        # Rolling RMS of the most recent IDLE frames, for the
        # WAKE_MIN_SPEECH_RMS energy gate. A bounded deque keeps this at a
        # fixed size for free — no manual trimming, and no unbounded growth
        # over a process that stays idle for hours.
        self._recent_idle_rms: "deque[float]" = deque(maxlen=WAKE_ENERGY_WINDOW_FRAMES)

        # Timestamp (time.monotonic()) marking when COOLDOWN began, so
        # _consume_frame_cooldown knows when COOLDOWN_SECONDS has elapsed.
        self._cooldown_start_time = 0.0

        # CONTINUOUS LISTENING MODE — see that constants section's
        # docstring above for what this changes. Off by default; toggled
        # by voice command, handled in _route_and_respond.
        self._continuous_mode = False
        # Set True the moment a LISTENING recording started from
        # continuous mode's VAD trigger (not a wake-word trigger) —
        # _process_recording reads this ONCE, right after transcribing,
        # to decide whether the _extract_command_without_name() gate applies, and
        # clears it back to False either way. A wake-word-triggered
        # recording never sets this, so normal-mode behavior is
        # completely unaffected regardless of what continuous mode does.
        self._continuous_segment_pending = False
        # Debounce counter for CONTINUOUS_MODE_VAD_CONSECUTIVE_FRAMES_REQUIRED
        # — same role as self._wake_word_consecutive_frame_count above, kept
        # as its own counter rather than reused since the two modes are
        # mutually exclusive at any given moment but this keeps that
        # exclusivity a runtime fact, not something this code has to
        # assume.
        self._continuous_vad_frame_count = 0
        # Rolling pre-roll buffer — see CONTINUOUS_MODE_PREROLL_FRAMES's
        # comment for why. Only populated/consumed while continuous mode
        # is active.
        self._continuous_preroll: "deque[bytes]" = deque(maxlen=CONTINUOUS_MODE_PREROLL_FRAMES)

    def _run_scheduled_command(self, command: str) -> None:
        """Runs a command whose scheduled time has arrived (see
        ReminderController.schedule_action), routing the text exactly as
        if it had just been spoken.

        Routed through the SEMANTIC router first, then the text router —
        the same two-stage path _route_and_respond uses — rather than
        jumping straight to the text router, so a scheduled "turn off the
        ac" behaves identically to a spoken one instead of taking a
        subtly different path to the same hardware.

        Deliberately does NOT speak, and does NOT touch self._state: this
        fires on a timer thread at an arbitrary moment, possibly while
        Citra is mid-conversation or while everyone is asleep. Announcing
        it would be both a race against the state machine and, at 2am,
        exactly the wrong behavior.
        """
        matched = self.semantic_router.route(command) if self.semantic_router else None
        if matched is not None:
            logger.info("Scheduled command matched %s", matched.name)
            matched.handler(self.controller, command)
            return
        result = self.text_router.route(command)
        logger.info(
            "Scheduled command ran via %s path: success=%s, %s",
            result.path, result.success, result.message,
        )

    def _load_heavy_models(self) -> None:
        """Loads Whisper + the semantic router on a background thread, then
        signals _heavy_models_ready. See the call site in run() for why
        deferring these specifically is safe (neither is reachable until
        after a wake word fires and the user stops talking)."""
        load_start = time.perf_counter()

        logger.info("Loading Whisper model '%s' in the background...", WHISPER_MODEL_SIZE)
        from faster_whisper import WhisperModel
        _whisper_kwargs = {"cpu_threads": WHISPER_CPU_THREADS} if WHISPER_CPU_THREADS else {}
        self._whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            **_whisper_kwargs,
        )

        logger.info("Loading semantic router in the background...")
        self.semantic_router = SemanticRouter(PROTOCOL_REGISTRY)

        # Loaded here rather than in run() so its 1.3s stays off the
        # startup path with everything else deferred. Safe: it's only
        # consulted during LISTENING, which a wake word has to fire first.
        self._speech_detector.load()

        self._heavy_models_ready.set()
        logger.info(
            "Background model loading finished in %.1fs — full pipeline ready.",
            time.perf_counter() - load_start,
        )

    def run(self) -> None:
        """
        Main loop: load models (heavy, done once), open exactly ONE
        continuous input stream, start the reader thread, then run the
        state-machine consumer loop forever.

        !!! WHY EXACTLY ONE PYAUDIO STREAM, NOT TWO !!!
        The original Layer 1 handoff flagged "a second concurrently-open
        PyAudio stream" as the likely approach for true barge-in, but
        specifically declined to build it without verifying Windows
        WASAPI behavior for concurrent input streams on the same device
        first. That verification did not turn up a clear "this works
        reliably" answer — WASAPI shared mode is documented to be picky
        about exact configuration matching between concurrent sessions,
        which is a real risk, not a solved problem. Rather than build on
        that uncertain foundation, this implementation opens exactly ONE
        input stream for the assistant's entire lifetime, and a single
        reader thread continuously pushes 80ms frames from it into
        self._frame_queue. Every consumer — wake-word detection, command
        recording, barge-in energy checking — reads from that SAME queue,
        with behavior selected by self._state. This sidesteps the
        concurrent-stream question entirely, because there is only ever
        one stream open on the microphone, matching the same
        "single-source-of-truth" principle used elsewhere in this
        project (e.g. jarvis_router.py owning the one LM Studio
        integration other files reuse rather than duplicate).

        WHY MODEL LOADING HAPPENS HERE, NOT IN __init__:
        Constructing a JarvisVoiceAssistant() for, say, a unit test of
        PROTOCOL_REGISTRY's handler functions shouldn't require a
        microphone to be present or multi-GB models to download. Keeping
        heavy, hardware/model-dependent initialization inside run()
        (called only when you actually want to start listening) keeps the
        rest of this class's components testable independently.
        """
        import pyaudio

        logger.info("Initializing wake word engine...")
        self._openwakeword_model = _create_openwakeword_model()

        logger.info("Initializing microphone...")
        self._pyaudio_instance = pyaudio.PyAudio()

        # -------------------------------------------------------------
        # DEFERRED HEAVY MODEL LOADING — what makes startup feel instant
        # -------------------------------------------------------------
        # Whisper and the sentence-transformers router are the two slowest
        # things to load (measured 3.4s and 3.7s respectively even with
        # HF_HUB_OFFLINE, and far worse cold), but NEITHER is needed to
        # start listening: wake-word detection only needs
        # self._openwakeword_model, already loaded above. Whisper isn't
        # touched until _process_recording, and the router not until
        # _route_and_respond — both of which happen only AFTER a wake word
        # fires AND the user finishes speaking, which is seconds of real
        # time later at minimum.
        #
        # Loading them on a background thread therefore costs nothing in
        # practice while cutting time-to-listening from "all models
        # loaded" to just the wake-word engine. _process_recording waits
        # on _heavy_models_ready before touching either, so the ordering
        # is enforced rather than assumed — if someone somehow triggers a
        # command before loading finishes, they wait exactly as long as
        # they would have anyway, instead of hitting an AttributeError on
        # a not-yet-assigned model.
        self._heavy_models_ready = threading.Event()
        threading.Thread(target=self._load_heavy_models, daemon=True).start()

        # -------------------------------------------------------------
        # Open the ONE continuous input stream, sized for openWakeWord's
        # native 1280-sample (80ms @ 16kHz) frame unit. Every downstream
        # consumer (recording, barge-in) reuses these same 80ms frames,
        # which is part of what makes one shared stream workable across
        # all the states that need microphone input.
        # -------------------------------------------------------------
        OWW_SAMPLE_RATE = 16000
        OWW_FRAME_SIZE = 1280
        self._input_stream = self._pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=OWW_SAMPLE_RATE,
            input=True,
            input_device_index=_microphone_device_index(self._pyaudio_instance),
            frames_per_buffer=OWW_FRAME_SIZE,
        )

        reader_thread = threading.Thread(
            target=self._reader_thread_main,
            args=(OWW_FRAME_SIZE,),
            daemon=True,
        )
        reader_thread.start()

        # Audio ingest server — see _AudioIngestHandler's docstring and
        # AUDIO_INGEST_PORT's comment. Runs on its own daemon thread for
        # the same reason the reader thread does: this can't block, or
        # share a thread with, the main consumer loop.
        _AudioIngestHandler.assistant = self
        ingest_server = http.server.HTTPServer(("127.0.0.1", AUDIO_INGEST_PORT), _AudioIngestHandler)
        ingest_thread = threading.Thread(target=ingest_server.serve_forever, daemon=True)
        ingest_thread.start()
        logger.info("Audio ingest listening on http://127.0.0.1:%d (local only)", AUDIO_INGEST_PORT)

        logger.info("Citra is ready. Listening for the wake word...")

        # Previously this spoke "Citra online." and .join()ed on it before
        # entering the consumer loop. Both halves of that were wrong for a
        # system that now auto-starts at login:
        #   1. The announcement itself is unwanted noise on every boot —
        #      the point of autostart is that she's simply THERE, not that
        #      she announces herself to an empty room.
        #   2. More importantly, .join() blocked the main loop from
        #      starting until TTS finished, and the FIRST speak_async call
        #      is what lazily loads the Piper voice — measured at 1.5s to
        #      11s in real startup logs. So the announcement was also
        #      delaying the moment she could actually hear anything.
        # Warming the voice on a background thread instead keeps the
        # benefit that made preloading worth doing (the first real answer
        # doesn't pay the load cost) while the consumer loop starts
        # listening immediately, in parallel.
        threading.Thread(target=_get_piper_voice, daemon=True).start()

        try:
            self._consumer_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            self._reader_stop_event.set()
            reader_thread.join(timeout=2.0)
            self._cleanup()

    # =====================================================================
    # READER THREAD — the ONLY thread that ever calls self._input_stream.read()
    # =====================================================================
    def _reader_thread_main(self, frame_size: int) -> None:
        """
        Runs for the assistant's entire lifetime on its own thread. Its
        ONLY job is: read one frame, push it to self._frame_queue, repeat.
        It does NOT decide what a frame means — that decision (wake-word
        check vs. recording vs. barge-in check) lives entirely in
        _consumer_loop, which reads self._state to decide. This
        separation is what was verified in isolated testing before being
        wired in here: the reader thread doesn't need to know about
        states at all, which keeps it simple and keeps exactly one
        thread ever touching the underlying PyAudio stream object —
        PyAudio Stream objects are not documented as safe for concurrent
        reads from multiple threads, so this single-reader discipline is
        a real safety property, not just a style choice.
        """
        while not self._reader_stop_event.is_set():
            try:
                raw_frame = self._input_stream.read(frame_size, exception_on_overflow=False)
            except Exception as exc:
                # A transient read error shouldn't kill the reader thread
                # permanently — log and keep going, since a single
                # dropped frame is recoverable and the alternative
                # (thread death) silently ends all audio input for the
                # rest of the process.
                logger.error("Audio read error: %s", exc)
                continue
            self._frame_queue.put(raw_frame)

    # =====================================================================
    # CONSUMER LOOP — the state machine itself, running on the main thread
    # =====================================================================
    def _consumer_loop(self) -> None:
        """
        Pulls frames from self._frame_queue and, based on self._state,
        routes each one to the right handler:
          IDLE       -> check for wake word
          LISTENING  -> buffer for recording, check for end-of-speech silence
          SPEAKING   -> check for barge-in energy
          PROCESSING / COOLDOWN -> frame is discarded (nothing to do with
                                    it in these states — see below for why
                                    discarding, not blocking, is correct)

        WHY DISCARD FRAMES DURING PROCESSING, RATHER THAN PAUSE THE READER:
        The reader thread keeps running unconditionally the whole time —
        it doesn't pause for PROCESSING or COOLDOWN. This is deliberate:
        pausing the reader would mean audio arriving DURING those states
        (e.g. you starting to say something new while Jarvis is still
        transcribing your last command) is simply never captured at all,
        which is worse than capturing it and discarding it, since a
        paused stream can also risk PyAudio-level buffer/driver issues on
        some backends when resumed. Discarding is a clean, cheap no-op;
        pausing is added complexity for no real benefit here, since
        PROCESSING is typically sub-second and COOLDOWN is capped at
        COOLDOWN_SECONDS.
        """
        while True:
            try:
                raw_frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # no frame arrived in the last second; loop and
                          # check again — keeps this loop responsive
                          # without blocking forever on an empty queue.

            if self._state == AssistantState.IDLE:
                self._consume_frame_idle(raw_frame)
            elif self._state == AssistantState.LISTENING:
                self._consume_frame_listening(raw_frame)
            elif self._state == AssistantState.SPEAKING:
                self._consume_frame_speaking(raw_frame)
            elif self._state == AssistantState.COOLDOWN:
                self._consume_frame_cooldown(raw_frame)
            # PROCESSING: frame discarded, no branch needed — falls
            # through to the next loop iteration.

    # -------------------------------------------------------------------
    # STATE: IDLE — checking for the wake word (or, in continuous mode,
    # for any speech starting at all — see _consume_frame_idle_continuous)
    # -------------------------------------------------------------------
    def _consume_frame_idle(self, raw_frame: bytes) -> None:
        if self._continuous_mode:
            self._consume_frame_idle_continuous(raw_frame)
            return

        audio_chunk = np.frombuffer(raw_frame, dtype=np.int16)

        # Track recent loudness BEFORE the model runs, so the window always
        # covers the audio leading up to whatever the model reacts to.
        self._recent_idle_rms.append(
            float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))
        )

        prediction = self._openwakeword_model.predict(audio_chunk)

        # Debounce: a genuine "hey citra" spans many frames, so require
        # WAKE_WORD_CONSECUTIVE_FRAMES_REQUIRED frames IN A ROW each above
        # threshold, not just one. See that constant's comment for why —
        # this is what actually catches a single noise-spike/reverb frame
        # that the threshold alone let through.
        frame_above_threshold = any(
            score > WAKE_WORD_DETECTION_THRESHOLD for score in prediction.values()
        )
        if not frame_above_threshold:
            self._wake_word_consecutive_frame_count = 0
            return

        self._wake_word_consecutive_frame_count += 1
        if self._wake_word_consecutive_frame_count < WAKE_WORD_CONSECUTIVE_FRAMES_REQUIRED:
            return
        self._wake_word_consecutive_frame_count = 0

        wake_word_name, score = max(prediction.items(), key=lambda item: item[1])

        # Energy gate — see WAKE_MIN_SPEECH_RMS. The model fires on an empty
        # room; nobody can say a wake word without making sound, so a wake
        # with no recent audio energy is provably not a real one. Logged at
        # INFO rather than dropped quietly: if this ever starts rejecting
        # real attempts, the log is where that shows up.
        recent_peak_rms = max(self._recent_idle_rms) if self._recent_idle_rms else 0.0
        if recent_peak_rms < WAKE_MIN_SPEECH_RMS:
            logger.info(
                "Ignoring wake (score=%.3f): room was silent, peak RMS %.0f < %d.",
                score, recent_peak_rms, WAKE_MIN_SPEECH_RMS,
            )
            return

        logger.info(
            "Wake word detected: %s (score=%.3f, peak RMS %.0f)",
            wake_word_name, score, recent_peak_rms,
        )
        self._state = AssistantState.LISTENING
        citra_ui_bridge.notify_state("LISTENING")
        _duck_other_audio()
        self._recording_frames = []
        self._recording_silent_frame_count = 0
        self._recording_speech_started = False
        self._speech_detector.reset()  # see _SpeechDetector.reset()'s
                                        # docstring — silero carries
                                        # recurrent state across calls.
        # A chime, NOT spoken words — see play_earcon_async()'s docstring
        # for why this swap is what actually breaks the self-triggering
        # feedback loop. Not awaited: it's ~150ms on its own thread, and
        # frames arriving during it are still correctly buffered by
        # _consume_frame_listening, so you can start talking immediately.
        play_earcon_async()
        return  # don't also process this same frame as LISTENING
                # input in the same iteration — the next frame
                # starts the LISTENING branch cleanly.

    def _consume_frame_idle_continuous(self, raw_frame: bytes) -> None:
        """
        Continuous mode's version of wake detection: no classifier, just
        "did speech start" via the same SILENCE_THRESHOLD end-of-speech
        detection already uses, applied in reverse. Reuses the existing
        LISTENING state/recording machinery unchanged once triggered —
        the only new thing here is HOW LISTENING gets entered, not what
        happens once it has.
        """
        audio_chunk = np.frombuffer(raw_frame, dtype=np.int16)
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))

        # Always feed the pre-roll buffer, whether or not this frame ends
        # up triggering — see CONTINUOUS_MODE_PREROLL_FRAMES's comment for
        # why the last few frames need to be captured before the trigger
        # actually fires, not starting from it.
        self._continuous_preroll.append(raw_frame)

        if rms < SILENCE_THRESHOLD:
            self._continuous_vad_frame_count = 0
            return

        self._continuous_vad_frame_count += 1
        if self._continuous_vad_frame_count < CONTINUOUS_MODE_VAD_CONSECUTIVE_FRAMES_REQUIRED:
            return
        self._continuous_vad_frame_count = 0

        logger.info("Continuous mode: speech detected (RMS %.0f), recording segment...", rms)
        self._state = AssistantState.LISTENING
        citra_ui_bridge.notify_state("LISTENING")
        # Seed the recording with the pre-roll frames so the debounced
        # onset (the ~240ms it took to confirm this wasn't a transient)
        # isn't lost from the start of what gets transcribed.
        self._recording_frames = list(self._continuous_preroll)
        self._recording_silent_frame_count = 0
        self._recording_speech_started = True  # continuous mode only enters
                                        # LISTENING because speech was
                                        # ALREADY detected (that's its
                                        # trigger), so the onset phase is
                                        # already satisfied here.
        self._speech_detector.reset()  # same reason as the wake-word path
        self._continuous_segment_pending = True
        # Deliberately NO earcon here, unlike the wake-word path above —
        # continuous mode can trigger on every sentence spoken anywhere
        # near the mic; chiming on each one would be constant and
        # unpleasant rather than a rare, useful confirmation.

    # -------------------------------------------------------------------
    # STATE: LISTENING — recording the command, watching for silence
    # -------------------------------------------------------------------
    def _consume_frame_listening(self, raw_frame: bytes) -> None:
        self._recording_frames.append(raw_frame)

        # Same silence-counting approach as the original
        # _record_until_silence, adapted to frame-at-a-time queue
        # consumption instead of a tight synchronous read loop. Frame
        # size here is fixed at 1280 samples (80ms) — this reuses the
        # SAME 80ms frames the whole pipeline is now built around.
        frame_duration = 1280 / 16000  # seconds per frame = 0.08
        silence_frames_needed = int(SILENCE_DURATION_SECONDS / frame_duration)
        max_frames = int(MAX_RECORDING_SECONDS / frame_duration)

        # Neural VAD first; None means it isn't available, in which case
        # fall back to the original RMS threshold — see _SpeechDetector's
        # comment for why the neural path exists and what the RMS one gets
        # measurably wrong in a room with the AC running.
        is_speech = self._speech_detector.is_speech(raw_frame)
        if is_speech is None:
            samples = np.frombuffer(raw_frame, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            is_speech = rms >= SILENCE_THRESHOLD

        if is_speech:
            self._recording_speech_started = True
            self._recording_silent_frame_count = 0
        else:
            self._recording_silent_frame_count += 1

        # WAIT FOR SPEECH TO ACTUALLY BEGIN before the end-of-speech timer
        # can fire. There is always a gap between the wake word ending and
        # the command starting — the user is drawing breath, or the earcon
        # is still playing. Counting that gap toward "they've stopped
        # talking" ends the recording before they've said anything, which
        # is exactly what happened when accurate VAD was first switched on:
        # every recording came out at precisely 0.800s (= 10 frames =
        # SILENCE_DURATION_SECONDS) and transcribed to nothing.
        #
        # This bug existed latently under the old RMS detector too, and was
        # masked by its own inaccuracy: room noise above SILENCE_THRESHOLD
        # counted as "speech", so the timer kept getting reset during the
        # gap. Making detection correct removed the accident that was
        # covering it. The fix is an explicit onset phase, which is what
        # the pipeline should have had regardless of detector.
        if not self._recording_speech_started:
            no_speech_frames = int(NO_SPEECH_TIMEOUT_SECONDS / frame_duration)
            if len(self._recording_frames) >= no_speech_frames:
                logger.info(
                    "No speech within %.1fs of the wake word — discarding.",
                    NO_SPEECH_TIMEOUT_SECONDS,
                )
                self._state = AssistantState.COOLDOWN
                citra_ui_bridge.notify_state("COOLDOWN")
                self._cooldown_start_time = time.monotonic()
            return

        recording_done = (
            self._recording_silent_frame_count >= silence_frames_needed
            or len(self._recording_frames) >= max_frames
        )

        if recording_done:
            self._state = AssistantState.PROCESSING
            citra_ui_bridge.notify_state("PROCESSING")
            pcm_bytes = b"".join(self._recording_frames)
            # Transcription + routing is genuinely slow (real model
            # inference, possibly a network call to LM Studio) — running
            # it INLINE here would block the consumer loop from pulling
            # any further frames off the queue for that whole duration.
            # Frames would still queue up correctly (no data loss — the
            # queue has no size cap), but the loop wouldn't be free to
            # notice a state change promptly. Running it on its own
            # thread keeps _consumer_loop free to keep pulling frames
            # (which, in PROCESSING, are simply discarded — see
            # _consumer_loop's docstring) and to promptly react once
            # this thread transitions state again.
            processing_thread = threading.Thread(
                target=self._process_recording,
                args=(pcm_bytes,),
                daemon=True,
            )
            processing_thread.start()

    def _process_recording(self, pcm_bytes: bytes) -> None:
        """Process recorded audio and route the recognized command.

        TIMING: logged so future latency tuning has real numbers to work
        from, same discipline as jarvis_router.py's own RouteResult
        timing — added during a latency pass that found
        SILENCE_DURATION_SECONDS's dead-air wait was actually the single
        biggest contributor in the whole pipeline, bigger than
        transcription itself, which nothing here would have surfaced
        without measuring it directly first."""
        # "GOT IT" - fires before anything else in this method, including
        # the model-readiness wait below, because its entire value is in
        # being immediate. The user has just stopped talking; every
        # millisecond between that and some acknowledgment is a
        # millisecond they spend wondering whether to say it again.
        play_working_earcon_async()

        # Background loading (see run()) normally finishes long before the
        # first command — this only ever actually blocks if someone speaks
        # within the first few seconds of launch, in which case they wait
        # exactly as long as they would have under the old load-everything-
        # upfront behavior. Never silently proceeds with a half-initialized
        # pipeline.
        #
        # Deliberately timed OUTSIDE processing_start below: this is a
        # one-off startup artifact, not a per-command cost, and folding it
        # into the transcription number would corrupt exactly the
        # measurements this instrumentation exists to inform.
        if not self._heavy_models_ready.is_set():
            logger.info("Command arrived before background loading finished — waiting...")
            wait_start = time.perf_counter()
            self._heavy_models_ready.wait()
            logger.info("Waited %.1fs for background loading.", time.perf_counter() - wait_start)

        processing_start = time.perf_counter()
        audio_array = _pcm_bytes_to_numpy(pcm_bytes)

        logger.info("Transcribing...")
        # Reuse the language we detected recently rather than paying for
        # ~99-language detection on every utterance. See
        # LANGUAGE_CACHE_SECONDS for the measurements behind this.
        cached_language = None
        if (self._cached_language is not None
                and time.monotonic() - self._cached_language_at < LANGUAGE_CACHE_SECONDS):
            cached_language = self._cached_language

        segments, info = self._whisper_model.transcribe(
            audio_array,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=WHISPER_USE_VAD_FILTER,
            language=cached_language,
        )
        transcribed_text = " ".join(segment.text for segment in segments).strip()
        transcribe_ms = (time.perf_counter() - processing_start) * 1000

        # Only a genuine DETECTION pass may seed the cache. When a language
        # was forced, info.language just echoes it back at probability 1.0,
        # so caching that would be caching our own assumption and the value
        # could never be corrected.
        if cached_language is None and transcribed_text:
            if (info.language in ALLOWED_TRANSCRIPTION_LANGUAGES
                    and info.language_probability >= MIN_LANGUAGE_PROBABILITY):
                self._cached_language = info.language
                self._cached_language_at = time.monotonic()
                logger.info("Language detected as '%s' (%.2f) - reusing it for %.0f minutes.",
                            info.language, info.language_probability,
                            LANGUAGE_CACHE_SECONDS / 60)

        if not transcribed_text:
            # Silent discard, NOT a spoken "I didn't catch that". By far
            # the most common way to reach here is a spurious wake with
            # nobody actually talking — answering that out loud is how the
            # "Citra talking to herself" complaint happened. If you really
            # did speak and weren't heard, the UI's state indicator already
            # showed it listening and returning to idle, which is the same
            # feedback a phone assistant gives in this situation.
            logger.info("Empty transcription (likely a spurious wake) — discarding silently.")
            self._state = AssistantState.COOLDOWN
            citra_ui_bridge.notify_state("COOLDOWN")
            self._cooldown_start_time = time.monotonic()
            return

        # Discard results where language auto-detect itself wasn't
        # confident, or landed outside the languages this assistant
        # actually supports — see ALLOWED_TRANSCRIPTION_LANGUAGES /
        # MIN_LANGUAGE_PROBABILITY's comment for the real incident this
        # targets (near-silence auto-detected as Greek at 19% confidence,
        # forwarded to the router anyway). This is a silent discard, not
        # a spoken "didn't catch that" — a low-confidence misdetection is
        # far more likely to be noise/echo than a genuine failed attempt
        # at a supported language, and reacting audibly to every such
        # blip would be its own annoyance.
        if (
            info.language not in ALLOWED_TRANSCRIPTION_LANGUAGES
            or info.language_probability < MIN_LANGUAGE_PROBABILITY
        ):
            logger.info(
                "Discarding transcription in unsupported/low-confidence language: "
                "'%s' (language=%s, probability=%.2f)",
                transcribed_text, info.language, info.language_probability,
            )
            self._state = AssistantState.COOLDOWN
            citra_ui_bridge.notify_state("COOLDOWN")
            self._cooldown_start_time = time.monotonic()
            return

        normalized_text = transcribed_text.lower().strip().strip(".,!?")

        if len(normalized_text) < 5 or normalized_text in _WHISPER_HALLUCINATIONS:
            logger.info(
                "Discarding likely Whisper hallucination: '%s' (normalized: '%s')",
                transcribed_text, normalized_text,
            )
            self._state = AssistantState.COOLDOWN
            citra_ui_bridge.notify_state("COOLDOWN")
            self._cooldown_start_time = time.monotonic()
            return

        logger.info(
            "Transcribed (%.0fms): '%s' (detected language: %s)",
            transcribe_ms, transcribed_text, info.language,
        )

        # CONTINUOUS MODE KEYWORD GATE — read-and-clear ONCE per recording;
        # see self._continuous_segment_pending's own comment in __init__
        # for why a wake-word-triggered recording never reaches here with
        # this True. A segment that doesn't mention Citra was never meant
        # for her — discarded silently (no spoken response, same as the
        # other discard branches above), same COOLDOWN transition they all
        # use, and _route_and_respond/_enter_speaking_state are skipped
        # entirely since nothing is going to be said. When it DOES mention
        # her, transcribed_text is replaced with the name-stripped version
        # — see _extract_command_without_name's docstring for why leaving
        # the name in measurably hurts routing quality.
        if self._continuous_segment_pending:
            self._continuous_segment_pending = False
            cleaned_command = _extract_command_without_name(transcribed_text)
            if cleaned_command is None:
                logger.info("Continuous mode: name not mentioned, discarding: '%s'", transcribed_text)
                self._state = AssistantState.COOLDOWN
                citra_ui_bridge.notify_state("COOLDOWN")
                self._cooldown_start_time = time.monotonic()
                return
            transcribed_text = cleaned_command

        citra_ui_bridge.notify_transcript(transcribed_text)
        self._route_and_respond(transcribed_text)
        total_ms = (time.perf_counter() - processing_start) * 1000
        logger.info(
            "Recording-to-response: %.0fms total (%.0fms transcription, %.0fms routing)",
            total_ms, transcribe_ms, total_ms - transcribe_ms,
        )
        self._enter_speaking_state()

    def _enter_speaking_state(self) -> None:
        """
        Transitions to SPEAKING. speak_async() itself was already called
        by the handler/route logic before this runs (protocol handlers
        call it internally; _process_recording calls it directly for the
        empty-transcription case) — this method's job is purely the
        STATE bookkeeping around that call, not calling speak_async()
        again.

        HOW _consume_frame_speaking KNOWS WHEN SPEECH ACTUALLY ENDS:
        Rather than this method threading a specific Thread reference
        through to the consumer loop (awkward, since protocol handlers
        call speak_async() internally and don't hand this method their
        thread), _consume_frame_speaking polls
        _tts_is_currently_playing() — a small helper added alongside this
        restructure that reports whether Piper/sounddevice playback is
        presently active. This keeps SPEAKING's exit condition based on
        the TTS layer's own ground truth (is audio actually still
        coming out of the speaker) rather than an indirect proxy.
        """
        self._barge_in_loud_frame_count = 0
        self._state = AssistantState.SPEAKING
        citra_ui_bridge.notify_state("SPEAKING")

    # -------------------------------------------------------------------
    # STATE: SPEAKING — waiting for TTS playback to finish on its own
    # (barge-in interruption is disabled — see the comment below)
    # -------------------------------------------------------------------
    def _consume_frame_speaking(self, raw_frame: bytes) -> None:
        # If nothing is actually playing anymore (the TTS thread
        # finished naturally), move to COOLDOWN.
        if not _tts_is_currently_playing():
            self._state = AssistantState.COOLDOWN
            citra_ui_bridge.notify_state("COOLDOWN")
            self._cooldown_start_time = time.monotonic()
            return

        # Barge-in (interrupting Citra mid-speech by talking over her) is
        # disabled by explicit request — while it turned out NOT to be the
        # actual cause of a "yes sir" self-triggering loop (that was
        # WAKE_WORD_DETECTION_THRESHOLD being miscalibrated, see its own
        # comment), listening for loud frames while she's actively playing
        # audio through the same room's speakers is still one more thing
        # that can mistake her own voice/echo for a user talking over her.
        # Removing it means she now always finishes speaking on her own
        # before COOLDOWN -> IDLE re-arms wake-word listening, same as the
        # "completed naturally" branch above already did.
        return

    # -------------------------------------------------------------------
    # STATE: COOLDOWN — brief pause before re-arming wake-word detection
    # -------------------------------------------------------------------
    def _consume_frame_cooldown(self, raw_frame: bytes) -> None:
        # Frame content itself is irrelevant during COOLDOWN — we're
        # only waiting for time to pass. See COOLDOWN_SECONDS's comment
        # for why this state exists at all (closing the TTS-echo loop).
        if time.monotonic() - self._cooldown_start_time >= COOLDOWN_SECONDS:
            self._state = AssistantState.IDLE
            citra_ui_bridge.notify_state("IDLE")
            _restore_other_audio()  # see that function's own docstring
                                    # for why this one call site (reached
                                    # regardless of which path this wake
                                    # cycle took) is enough.

            # STALE-BUFFER FIX — found via a real session log showing
            # Citra re-triggering shortly after answering, on genuine
            # (non-silent) RMS readings that were suspiciously IDENTICAL
            # across two separate "Wake word detected" log lines a few
            # seconds apart. Neither self._recent_idle_rms (the energy
            # gate's rolling window) nor self._openwakeword_model's own
            # internal audio-feature buffer is touched by anything during
            # LISTENING/PROCESSING/SPEAKING/COOLDOWN — only
            # _consume_frame_idle feeds either of them, and that method
            # only runs in IDLE. So both were sitting frozen with
            # whatever they held from the frames immediately before the
            # ORIGINAL wake trigger — for as long as ten-plus seconds,
            # across recording, transcription, and a full answer — and
            # when IDLE resumed, those STALE readings were still eligible
            # to satisfy the energy gate and feed the model's scoring for
            # up to ~1-2s (however many fresh frames it takes to flush a
            # bounded/sliding buffer), even in an already-silent room.
            # That's a real, distinct bug from anything the energy gate
            # or the retrained model were meant to fix — it's a state-
            # leak across the gap, not a threshold or model-quality
            # problem, and it only became visible once the retrained
            # model got accurate enough for genuine multi-turn sessions
            # to actually exercise this path repeatedly.
            #
            # Clearing both here means IDLE always starts from a genuinely
            # clean slate — no memory of the room or the model's own
            # feature buffer carries across from before the last command.
            self._recent_idle_rms.clear()
            self._openwakeword_model.reset()

    def _route_and_respond(self, transcribed_text: str) -> None:
        """
        Stage 2+3+4 combined: try semantic routing first; if no protocol
        clears the similarity threshold, fall through to the Smart Path
        (LM Studio via jarvis_router.py) and speak its answer.

        UNCHANGED FROM LAYER 1: this method's logic doesn't touch audio
        I/O directly — it just decides WHAT to say and calls
        speak_async() — so it needed no changes for Layer 2's
        concurrency restructure. It's called from _process_recording
        (running on that thread), not from the consumer loop directly.

        Checked here (rather than as a semantic protocol) since toggling
        continuous mode needs direct access to self, which
        ProtocolIntent's handler signature — (SmartRoomController, text)
        — doesn't carry. Simple substring matching, deliberately, the
        same reasoning jarvis_router.py's own fast path uses for
        deterministic commands: this needs to be unambiguous, not fuzzy-
        matched against conversation.
        """
        normalized = transcribed_text.lower()
        if any(phrase in normalized for phrase in CONTINUOUS_MODE_ON_PHRASES):
            self._continuous_mode = True
            self._continuous_vad_frame_count = 0
            self._continuous_preroll.clear()
            logger.info("Continuous listening mode: ON")
            speak_async("Continuous listening on. I'll only respond when I hear my name.")
            return
        if any(phrase in normalized for phrase in CONTINUOUS_MODE_OFF_PHRASES):
            self._continuous_mode = False
            logger.info("Continuous listening mode: OFF")
            speak_async("Back to wake-word mode.")
            return

        matched_protocol = self.semantic_router.route(transcribed_text)

        if matched_protocol is not None:
            logger.info("Routed to %s", matched_protocol.name)
            # The handler itself calls speak_async() and starts its own
            # hardware-dispatch thread internally (see
            # _handle_lighting_protocol / _handle_cooling_protocol above)
            # — this method doesn't need to do either; it just invokes
            # the handler and lets it manage its own parallel speak+act.
            matched_protocol.handler(self.controller, transcribed_text)
            return

        # No protocol matched -> Smart Path. This reuses
        # jarvis_router.py's JarvisRouter.route(), which itself tries the
        # REGEX fast path first (harmless — it'll also fail to match
        # anything a semantic protocol didn't already catch, in the
        # common case) and falls through to LM Studio.
        logger.info("No protocol matched — falling back to Smart Path (LM Studio)")

        # THE ONE PLACE IN THIS ASSISTANT THAT GOES QUIET. Every other
        # path speaks the instant it understands you - the protocol
        # handlers above call speak_async() and dispatch hardware on a
        # separate thread, so "Turning on light 2, sir." is already
        # leaving the speaker while the relay is still clicking. This
        # call blocks on a network round trip instead, and without the
        # narrator the room hears nothing at all for the duration.
        #
        # try/finally is not defensive habit here, it is load-bearing: if
        # route() raises, the timers are still armed, and a progress line
        # firing after the error response had already been spoken would
        # have Citra talking over herself.
        self._narrator.start(transcribed_text)
        try:
            result = self.text_router.route(transcribed_text)
        finally:
            narrated = self._narrator.finish()
        if narrated:
            logger.info("Spoke a progress line while the Smart Path ran.")

        if result.success:
            speak_async(result.message)
        else:
            speak_async("Sorry sir, I ran into a problem answering that.")
            logger.error("Smart Path failure: %s", result.message)

    def _cleanup(self) -> None:
        # NOTE ON openWakeWord CLEANUP: unlike Porcupine's handle.delete(),
        # openwakeword.Model has no explicit close/delete/release method —
        # checked against the library's official README, its source-level
        # walkthrough, and multiple independent usage examples, none of
        # which call anything on the model at shutdown. It's a plain
        # Python object wrapping ONNX Runtime inference sessions
        # underneath; letting self._openwakeword_model go out of scope
        # (which happens naturally once this method returns and the
        # assistant object is done with it) is sufficient — Python's
        # garbage collector and ONNX Runtime's own destructors handle
        # releasing those sessions. Calling a nonexistent .delete() here
        # would raise AttributeError on exit instead of cleaning up
        # anything, so this is intentionally a no-op for
        # self._openwakeword_model rather than a guess dressed up as a
        # real cleanup step.
        #
        # LAYER 2 ADDITION: the single continuous input stream (opened
        # in run()) is closed here too — previously _listen_loop opened
        # and closed its own wake-word stream internally with a
        # try/finally; now that stream ownership lives at the instance
        # level (self._input_stream) for the assistant's whole lifetime,
        # closing it belongs in the same centralized cleanup path as the
        # PyAudio instance itself.
        if self._input_stream is not None:
            self._input_stream.stop_stream()
            self._input_stream.close()
        if self._pyaudio_instance is not None:
            self._pyaudio_instance.terminate()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CITRA V2 — VOICE ASSISTANT")
    print("=" * 70)
    print("Pipeline: Wake Word -> STT -> Semantic Routing -> TTS/Hardware")
    print()
    print("Registered protocols:")
    for intent in PROTOCOL_REGISTRY:
        print(f"  - {intent.name} ({len(intent.example_phrases)} example phrases)")
    print()
    print("Anything not matching a protocol falls back to LM Studio (Smart Path).")
    print("Press Ctrl+C to stop.")
    print("=" * 70)
    print()

    assistant = JarvisVoiceAssistant()
    assistant.run()