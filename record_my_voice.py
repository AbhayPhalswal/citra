"""
=============================================================================
RECORD MY VOICE — real training data for the custom "hey citra" wake word
=============================================================================
Every sample the wake word model has been trained on so far is synthetic
TTS (piper-sample-generator imitating "hey citra" in various synthetic
voices) — it has never once heard the actual person it needs to recognize,
in their actual room, through their actual microphone. This script fixes
that gap: it records real "hey citra" utterances (and some real negative
speech/background noise) to fold into the next training round alongside
the synthetic data.

USAGE:
    python record_my_voice.py positive
        Records repeated "hey citra" utterances. Just keep saying it with
        a short pause between each — auto-segmented by silence detection
        (same RMS-threshold approach jarvis_voice_assistant.py already
        uses for recording commands), so there's no button to press
        between repetitions. Aim for 60-100+ clips: vary your distance
        from the mic, your tone, and pace a little across repetitions —
        that variety matters more than raw count past a certain point.
        Ctrl+C to stop early; it also stops automatically at TARGET_COUNT.

    python record_my_voice.py negative
        Records continuous background audio (silence, room noise, or you
        talking about anything OTHER than "hey citra") sliced into fixed-
        length chunks. Useful negative data because it's real background
        noise from the actual deployment room, not just AudioSet's
        generic clips. Run for a couple of minutes; Ctrl+C to stop.

Output goes to ./my_voice_positive/ and ./my_voice_negative/ as 16kHz
mono WAV files, ready to hand to the WSL training pipeline.
=============================================================================
"""
import os
import sys
import time
import wave

import numpy as np
import pyaudio

SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80ms frames -- same convention jarvis_voice_assistant.py uses

# Positive-mode silence segmentation
SILENCE_THRESHOLD = 500        # same RMS floor as the main assistant's own recording logic
SILENCE_DURATION_SECONDS = 0.7  # shorter than the main assistant's 1.5s --
                                  # "hey citra" is a quick phrase, we want
                                  # a snappy pause between repetitions, not
                                  # a long one
MIN_UTTERANCE_SECONDS = 0.35    # discard anything shorter (a cough, a click)
MAX_UTTERANCE_SECONDS = 2.5     # safety ceiling per utterance
TARGET_COUNT = 100              # auto-stop after this many positive clips

# Negative-mode fixed-length chunking
NEGATIVE_CHUNK_SECONDS = 2.0
NEGATIVE_TARGET_MINUTES = 3.0

OUTPUT_DIR_POSITIVE = "my_voice_positive"
OUTPUT_DIR_NEGATIVE = "my_voice_negative"


def _open_stream(pa: pyaudio.PyAudio) -> pyaudio.Stream:
    return pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    )


def _save_wav(path: str, pcm_bytes: bytes) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def record_positive() -> None:
    os.makedirs(OUTPUT_DIR_POSITIVE, exist_ok=True)
    existing = len([f for f in os.listdir(OUTPUT_DIR_POSITIVE) if f.endswith(".wav")])

    print(f"=== Recording 'hey citra' utterances (existing: {existing}, target: {TARGET_COUNT}) ===")
    print("Say 'hey citra' clearly, pause briefly, say it again. Vary your")
    print("distance/tone/pace a little across repetitions. Ctrl+C to stop early.\n")
    time.sleep(1.5)

    pa = pyaudio.PyAudio()
    stream = _open_stream(pa)

    count = existing
    buffer = []
    in_speech = False
    silent_frames = 0
    silence_frames_needed = int(SILENCE_DURATION_SECONDS * SAMPLE_RATE / FRAME_SIZE)
    max_frames = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE / FRAME_SIZE)

    try:
        while count < TARGET_COUNT:
            raw = stream.read(FRAME_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

            if rms > SILENCE_THRESHOLD:
                if not in_speech:
                    in_speech = True
                    buffer = []
                buffer.append(raw)
                silent_frames = 0
            elif in_speech:
                buffer.append(raw)
                silent_frames += 1
                if silent_frames >= silence_frames_needed or len(buffer) >= max_frames:
                    duration = len(buffer) * FRAME_SIZE / SAMPLE_RATE
                    if duration >= MIN_UTTERANCE_SECONDS:
                        count += 1
                        path = os.path.join(OUTPUT_DIR_POSITIVE, f"{count:04d}.wav")
                        _save_wav(path, b"".join(buffer))
                        print(f"  [{count}/{TARGET_COUNT}] saved ({duration:.2f}s)")
                    in_speech = False
                    buffer = []
                    silent_frames = 0
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    print(f"\nDone. {count} clips in ./{OUTPUT_DIR_POSITIVE}/")


def record_negative() -> None:
    os.makedirs(OUTPUT_DIR_NEGATIVE, exist_ok=True)
    existing = len([f for f in os.listdir(OUTPUT_DIR_NEGATIVE) if f.endswith(".wav")])

    print(f"=== Recording background/other-speech ({NEGATIVE_CHUNK_SECONDS}s chunks) ===")
    print("Talk about anything (NOT 'hey citra'), or just leave this running to")
    print(f"capture room background noise. Runs ~{NEGATIVE_TARGET_MINUTES} min. Ctrl+C to stop early.\n")
    time.sleep(1.5)

    pa = pyaudio.PyAudio()
    stream = _open_stream(pa)

    frames_per_chunk = int(NEGATIVE_CHUNK_SECONDS * SAMPLE_RATE / FRAME_SIZE)
    total_chunks_target = int(NEGATIVE_TARGET_MINUTES * 60 / NEGATIVE_CHUNK_SECONDS)

    count = existing
    buffer = []

    try:
        for _ in range(total_chunks_target):
            buffer = []
            for _ in range(frames_per_chunk):
                raw = stream.read(FRAME_SIZE, exception_on_overflow=False)
                buffer.append(raw)
            count += 1
            path = os.path.join(OUTPUT_DIR_NEGATIVE, f"{count:04d}.wav")
            _save_wav(path, b"".join(buffer))
            print(f"  [{count - existing}/{total_chunks_target}] saved chunk")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    print(f"\nDone. {count} chunks in ./{OUTPUT_DIR_NEGATIVE}/")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("positive", "negative"):
        print("Usage: python record_my_voice.py positive|negative")
        sys.exit(1)

    if sys.argv[1] == "positive":
        record_positive()
    else:
        record_negative()
