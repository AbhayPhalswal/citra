"""
MODEL ROUTING - which brain answers which question.

Citra now has three very different models available, and the whole point
of this module is that they are NOT interchangeable. Picking by task
rather than by preference is what keeps "turn on the lights" instant
while still letting "build me a working website" take as long as it
needs, without a plain "write me a script" paying that same cost.

    LANE      MODEL                        WHERE      MEASURED
    text      gemini-3.5-flash-lite        cloud      ~1.1s
    vision    qwen2.5-vl-7b-instruct       local      ~24s (downscaled)
    heavy     opencode/nemotron-3-ultra    cloud      ~74s, 128k context

EVERY ONE OF THOSE NUMBERS WAS MEASURED ON THIS MACHINE, not read off a
datasheet. The Nemotron figure is a real `opencode run` round trip for a
one-paragraph answer; do not assume a long code generation is faster.

WHY NEMOTRON DOES NOT GET IMAGES: asked directly, it answered "I cannot
see or process images - I only work with text input and output." So the
vision lane stays on Qwen-VL regardless of the cloud/local switch. This
is not a policy choice, it is a capability limit.

WHY TEXT ALWAYS GOES TO GEMINI: at ~1.1s it is the only one of the three
fast enough to answer someone standing in a room waiting. Nemotron is
sixty-odd times slower for the same question. "Better model" is the wrong
axis when the question is "what time is it".

CODE-WRITING IS A SPECIAL CASE, not a fourth lane: write_python_code (the
tool jarvis_router dispatches to) picks EITHER model itself, per request.
Defaults to Gemini - most requests are one ordinary script, and ~2s beats
~70s when the extra time buys nothing. It only escalates to Nemotron when
the user explicitly names it, or the request is large in scope on its own
(a full website, a multi-file app) - see CODE_TOOL_SCHEMA's "model"
parameter for the exact wording the LLM is given to decide this.

THE CLOUD/LOCAL SWITCH is deliberately narrow. It does NOT mean "run
everything locally" - there is no local model on this machine that can
do what Nemotron does, and pretending otherwise would just make Citra
worse without telling anyone. It means: prefer local where a local
option genuinely exists, and say plainly when one does not.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# --------------------------------------------------------------------------
# MODELS
# --------------------------------------------------------------------------
NEMOTRON_MODEL = "opencode/nemotron-3-ultra-free"
NEMOTRON_CONTEXT_TOKENS = 128_000

# Measured: 74s for a one-paragraph reply. A long file will be longer, so
# the ceiling is generous. Anything that hits this is a genuine failure,
# not slowness - opencode has already given up or hung by then.
NEMOTRON_TIMEOUT_SECONDS = 900.0

VISION_MODEL_LOCAL = "qwen2.5-vl-7b-instruct"        # via LM Studio
VISION_MODEL_CLOUD = "cloudflare-workers-ai/@cf/meta/llama-3.2-11b-vision-instruct"

logger = logging.getLogger("citra_models")

PREFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "citra_model_preference.json")


# --------------------------------------------------------------------------
# LANES
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Lane:
    name: str
    model: str
    where: str
    typical_seconds: float
    why: str


TEXT = Lane("text", "gemini-3.5-flash-lite", "cloud", 1.1,
            "Fast enough to answer someone standing in the room.")
VISION = Lane("vision", VISION_MODEL_LOCAL, "local", 24.0,
              "Screenshots never leave the flat. Nemotron cannot see images at all.")
HEAVY = Lane("heavy", NEMOTRON_MODEL, "cloud", 74.0,
             "128k context and genuinely strong reasoning, at sixty times the latency.")


# WHAT THE HEAVY LANE IS ACTUALLY FOR.
#
# The test is not "is this hard?" - it is "would I accept waiting a full
# minute or more for the answer?". If the person is standing there
# expecting a reply, the answer is no, however clever the model is.
HEAVY_TASKS = [
    ("Write a full app or website", "Multi-file, real backend and frontend - big enough that a "
                                    "quick script is not a fair comparison."),
    ("Refactor or review a long file", "128k context means a whole module fits in one go."),
    ("Explain a codebase", "Point it at several files and ask how they fit together."),
    ("Debug from a log", "Paste a long traceback plus the source and let it reason."),
    ("Write documentation", "READMEs, wiring notes, install guides from rough bullets."),
    ("Plan something multi-step", "An install plan, a migration, a build order."),
    ("Summarise a long document", "A datasheet, a spec, a society circular."),
    ("Draft long-form text", "A resident letter, a pitch, an explanation for a neighbour."),
]

NOT_HEAVY_TASKS = [
    ("An ordinary script or small program", "Defaults to Gemini, ~2s - see write_python_code's "
                                            "'model' parameter for the actual escalation rule."),
    ("Anything with an image", "Nemotron is text-only, confirmed by asking it. Use the vision lane."),
    ("Switching lights or the AC", "Regex fast path, milliseconds. Never involve a model."),
    ("Questions with someone waiting", "'What time is it' at 74 seconds is a broken product."),
    ("Reminders and schedules", "Local, instant, no model needed."),
]


# --------------------------------------------------------------------------
# CLOUD / LOCAL PREFERENCE
# --------------------------------------------------------------------------
def get_preference() -> str:
    """'auto' (default), 'local', or 'cloud'."""
    try:
        with open(PREFERENCE_PATH, encoding="utf-8") as fh:
            return json.load(fh).get("preference", "auto")
    except Exception:
        return "auto"


def set_preference(value: str) -> str:
    if value not in ("auto", "local", "cloud"):
        raise ValueError("preference must be auto, local or cloud")
    with open(PREFERENCE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"preference": value}, fh, indent=2)
    return value


def describe_routing() -> str:
    """One human-readable block, for the chat and for `--explain`."""
    pref = get_preference()
    lines = [f"Preference: {pref}", ""]
    for lane in (TEXT, VISION, HEAVY):
        lines.append(f"  {lane.name:<7} {lane.model:<42} {lane.where:<6} ~{lane.typical_seconds:g}s")
    if pref == "local":
        lines.append("")
        lines.append("  Note: 'local' still sends text to Gemini and heavy work to Nemotron.")
        lines.append("  There is no local model on this machine that can do either job;")
        lines.append("  saying otherwise would quietly make Citra worse. Only vision is")
        lines.append("  genuinely local.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# NEMOTRON, VIA OPENCODE
# --------------------------------------------------------------------------
def _opencode_exe() -> str | None:
    """
    The path Windows can actually CreateProcess.

    npm installs BOTH an extensionless shell script (for Git Bash / WSL)
    and a .cmd shim (for Windows). shutil.which("opencode") finds the
    shell script first when running under Git Bash, and handing that to
    subprocess with shell=False fails with a bare
    "[WinError 2] The system cannot find the file specified" - which
    looks exactly like opencode not being installed, and cost a test run
    to diagnose. Ask for the .cmd explicitly first.
    """
    for candidate in ("opencode.cmd", "opencode.exe", "opencode"):
        found = shutil.which(candidate)
        if found:
            # An extensionless match on Windows is the unrunnable shell
            # script; only accept it on platforms where it works.
            if os.name == "nt" and not found.lower().endswith((".cmd", ".exe", ".bat")):
                continue
            return found
    return None


def opencode_available() -> bool:
    return _opencode_exe() is not None


def ask_nemotron(prompt: str, timeout: float = NEMOTRON_TIMEOUT_SECONDS) -> tuple:
    """
    One blocking call to Nemotron. Returns (ok, text, seconds).

    BLOCKING, and it will block for over a minute. Nothing on the voice
    path may call this directly - see run_in_background() below, which is
    what the assistant should use so a 74-second job does not freeze the
    thread that also answers "turn on the lights".
    """
    exe = _opencode_exe()
    if exe is None:
        return False, "opencode is not installed or not on PATH.", 0.0

    # FLATTEN TO ONE LINE. On Windows `opencode` is a .cmd batch shim, and
    # cmd.exe truncates an argument at the first newline - so a multi-line
    # prompt arrives with everything after line one silently missing. That
    # failed in the most confusing way available: the model got the
    # instructions but not the request, and politely replied "You haven't
    # specified what you want the Python code to do." Nothing errored. It
    # just quietly answered a question nobody asked.
    flat = " ".join(prompt.split())

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [exe, "run", flat, "-m", NEMOTRON_MODEL],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"Nemotron did not answer within {timeout:.0f}s.", time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and not out:
        return False, (proc.stderr or "opencode failed with no output").strip()[:400], elapsed

    # opencode prints a banner and a "> build · model" header before the
    # answer. Strip those rather than handing them to the user as if the
    # model had said them.
    lines = [line for line in out.splitlines() if line.strip()]
    while lines and (lines[0].lstrip().startswith(">") or set(lines[0].strip()) <= set("█▀▄ ⠀")):
        lines.pop(0)
    return True, "\n".join(lines).strip(), elapsed


def run_in_background(prompt: str, on_done: Callable[[bool, str, float], None]) -> threading.Thread:
    """
    Fire a Nemotron job and call on_done(ok, text, seconds) when it lands.

    The assistant stays responsive throughout: you can keep switching
    lights and asking Gemini questions while this runs. That matters more
    here than anywhere else in the system, because this lane is
    measured in MINUTES.
    """
    def _work():
        ok, text, secs = ask_nemotron(prompt)
        try:
            on_done(ok, text, secs)
        except Exception:
            pass

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    print(describe_routing())
    print("\nHEAVY LANE IS FOR:")
    for name, why in HEAVY_TASKS:
        print(f"  {name:<34} {why}")
    print("\nHEAVY LANE IS NOT FOR:")
    for name, why in NOT_HEAVY_TASKS:
        print(f"  {name:<34} {why}")
    print(f"\nopencode on PATH: {opencode_available()}")


# --------------------------------------------------------------------------
# GEMINI, FOR WHEN SPEED MATTERS MORE THAN DEPTH
# --------------------------------------------------------------------------
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_CODE_MODEL = "gemini-3.5-flash-lite"
GEMINI_CODE_TIMEOUT_SECONDS = 60.0


def ask_gemini(prompt: str, timeout: float = GEMINI_CODE_TIMEOUT_SECONDS) -> tuple:
    """
    One plain completion from Gemini. Returns (ok, text, seconds).

    Deliberately NOT routed through JarvisRouter: that path carries the
    whole 41-tool schema and the conversation history, neither of which
    belongs in "write me a script". This is a bare generateContent call,
    which is both faster and far cheaper per request.

    Roughly SIXTY TIMES faster than the Nemotron lane (~1.1s against
    ~74s), with a smaller context and less depth. That is exactly the
    trade the caller is making when they say "write it with Gemini".
    """
    import requests

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return False, "GEMINI_API_KEY is not set.", 0.0

    t0 = time.perf_counter()
    try:
        response = requests.post(
            f"{GEMINI_BASE_URL}/models/{GEMINI_CODE_MODEL}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 8192}},
            timeout=timeout,
        )
    except Exception as exc:
        return False, f"Gemini call failed: {exc}", time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    if response.status_code != 200:
        return False, f"Gemini returned {response.status_code}: {response.text[:200]}", elapsed
    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        return True, "".join(p.get("text", "") for p in parts).strip(), elapsed
    except (KeyError, IndexError, ValueError) as exc:
        return False, f"Could not read Gemini's reply: {exc}", elapsed


# --------------------------------------------------------------------------
# WRITE CODE -> NOTEPAD
# --------------------------------------------------------------------------
GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_code")

# Asking for ONLY code matters more than it looks. Left to itself the model
# returns an explanation, then a fenced block, then usage notes - all of
# which would land in the .py file and stop it running. The fence stripping
# below is a second line of defence, not the primary one.
CODE_PROMPT = (
    "Write complete, runnable Python for this request. Output ONLY the code - "
    "no explanation before or after, no markdown fences, no commentary. Include "
    "a brief module docstring and inline comments where a reader would genuinely "
    "need them. The request is: {request}"
)


def _strip_fences(text: str) -> str:
    """Remove ```python ... ``` if the model added it despite being asked not to."""
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        while lines and not lines[-1].lstrip().startswith("```"):
            if lines[-1].strip() == "":
                lines.pop()
            else:
                break
        if lines and lines[-1].lstrip().startswith("```"):
            lines.pop()
    return "\n".join(lines).strip() + "\n"


def _safe_slug(text: str, limit: int = 40) -> str:
    keep = [c if (c.isalnum() or c in " -_") else " " for c in text.lower()]
    words = "".join(keep).split()
    return ("_".join(words))[:limit] or "script"


def write_code_to_notepad(request: str, on_done: Callable | None = None,
                          model: str = "gemini") -> threading.Thread:
    """
    Generate Python for `request`, save it, open Notepad.

    DEFAULTS TO GEMINI, NOT NEMOTRON. Most code requests are one ordinary
    script, and ~2s beats ~70s when the extra time buys nothing. The
    caller only passes model="nemotron" when the user explicitly asked
    for it or named it, or the task is genuinely large on its own - a
    full working website, a multi-file app with a real backend and
    frontend. See CODE_TOOL_SCHEMA's "model" description for the actual
    wording an LLM is given to decide this.

    Runs entirely in the background either way. Nemotron is measured at
    up to ~70s+ for a real request, so blocking the assistant on it would
    mean the lights stop responding while she writes code - not a trade
    anyone would accept. She stays fully usable throughout and Notepad
    simply appears when it is done.

    on_done(ok, path_or_error, seconds) fires on completion, so the caller
    can speak or toast the result.
    """
    os.makedirs(GENERATED_DIR, exist_ok=True)

    def _finished(ok, text, secs):
        if not ok:
            if on_done:
                on_done(False, text, secs)
            return
        code = _strip_fences(text)
        name = f"{_safe_slug(request)}_{time.strftime('%H%M%S')}.py"
        path = os.path.join(GENERATED_DIR, name)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
            # notepad.exe explicitly, not os.startfile: .py is usually
            # associated with the Python launcher, so "open" would RUN the
            # freshly generated file instead of showing it. Running unseen
            # generated code is not what "open notepad" means.
            subprocess.Popen(["notepad.exe", path])
            if on_done:
                on_done(True, path, secs)
        except Exception as exc:
            if on_done:
                on_done(False, f"could not save or open the file: {exc}", secs)

    prompt = CODE_PROMPT.format(request=request)
    if model == "gemini":
        # Same background thread either way, so the caller's contract does
        # not change with the model - only how long the wait is.
        def _work():
            _finished(*ask_gemini(prompt))
        thread = threading.Thread(target=_work, daemon=True)
        thread.start()
        return thread
    return run_in_background(prompt, _finished)


# --------------------------------------------------------------------------
# TOOL: write_python_code
# --------------------------------------------------------------------------
class CodeController:
    """
    The write-code tool, shaped like every other controller in this
    project so jarvis_router can dispatch to it unchanged.

    RETURNS IMMEDIATELY, ALWAYS. The generation itself takes minutes, and
    a tool call that blocked for that long would hold the Smart Path open,
    stall the conversation, and almost certainly trip a timeout somewhere
    upstream. So this starts the job and reports that it started; Notepad
    opens on its own when the code is ready, and notify_fn (if given) says
    so out loud.
    """

    def __init__(self, notify_fn: Callable[[str], None] | None = None):
        self._notify = notify_fn

    def write_python_code(self, request: str, model: str = "gemini"):
        from jarvis_hardware_api import HardwareResult

        model = (model or "gemini").strip().lower()
        if model not in ("nemotron", "gemini"):
            model = "gemini"

        if model == "nemotron" and not opencode_available():
            return HardwareResult(
                success=False, endpoint="(nemotron)",
                message="I can't reach opencode, so I can't write that with Nemotron right now.",
            )

        def _done(ok, path_or_error, secs):
            if ok:
                msg = f"Your code is ready, sir. It's open in Notepad. Took {secs:.0f} seconds."
            else:
                msg = f"I couldn't finish that code: {path_or_error}"
            logger.info("write_python_code finished: ok=%s (%.0fs)", ok, secs)
            if self._notify:
                self._notify(msg)

        write_code_to_notepad(request, on_done=_done, model=model)
        if model == "gemini":
            note = "Writing that with Gemini now - should be a few seconds."
        else:
            note = ("Writing that with Nemotron - it takes a couple of minutes. "
                    "Notepad will open by itself when it's done.")
        return HardwareResult(
            success=True, endpoint=f"({model})",
            message=note,
            data={"request": request,
                  "model": GEMINI_CODE_MODEL if model == "gemini" else NEMOTRON_MODEL},
        )


CODE_TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "write_python_code",
        "description": (
            "Write a Python script or program and open it in Notepad. Use this for ANY "
            "request to create, write, generate or make code - 'write me a python script "
            "that...', 'create code for...', 'make a program to...'. Defaults to a fast "
            "model (a few seconds) and only escalates to a much slower, stronger one when "
            "the model parameter's own criteria say so. Either way, say it is being "
            "written and will open by itself; do NOT wait for it and do NOT write the "
            "code yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["gemini", "nemotron"],
                    "description": (
                        "Which model writes it. Default 'gemini' - a few seconds, and right "
                        "for an ordinary script or small program. Choose 'nemotron' instead "
                        "ONLY when at least one of these is true: (1) the user explicitly "
                        "names it or asks for the bigger/stronger model (e.g. 'use nemotron', "
                        "'write it properly', 'take your time and do it right'), or (2) the "
                        "request is genuinely large in scope on its own - a full working "
                        "website, a multi-file application, something with a real backend "
                        "and frontend rather than one utility script. Nemotron takes a "
                        "couple of minutes, so do not reach for it unless one of those two "
                        "is actually true - a request just being 'complex-sounding' is not "
                        "enough on its own."
                    ),
                },
                "request": {
                    "type": "string",
                    "description": (
                        "The full description of what the code should do, in the user's own "
                        "terms, including any details they gave about inputs, outputs or flags."
                    ),
                },
            },
            "required": ["request"],
        },
    },
}]
