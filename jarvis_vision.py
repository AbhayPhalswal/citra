"""
=============================================================================
JARVIS VISION — local, on-demand screen understanding + input injection
=============================================================================
Runs on: the same Windows 11 laptop everything else in this project runs on.
Talks to: LM Studio's local OpenAI-compatible server (http://localhost:1234),
          NEVER any cloud API — see PRIVACY BOUNDARY below. That's the whole
          reason this lives in its own file instead of inside
          jarvis_pc_control.py or jarvis_router.py: everything else in this
          project either talks to the OS directly or to Gemini; this is the
          one subsystem that talks to a second, LOCAL LLM server instead,
          and that distinction is exactly the point.

WHY THIS EXISTS
---------------
"Micro-UI control" (moving the cursor, clicking a specific thing, reading
text that's only visible on screen) doesn't work reliably without actually
SEEING the screen — guessing coordinates blind from a text description
isn't good enough. Screenshots are also about the most privacy-sensitive
thing this project could ever transmit (open messages, financial
dashboards, whatever happens to be on screen) — sending them to Gemini was
a hard no. So this is a self-contained local-only pipeline: Gemini
(jarvis_router.py's Smart Path) can CALL analyze_screen() as a tool, but it
only ever sees this function's TEXT return value, never the image bytes.

PRIVACY BOUNDARY — READ THIS BEFORE CHANGING ANYTHING HERE:
analyze_screen() sends the screenshot ONLY to LM Studio at
LM_STUDIO_VISION_BASE_URL (loopback, this machine). It must never be
pointed at anything else. Pixels never leave this machine; a short text
description of them can.

ON-DEMAND LOCAL MODEL — NOTHING NEEDS TO BE PRE-STARTED:
Verified directly against this machine's LM Studio install:
  - `lms server start` (the LM Studio CLI) brings the server up from fully
    stopped in a few seconds, and costs ~0.3s if it's already running — so
    _ensure_lm_studio_server_running() below just calls it unconditionally
    before every request instead of trying to detect whether it's needed.
  - Once the server is up, LM Studio's own just-in-time model loading
    brings the vision model into memory on first use with no separate
    "load" step from this code — measured ~13s cold, then instant while
    warm.
Together, LM Studio can sit completely closed the rest of the time; this
project doesn't need it running continuously the way jarvis_voice_
assistant.py's own process does.

WHY THE SCREENSHOT IS DOWNSCALED BEFORE SENDING:
Measured directly, same question, same machine: a full native-resolution
screenshot (3840x2400 here) took 77.6s for one description. Downscaling to
VISION_MAX_WIDTH=1280 wide first cut that to 24.1s — over 3x faster — with
no visible loss in the model's ability to read on-screen text. That number
is measured, not guessed.

WHY COORDINATES ARE PERCENTAGES, NOT PIXELS:
The model only ever SEES the downscaled screenshot, so any pixel
coordinates it reported would be in the downscaled image's coordinate
space, not the real screen's — correctly translating that back means
carrying an exact scale factor through every single call. Percentages
(0-100 of width/height) sidestep that entirely: click_at() converts
straight to THIS screen's real resolution via pyautogui.size(), with no
scale-factor bookkeeping anywhere, and no dependency on display DPI
scaling matching between the screenshot and the click.
=============================================================================
"""

import base64
import io
import logging
import subprocess
from dataclasses import dataclass, field

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jarvis_vision")

LMS_CLI_PATH = r"C:\Users\HP\.lmstudio\bin\lms.exe"
LM_STUDIO_VISION_BASE_URL = "http://localhost:1234"
LM_STUDIO_VISION_MODEL = "qwen2.5-vl-7b-instruct"  # verified against a
                                    # live GET /v1/models on this machine —
                                    # note the hyphen before "vl". Distinct
                                    # from (and correcting a real, if
                                    # currently unexercised, typo in)
                                    # jarvis_router.py's own LM_STUDIO_MODEL
                                    # constant ("qwen2.5vl-7b-instruct",
                                    # missing that hyphen).
VISION_MAX_WIDTH = 1280  # measured tradeoff — see module docstring
VISION_TIMEOUT_SECONDS = 120.0  # generous on purpose: covers a cold model
                                    # load (~13s measured) plus genuinely
                                    # slow local inference (24-80s measured
                                    # depending on resolution), with real
                                    # headroom rather than a tight bound.

VISION_SYSTEM_PROMPT = (
    "You are looking at a screenshot of the user's computer screen. Answer "
    "the question concisely — a sentence or two, not a lecture. If asked "
    "for the location of something, report it as a PERCENTAGE of the "
    "screen's width and height (0-100 for each), not pixels — e.g. 'the "
    "Save button is at approximately 45% across, 92% down'."
)


@dataclass
class VisionActionResult:
    """Same shape as PCActionResult/HardwareResult — see PCActionResult's
    own docstring for why each subsystem keeps its own copy of this rather
    than sharing one class across unrelated subsystems."""
    success: bool
    action: str
    message: str
    data: dict | None = field(default=None)

    def to_dict(self) -> dict:
        return {"success": self.success, "action": self.action, "message": self.message, "data": self.data}


def _ensure_lm_studio_server_running() -> None:
    """~0.3s if already running (measured), a few seconds cold — cheap
    enough to call before every single request rather than only after a
    detected connection failure. This is what makes analyze_screen()
    genuinely on-demand: nothing else has to remember to start LM Studio
    first."""
    try:
        subprocess.run(
            [LMS_CLI_PATH, "server", "start"],
            capture_output=True, timeout=30, check=False,
        )
    except Exception as exc:
        logger.warning(
            "Could not run `lms server start` (%s) -- proceeding anyway "
            "in case the server is already up.", exc,
        )


def _screenshot_b64(max_width: int = VISION_MAX_WIDTH) -> str:
    from PIL import Image, ImageGrab

    img = ImageGrab.grab()
    if img.width > max_width:
        scale = max_width / img.width
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((max_width, int(img.height * scale)), resample=resample)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VisionController:
    """
    Mirrors PCController's/SmartRoomController's shape (methods return a
    small result dataclass, never raise for an expected failure) so
    jarvis_router.py's tool-dispatch error handling works identically for
    every controller without needing subsystem-specific special cases.
    """

    def analyze_screen(self, question: str) -> VisionActionResult:
        """Screenshots the current screen and asks the LOCAL vision model
        `question` about it. See this module's docstring for the privacy
        boundary this maintains — the image never leaves this machine."""
        question = (question or "").strip() or "Describe what's currently visible on the screen."
        _ensure_lm_studio_server_running()

        try:
            image_b64 = _screenshot_b64()
        except Exception as exc:
            logger.error("Failed to capture screenshot: %s", exc)
            return VisionActionResult(
                success=False, action="analyze_screen",
                message=f"Couldn't capture the screen: {exc}",
            )

        try:
            response = requests.post(
                f"{LM_STUDIO_VISION_BASE_URL}/v1/chat/completions",
                json={
                    "model": LM_STUDIO_VISION_MODEL,
                    "messages": [
                        {"role": "system", "content": VISION_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ]},
                    ],
                    "max_tokens": 200,
                },
                timeout=VISION_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return VisionActionResult(
                    success=False, action="analyze_screen",
                    message="Local vision model returned no answer.",
                )
            answer = (choices[0].get("message", {}).get("content") or "").strip()
            if not answer:
                return VisionActionResult(
                    success=False, action="analyze_screen",
                    message="Local vision model returned an empty answer.",
                )
            logger.info("analyze_screen(%r) -> %r", question, answer)
            return VisionActionResult(success=True, action="analyze_screen", message=answer)
        except requests.exceptions.ConnectionError:
            return VisionActionResult(
                success=False, action="analyze_screen",
                message=(
                    "Couldn't reach the local vision model — LM Studio may "
                    "have failed to start. Try opening LM Studio manually once."
                ),
            )
        except requests.exceptions.Timeout:
            return VisionActionResult(
                success=False, action="analyze_screen",
                message=f"Local vision model took longer than {VISION_TIMEOUT_SECONDS:.0f}s to respond.",
            )
        except Exception as exc:
            logger.error("analyze_screen failed: %s", exc)
            return VisionActionResult(success=False, action="analyze_screen", message=f"Vision analysis failed: {exc}")

    def click_at(self, x_percent: float, y_percent: float, button: str = "left") -> VisionActionResult:
        """Clicks at a position given as a PERCENTAGE (0-100) of screen
        width/height — see this module's docstring for why percentages,
        not pixels."""
        import pyautogui
        try:
            x_percent = max(0.0, min(100.0, float(x_percent)))
            y_percent = max(0.0, min(100.0, float(y_percent)))
            screen_w, screen_h = pyautogui.size()
            x = int(screen_w * x_percent / 100)
            y = int(screen_h * y_percent / 100)
            pyautogui.click(x, y, button=button)
            logger.info("Clicked at (%.0f%%, %.0f%%) -> pixel (%d, %d)", x_percent, y_percent, x, y)
            return VisionActionResult(
                success=True, action="click_at",
                message=f"Clicked at {x_percent:.0f}%, {y_percent:.0f}%.",
                data={"x": x, "y": y},
            )
        except Exception as exc:
            logger.error("click_at failed: %s", exc)
            return VisionActionResult(success=False, action="click_at", message=f"Couldn't click: {exc}")

    def type_text(self, text: str) -> VisionActionResult:
        """Types text at whatever currently has keyboard focus."""
        import pyautogui
        try:
            pyautogui.write(text, interval=0.02)
            logger.info("Typed %d characters.", len(text))
            return VisionActionResult(success=True, action="type_text", message="Typed it.")
        except Exception as exc:
            logger.error("type_text failed: %s", exc)
            return VisionActionResult(success=False, action="type_text", message=f"Couldn't type: {exc}")

    def press_key(self, key: str) -> VisionActionResult:
        """Presses a single key or a '+'-joined hotkey combo, e.g. 'enter',
        'escape', 'ctrl+c'."""
        import pyautogui
        try:
            parts = [p.strip() for p in (key or "").lower().split("+") if p.strip()]
            if not parts:
                return VisionActionResult(success=False, action="press_key", message="No key given.")
            if len(parts) == 1:
                pyautogui.press(parts[0])
            else:
                pyautogui.hotkey(*parts)
            logger.info("Pressed key: %s", key)
            return VisionActionResult(success=True, action="press_key", message=f"Pressed {key}.")
        except Exception as exc:
            logger.error("press_key failed: %s", exc)
            return VisionActionResult(success=False, action="press_key", message=f"Couldn't press '{key}': {exc}")


# -----------------------------------------------------------------------------
# TOOL SCHEMA — same OpenAI-style shape as JARVIS_TOOL_SCHEMA / PC_TOOL_SCHEMA
# -----------------------------------------------------------------------------
VISION_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_screen",
            "description": (
                "Look at the user's screen right now and answer a question about "
                "it, using a LOCAL vision model — the screenshot never leaves this "
                "machine or reaches any cloud service. Use this whenever you "
                "genuinely need to see the screen: finding where to click "
                "something, reading text you have no other way to access, "
                "checking if something finished loading. This can take up to a "
                "couple of minutes (loading the local model the first time is "
                "slow), so only call it when actually necessary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What to look for or ask about the screen, e.g. 'where is the Save button?' or 'what does this error say?'",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_at",
            "description": (
                "Click at a position on screen, given as a PERCENTAGE of screen "
                "width/height (0-100 for each), not pixels. Get the position "
                "first by calling analyze_screen and asking where something is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x_percent": {"type": "number", "description": "Horizontal position, 0-100 (0=left edge, 100=right edge)."},
                    "y_percent": {"type": "number", "description": "Vertical position, 0-100 (0=top edge, 100=bottom edge)."},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Which mouse button. Defaults to left."},
                },
                "required": ["x_percent", "y_percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text at whatever currently has keyboard focus (e.g. into a text field you just clicked).",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text to type."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a single key or key combo, e.g. 'enter', 'escape', 'tab', 'ctrl+c', 'ctrl+s', 'alt+tab'.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "Key name, or '+'-joined combo (e.g. 'ctrl+s')."}},
                "required": ["key"],
            },
        },
    },
]


if __name__ == "__main__":
    import json
    print("\n=== Vision Tool Schema ===\n")
    print(json.dumps(VISION_TOOL_SCHEMA, indent=2))
