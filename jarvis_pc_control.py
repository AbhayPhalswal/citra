"""
=============================================================================
JARVIS PC CONTROL — safe, scoped local-machine actions
=============================================================================
Runs on: the same Windows 11 laptop Citra's voice pipeline runs on.
Talks to: the OS directly (no network, no hardware board involved) — this
          is the local-machine counterpart to jarvis_hardware_api.py's
          smart-room control, kept in its own file for the same reason
          that one is separate from jarvis_router.py: one clear owner per
          subsystem, not because the pattern needs repeating for its own
          sake.

SCOPE
-----
open_application, open_path, lock_screen, and take_screenshot are
deliberately narrow: a known application, an existing file/folder, lock,
screenshot — nothing here can delete, overwrite, or modify anything.

run_command is the deliberate exception: genuine arbitrary command
execution, added because that's specifically what was asked for — full
PC automation, not a curated menu of pre-approved actions. It runs with
exactly the privileges of the account this process runs as (whatever you
could do sitting at the keyboard yourself, nothing more, nothing less).
citra_ui_server.py (and therefore every tool here, this one included) is
loopback-only — see that file's module docstring — so today this is only
ever reachable from this machine: the local voice pipeline, or a browser
on this machine itself. It briefly had a PIN gate for LAN/phone access
too; see that file's git history if that ever comes back.

WHY open_application() USES A WHITELIST, NOT ARBITRARY EXECUTION:
tool_call.function.arguments comes from an LLM's output — treating a
string it produced as a command to execute (subprocess.Popen(name,
shell=True), os.system(name), etc.) is a real command-injection surface,
not a hypothetical one, the same class of risk _dispatch_tool_call's own
tool-name whitelist in jarvis_router.py already guards against for
hardware tools. APP_LAUNCHERS below maps a small set of KNOWN friendly
names to KNOWN safe commands, chosen at review time by a human, not
constructed at runtime from model output. A name that isn't in the map
fails with a clear message instead of being executed speculatively —
extend the map (it's a plain dict) once you know what else you actually
want Citra to open, rather than trusting whatever a model guesses.

WHY open_path() REQUIRES THE PATH TO ALREADY EXIST:
Same reasoning, applied to files/folders instead of app names. Checking
os.path.exists() before calling os.startfile() means a malformed or
unexpected string from the LLM (a URL, a shell metacharacter sequence, a
path that doesn't correspond to anything real) fails a validation check
with a clear reason, rather than being handed to the OS shell handler on
the hope that it's harmless.
=============================================================================
"""

import ctypes
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("jarvis_pc_control")

# Marks THIS process as per-monitor-DPI-aware, once, at import time.
# play_apple_music needs this: without it, Windows silently reports
# OTHER processes' window/element coordinates to us in a virtualized,
# DPI-scaled space instead of real pixels -- verified live, on this
# machine's 250% scaling, Apple Music's own window rect read as
# (0, 0, 1536, 960) instead of its real (0, 0, 3840, 2400) until this
# call was made. pyautogui's click() targets REAL physical pixels
# regardless, so without this fix, UI Automation's reported coordinates
# and pyautogui's click coordinates silently disagree. Wrapped in
# try/except because a process's DPI awareness can only be set ONCE --
# calling this a second time (e.g. if something else already set it)
# raises rather than being a no-op, and that's fine to ignore here.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    pass


# -----------------------------------------------------------------------------
# RESULT OBJECT
# -----------------------------------------------------------------------------
@dataclass
class PCActionResult:
    """
    Same shape as jarvis_hardware_api.HardwareResult (success/message/data),
    kept as its own small type rather than importing that one directly —
    this subsystem isn't hardware, and giving it its own result type keeps
    that naming honest rather than borrowing a class whose name says
    otherwise. The LLM tool-calling loop only cares about the shape
    (to_dict()'s JSON), not which Python class produced it.
    """
    success: bool
    action: str
    message: str
    data: Optional[dict] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "action": self.action,
            "message": self.message,
            "data": self.data,
        }


# -----------------------------------------------------------------------------
# APPLICATION WHITELIST
# -----------------------------------------------------------------------------
# Friendly name -> a callable that launches it. Using callables (not just
# bare executable-name strings) so a few entries can do their own
# candidate-path probing (e.g. the browser, which isn't reliably on PATH)
# while the simple ones stay a one-liner.
def _launch_exe(exe_name: str):
    def _launch():
        subprocess.Popen([exe_name])
    return _launch


def _launch_first_existing(candidate_paths: list):
    """For apps without a guaranteed PATH entry (e.g. Chrome) — tries each
    known real install location in order, using the first that actually
    exists rather than guessing blindly."""
    def _launch():
        for path in candidate_paths:
            if os.path.exists(path):
                subprocess.Popen([path])
                return
        raise FileNotFoundError(
            f"None of the expected install locations exist: {candidate_paths}"
        )
    return _launch


def _launch_uri(uri: str):
    def _launch():
        os.startfile(uri)
    return _launch


# Only confirmed Windows-builtin utilities (always present, no install-path
# guessing needed) plus the two browsers already confirmed installed on
# this machine. Add more entries here once you know what else you want
# Citra to be able to open — see this module's docstring for why this is
# a deliberate whitelist, not a fallback-to-anything launcher.
APP_LAUNCHERS = {
    "notepad": _launch_exe("notepad.exe"),
    "calculator": _launch_exe("calc.exe"),
    "file explorer": _launch_exe("explorer.exe"),
    "explorer": _launch_exe("explorer.exe"),
    "paint": _launch_exe("mspaint.exe"),
    "task manager": _launch_exe("taskmgr.exe"),
    "control panel": _launch_exe("control.exe"),
    "command prompt": _launch_exe("cmd.exe"),
    "settings": _launch_uri("ms-settings:"),
    "camera": _launch_uri("microsoft.windows.camera:"),  # the built-in
                                    # Camera app is a UWP app with no
                                    # camera.exe to launch directly — this
                                    # is its registered URI protocol,
                                    # same mechanism as ms-settings: above.
                                    # Verified live: launches the real
                                    # Camera app on this machine.
    "chrome": _launch_first_existing([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]),
    "browser": _launch_first_existing([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]),
    "edge": _launch_first_existing([
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]),
    "apple music": _launch_uri("music:"),  # the Windows Store app
                                    # (AppleInc.AppleMusicWin) registers
                                    # "music:" as its own URI protocol —
                                    # verified live: confirmed installed
                                    # via Get-AppxPackage, and os.startfile
                                    # on this URI brings up the real app,
                                    # already signed in, same as clicking
                                    # its Start Menu tile.
}

# Common folder name -> real path on THIS machine, resolved at call time via
# os.path.expanduser("~") rather than hardcoded. Exists because of a real
# failure, not a hypothetical: asked to "open my downloads folder", Gemini
# had no way to know the actual Windows account name and guessed
# C:\Users\User\Downloads — a plausible-looking path that doesn't exist on
# this machine (the real one is C:\Users\HP\Downloads). open_path() checked
# existence first (see its own docstring), so this failed safely rather
# than silently doing nothing or erroring oddly — but "open my downloads
# folder" is exactly the kind of ordinary request this feature needs to
# actually handle, not just fail safely on. Resolving well-known folder
# NAMES locally, instead of asking the model to construct a full path it
# fundamentally cannot know, fixes that at the source.
_SPECIAL_FOLDERS = {
    "downloads": lambda: os.path.join(os.path.expanduser("~"), "Downloads"),
    "documents": lambda: os.path.join(os.path.expanduser("~"), "Documents"),
    "desktop": lambda: os.path.join(os.path.expanduser("~"), "Desktop"),
    "pictures": lambda: os.path.join(os.path.expanduser("~"), "Pictures"),
    "music": lambda: os.path.join(os.path.expanduser("~"), "Music"),
    "videos": lambda: os.path.join(os.path.expanduser("~"), "Videos"),
    "home": lambda: os.path.expanduser("~"),
}

SCREENSHOT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "CitraScreenshots")


# -----------------------------------------------------------------------------
# HOSTS-FILE LINE TRANSFORMS — pure functions, no file I/O
# -----------------------------------------------------------------------------
# Kept as standalone functions operating on a list of lines (not methods that
# read/write HOSTS_FILE_PATH directly) specifically so the actual blocking
# LOGIC can be unit-tested against an in-memory list, with zero risk of ever
# touching the real system hosts file during development or testing.
def _hosts_block_lines(domain: str) -> list:
    return [
        f"127.0.0.1 {domain}  {HOSTS_BLOCK_MARKER}\n",
        f"127.0.0.1 www.{domain}  {HOSTS_BLOCK_MARKER}\n",
    ]


def _add_domain_block(lines: list, domain: str) -> list:
    """Returns a NEW list with `domain` blocked, unless it's already
    blocked (idempotent — calling this twice doesn't duplicate entries)."""
    existing = "".join(lines)
    if HOSTS_BLOCK_MARKER in existing and domain in existing:
        return list(lines)  # already blocked, no change
    new_lines = list(lines)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    new_lines.extend(_hosts_block_lines(domain))
    return new_lines


def _remove_domain_block(lines: list, domain: str) -> list:
    """Returns a NEW list with every citra-managed block line for `domain`
    removed. Only ever touches lines carrying HOSTS_BLOCK_MARKER — a line
    blocking the same domain that a human added by hand is left alone."""
    return [
        line for line in lines
        if not (HOSTS_BLOCK_MARKER in line and domain in line)
    ]

# run_command's bounds: a hang here would otherwise block the whole Smart
# Path turn (the LLM is waiting on this tool's result before it can
# answer), and an unbounded reply could blow well past
# SMART_PATH_MAX_TOKENS's spoken-answer budget in jarvis_router.py once
# the model tries to summarize it.
RUN_COMMAND_TIMEOUT_SECONDS = 20.0
RUN_COMMAND_MAX_OUTPUT_CHARS = 2000

# Named app-launch presets — "open my X setup" opens each app in the list,
# in order. Same reasoning as APP_LAUNCHERS above: a small, explicit,
# human-editable map rather than asking the model to invent one, so a
# preset does exactly what you defined it to do, every time. These two are
# just a starting example — add your own (extend the dict, same as every
# other whitelist in this file).
LAYOUT_PRESETS = {
    "writing setup": ["notepad", "chrome"],
    "coding setup": ["command prompt", "file explorer"],
}

APPLE_MUSIC_LAUNCH_WAIT_SECONDS = 3.5  # empirically enough for a cold
                                    # launch of the app (verified live);
                                    # a fixed sleep rather than polling
                                    # for its window, matching this file's
                                    # existing level of simplicity (see
                                    # open_application, which doesn't wait
                                    # for anything either) rather than
                                    # adding a new win32-window-polling
                                    # dependency for one caller.
APPLE_MUSIC_SEARCH_SETTLE_SECONDS = 1.5  # verified live: enough for the
                                    # search results page to actually
                                    # render before UI Automation looks
                                    # for a result on it.
APPLE_MUSIC_DETAIL_PAGE_WAIT_SECONDS = 1.2  # same idea, for the song/
                                    # album page that opening the top
                                    # search result navigates to.
APPLE_MUSIC_MAX_TAB_ATTEMPTS = 30  # generous bound on play_apple_music's
                                    # Tab-and-check loop -- see that
                                    # method's comment for why a fixed
                                    # number of presses can't be assumed.

WINGET_TIMEOUT_SECONDS = 180.0  # installers genuinely take minutes
                                    # (download + install), a much longer
                                    # ceiling than RUN_COMMAND_TIMEOUT_SECONDS
                                    # on purpose — this is real network I/O
                                    # plus disk writes, not a quick local
                                    # command.

FILE_SEARCH_ROOTS = [os.path.expanduser("~")]  # scoped to the user's home
                                    # directory tree, not the whole C:
                                    # drive — a genuinely whole-drive
                                    # search needs the Windows Search
                                    # index (COM interop) to be fast at
                                    # all; a plain recursive walk of C:\
                                    # would be minutes-slow and mostly
                                    # irrelevant system files. Home
                                    # directory covers the realistic "find
                                    # my file" case.
FILE_SEARCH_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "jarvis_venv", ".venv",
    "venv", "$RECYCLE.BIN", "AppData",
}  # directories that are either huge, irrelevant to "find my file", or
   # both — skipped during the walk rather than filtered after, since
   # os.walk lets a directory be pruned before descending into it.
FILE_SEARCH_MAX_RESULTS = 20
FILE_SEARCH_TIMEOUT_SECONDS = 15.0

HOSTS_FILE_PATH = r"C:\Windows\System32\drivers\etc\hosts"
HOSTS_BLOCK_MARKER = "# citra-managed-block"  # appended to every line this
                                    # tool adds, so unblock_domain (and any
                                    # future "list what's blocked") can
                                    # find and remove exactly its own
                                    # entries without touching lines a
                                    # human (or another program) added to
                                    # this file by hand.


# -----------------------------------------------------------------------------
# AUDIO OUTPUT DEVICE CONTROL
# -----------------------------------------------------------------------------
LAPTOP_SPEAKERS_FRIENDLY_NAME = "Speakers (Realtek(R) Audio)"  # same physical
                                    # device as jarvis_voice_assistant.py's
                                    # LAPTOP_SPEAKERS_DEVICE_NAME (that file
                                    # matches it through sounddevice, this
                                    # one through pycaw/Core Audio directly
                                    # -- the two libraries report the exact
                                    # same friendly name for it, verified
                                    # live, so this is intentionally kept as
                                    # a second constant rather than an
                                    # import between the two files). Update
                                    # BOTH constants together if the audio
                                    # hardware ever changes.

# Lets play_apple_music (below) ask "where do you want to play this" and
# actually act on the answer by switching WINDOWS' DEFAULT PLAYBACK DEVICE --
# Apple Music (like most apps) just plays through whatever that is, so
# changing it before opening/playing is all that's needed, with no
# per-application audio routing trickery required.
#
# Deliberately does NOT touch Citra's own voice: jarvis_voice_assistant.py
# pins Citra's TTS to LAPTOP_SPEAKERS_DEVICE_NAME via an explicit `device=`
# argument to every sounddevice.OutputStream() call, bypassing "whatever the
# system default is" entirely — see that file's own comment. So switching
# the system default for music's sake here can never route Citra's own
# speech somewhere unexpected, regardless of which device the user picks.
#
# WHY IPolicyConfig (an UNDOCUMENTED COM interface) INSTEAD OF A PUBLIC API:
# Windows has never shipped a public, documented way to change the default
# audio *playback* device — only to read the current one. IPolicyConfig is
# the interface Windows' own Sound Control Panel calls internally to do
# exactly this; it's undocumented by Microsoft but has stayed
# ABI-compatible since Windows 7 (verified live on this Windows 11 machine)
# and is the same interface widely-used tools (NirSoft's SoundVolumeView,
# the AudioDeviceCmdlets PowerShell module, EarTrumpet) all rely on for the
# same reason -- there simply isn't another way. GUIDs and method ordering
# below are the well-established layout from the community-maintained
# reference at https://github.com/tartakynov/audioswitch/blob/master/
# IPolicyConfig.h, cross-checked against this project's own live testing
# (verified: switching the default device and back, confirmed via
# AudioUtilities.GetSpeakers() before/after).
#
# Methods 0-9 below (GetMixFormat through SetPropertyValue) are declared
# ONLY to occupy the correct VTABLE SLOTS ahead of SetDefaultEndpoint --
# comtypes calls a COM method by its position in this list, so
# SetDefaultEndpoint has to be preceded by exactly as many entries as it
# actually has ahead of it in the real interface, but since nothing here
# ever CALLS those first 10, their argument types only need to be
# ctypes-legal, not perfectly accurate to the real (partially undocumented)
# signatures.
_POLICY_CONFIG_CLSID = "{870af99c-171d-4f9e-af0d-e63df40c2bc9}"
_POLICY_CONFIG_IID = "{f8679f50-850a-41cf-9c72-430f290290c8}"


def _make_policy_config_interface():
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT
    from ctypes import POINTER, c_void_p, c_wchar_p
    from ctypes.wintypes import DWORD

    class IPolicyConfig(comtypes.IUnknown):
        _iid_ = GUID(_POLICY_CONFIG_IID)
        _methods_ = [
            COMMETHOD([], HRESULT, "GetMixFormat",
                      (['in'], c_wchar_p, "device_id"),
                      (['out'], POINTER(c_void_p), "format")),
            COMMETHOD([], HRESULT, "GetDeviceFormat",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], c_void_p, "default"),
                      (['out'], POINTER(c_void_p), "format")),
            COMMETHOD([], HRESULT, "ResetDeviceFormat",
                      (['in'], c_wchar_p, "device_id")),
            COMMETHOD([], HRESULT, "SetDeviceFormat",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], c_void_p, "endpoint_format"),
                      (['in'], c_void_p, "mix_format")),
            COMMETHOD([], HRESULT, "GetProcessingPeriod",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], c_void_p, "default"),
                      (['out'], c_void_p, "default_period"),
                      (['out'], c_void_p, "minimum_period")),
            COMMETHOD([], HRESULT, "SetProcessingPeriod",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], c_void_p, "period")),
            COMMETHOD([], HRESULT, "GetShareMode",
                      (['in'], c_wchar_p, "device_id"),
                      (['out'], c_void_p, "mode")),
            COMMETHOD([], HRESULT, "SetShareMode",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], c_void_p, "mode")),
            COMMETHOD([], HRESULT, "GetPropertyValue",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], c_void_p, "key"),
                      (['out'], c_void_p, "value")),
            COMMETHOD([], HRESULT, "SetPropertyValue",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], c_void_p, "key"),
                      (['in'], c_void_p, "value")),
            COMMETHOD([], HRESULT, "SetDefaultEndpoint",
                      (['in'], c_wchar_p, "device_id"),
                      (['in'], DWORD, "role")),
        ]

    return IPolicyConfig


_STEAM_VIRTUAL_DEVICE_MARKER = "steam streaming"  # excluded from the
                                    # device list offered to the user --
                                    # Steam's in-home-streaming feature
                                    # registers virtual playback devices
                                    # that aren't real, physically
                                    # present speakers, so offering them
                                    # as a place to "play music" would be
                                    # confusing (verified live: this
                                    # machine has "Speakers (Steam
                                    # Streaming Speakers)" and "Speakers
                                    # (Steam Streaming Microphone)"
                                    # permanently listed as Active
                                    # regardless of whether Steam is even
                                    # running).


def _list_active_playback_devices():
    """
    Returns [(friendly_name, device_id), ...] for every currently-ACTIVE
    playback device (real, physically present/connected right now --
    see AudioDeviceState.Active), excluding the built-in laptop speakers
    themselves and Steam's virtual devices. In practice, on a laptop with
    no other real output hardware, whatever's left here is exactly "the
    Bluetooth (or wired) device connected right now" -- which is what
    play_apple_music uses to decide whether there's an actual CHOICE to
    ask the user about.
    """
    from pycaw.pycaw import AudioUtilities, EDataFlow

    devices = []
    for device in AudioUtilities.GetAllDevices(data_flow=EDataFlow.eRender.value):
        name = device.FriendlyName
        if not name or str(device.state) != "AudioDeviceState.Active":
            continue
        if name == LAPTOP_SPEAKERS_FRIENDLY_NAME:
            continue
        if _STEAM_VIRTUAL_DEVICE_MARKER in name.lower():
            continue
        devices.append((name, device.id))
    return devices


def _find_playback_device_id(name_query: str):
    """Finds an active playback device's id by a partial, case-insensitive
    name match against either the laptop speakers or any other currently
    active device -- e.g. 'airpods' matching 'Headphones (Manu's AirPods
    Pro - Find My)'. Returns None if nothing matches."""
    from pycaw.pycaw import AudioUtilities, EDataFlow

    name_query = name_query.strip().lower()
    if name_query in ("laptop speakers", "laptop", "speakers", "built-in", "built-in speakers"):
        for device in AudioUtilities.GetAllDevices(data_flow=EDataFlow.eRender.value):
            if device.FriendlyName == LAPTOP_SPEAKERS_FRIENDLY_NAME:
                return device.id
        return None
    for name, device_id in _list_active_playback_devices():
        if name_query in name.lower():
            return device_id
    return None


def _set_default_playback_device(device_id: str) -> bool:
    """Switches Windows' system default playback device -- see this
    section's module comment for why (IPolicyConfig, no public
    alternative exists) and how this stays scoped to music/general
    playback only, never Citra's own pinned-device voice. Sets all
    three roles (console, multimedia, communications) since different
    apps query different ones for "the default device" -- verified live
    that setting only one left some apps still on the old device."""
    import comtypes

    IPolicyConfig = _make_policy_config_interface()
    comtypes.CoInitialize()
    try:
        policy_config = comtypes.CoCreateInstance(
            comtypes.GUID(_POLICY_CONFIG_CLSID), IPolicyConfig, comtypes.CLSCTX_ALL,
        )
        success = True
        for role in (0, 1, 2):  # eConsole, eMultimedia, eCommunications
            hr = policy_config.SetDefaultEndpoint(device_id, role)
            if hr != 0:
                success = False
                logger.error("SetDefaultEndpoint(role=%d) failed: hr=0x%08x", role, hr & 0xFFFFFFFF)
        return success
    except Exception as exc:
        logger.error("Couldn't set default playback device: %s", exc)
        return False
    finally:
        comtypes.CoUninitialize()


# play_apple_music's targeting: Windows UI Automation, not the local
# vision model. A vision-model version was tried first and abandoned --
# verified live, qwen2.5-vl-7b-instruct repeatedly answered "50% across"
# for a search-result card regardless of where it actually was in a
# multi-column grid, close enough to sound plausible but wrong often
# enough to click blank space. UI Automation instead asks the OS
# directly for an element's real on-screen rectangle -- exact, and much
# faster since it skips a round trip to LM Studio entirely.
def _find_process_window(exe_name: str):
    """Returns the hwnd of the first visible top-level window belonging
    to a process named `exe_name` (e.g. "AppleMusic.exe"), or None.
    Matched by PROCESS NAME, not window title -- verified live that
    Apple Music's title bar just says "Apple Music" whether idle,
    browsing search results, or mid-playback, so the title carries no
    useful signal, while the owning process's name is stable."""
    import psutil
    import win32gui
    import win32process

    matches = []

    def _callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or not win32gui.GetWindowText(hwnd):
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if psutil.Process(pid).name().lower() == exe_name.lower():
                matches.append(hwnd)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    win32gui.EnumWindows(_callback, None)
    return matches[0] if matches else None


def _focus_window(hwnd) -> None:
    """
    Brings a window to the foreground and maximizes it, so keystrokes
    actually reach it. Plain SetForegroundWindow can silently fail if
    the calling process isn't already the foreground process (a Windows
    restriction meant to stop background apps from stealing focus) --
    tapping Alt first is the standard workaround, since Windows grants
    the focus change to whichever process most recently saw a real
    input event. Verified live: without this, a freshly launched Apple
    Music window can be fully visible and NOT actually hold keyboard
    focus, so typed keystrokes silently go to whatever else had focus
    instead.

    Also maximizes rather than just restoring -- both because that's
    what was actually asked for, and because it removes any ambiguity
    about the window's on-screen position while reasoning about clicks.

    Retries the activation up to 5 times (checking GetForegroundWindow()
    actually changed each time) rather than firing it once and trusting
    it worked -- verified live that a single attempt can silently fail
    to take effect even with the Alt-tap workaround.
    """
    import win32api
    import win32con
    import win32gui

    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    for _attempt in range(5):
        if win32gui.GetForegroundWindow() == hwnd:
            return
        win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        finally:
            win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.3)


def _uia_center(element) -> Optional[tuple]:
    """Returns the (x, y) screen-pixel center of a UI Automation
    element's bounding rectangle, or None if it has no real on-screen
    presence (e.g. nothing is actually focused)."""
    rect = element.CurrentBoundingRectangle
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return None
    return ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)


def _find_content_play_button(uia, root):
    """
    Finds Apple Music's big 'Play' button for whatever page is currently
    showing (song, album, playlist, or artist) -- NOT the small 'Play'
    button that's always sitting in the top transport bar. Verified
    live: both exist at the same time and share the exact same
    accessible name ("Play"), distinguished only by position -- the
    transport bar one stays within the top ~100px, the content one is
    always further down the page. Picking whichever match has the
    largest top-edge coordinate reliably picks the content one.
    """
    from comtypes.gen import UIAutomationClient as UIA

    condition = uia.CreatePropertyCondition(UIA.UIA_ControlTypePropertyId, UIA.UIA_ButtonControlTypeId)
    buttons = root.FindAll(UIA.TreeScope_Descendants, condition)
    best = None
    best_top = -1
    for i in range(buttons.Length):
        button = buttons.GetElement(i)
        if (button.CurrentName or "").strip().lower() != "play":
            continue
        rect = button.CurrentBoundingRectangle
        if rect.bottom <= rect.top:
            continue
        if rect.top > best_top:
            best_top = rect.top
            best = button
    return best


def _is_playing(uia, root) -> bool:
    """Checks whether Apple Music is actually playing something, via the
    transport bar's own Play/Pause button -- its accessible name flips
    from 'Play' to 'Pause' once playback genuinely starts, not just once
    something's been clicked. play_apple_music uses this to verify its
    final click actually worked instead of trusting that a click landing
    on the right element's coordinates necessarily started playback."""
    from comtypes.gen import UIAutomationClient as UIA

    condition = uia.CreatePropertyCondition(UIA.UIA_ControlTypePropertyId, UIA.UIA_ButtonControlTypeId)
    buttons = root.FindAll(UIA.TreeScope_Descendants, condition)
    best_name = None
    best_top = None
    for i in range(buttons.Length):
        button = buttons.GetElement(i)
        name = (button.CurrentName or "").strip().lower()
        if name not in ("play", "pause"):
            continue
        rect = button.CurrentBoundingRectangle
        if rect.bottom <= rect.top:
            continue
        # The TRANSPORT BAR's Play/Pause button, not the content page's
        # -- the smallest top-edge match, same logic as
        # _find_content_play_button but inverted.
        if best_top is None or rect.top < best_top:
            best_top = rect.top
            best_name = name
    return best_name == "pause"


def _find_track_row(uia, root, query: str):
    """
    Finds a specific track's row in an album/playlist tracklist, if the
    top search result opened onto one instead of a standalone song page.

    Verified live: when a search's top result is a "Song" that doesn't
    have its own standalone single release, opening it routes to the
    song's PARENT ALBUM page instead (with that one track visibly
    marked), and _find_content_play_button's big "Play" button there
    plays the album from track 1 -- not the song that was actually
    searched for. Each tracklist row's accessible name is "Track N
    <title> <duration>", and -- unlike the search-results grid cards,
    which only report a real bounding rectangle once focused, see
    play_apple_music's docstring -- these rows report one regardless of
    focus, so this can search for the matching row directly.
    """
    from comtypes.gen import UIAutomationClient as UIA

    words = [w.lower() for w in query.split() if len(w) > 3]
    if not words:
        return None
    condition = uia.CreateTrueCondition()
    elements = root.FindAll(UIA.TreeScope_Descendants, condition)
    for i in range(elements.Length):
        element = elements.GetElement(i)
        name = (element.CurrentName or "").lower()
        if not name.startswith("track "):
            continue
        # ALL significant words must match, not just any one -- verified
        # live that "any" picks the wrong track when two on the same
        # album share a word (e.g. searching "Uptown Funk Bruno Mars"
        # matched on "uptown" alone against track 1, "Uptown's First
        # Finale", before track 4, the actual "Uptown Funk", was ever
        # reached).
        if not all(word in name for word in words):
            continue
        rect = element.CurrentBoundingRectangle
        if rect.right > rect.left and rect.bottom > rect.top:
            return element
    return None


_APPLE_MUSIC_NAV_LANDMARKS = {
    "home", "open navigation", "click to search", "new", "radio",
    "library", "playlists",
}  # accessible names Tab can land on when it did NOT reach a real search
   # result -- e.g. because Enter silently no-op'd (see play_apple_music's
   # docstring) and focus is still sitting on a nav element instead.


def _looks_like_search_result(element, query: str) -> bool:
    """
    Heuristic check that Tab actually landed on a real search-result
    card and not some unrelated element -- used to detect a real,
    observed race: on a cold Apple Music launch, pressing Enter in the
    search box can silently no-op if it lands before the app's search
    subsystem has finished initializing, even though the window itself
    is already fully drawn and accepting other keystrokes. When that
    happens, Tab lands on whatever nav element already had focus instead
    of a result card, and play_apple_music retries the search once.

    A genuine top-result card's accessible name mentions at least one
    real word from the query (e.g. "Queen" for "Bohemian Rhapsody
    Queen").
    """
    name = (element.CurrentName or "").strip().lower()
    if not name or name in _APPLE_MUSIC_NAV_LANDMARKS:
        return False
    words = [w.lower() for w in query.split() if len(w) > 3]
    return any(word in name for word in words) if words else bool(name)


def _search_apple_music_for(query: str) -> None:
    """Focuses Apple Music's search box (Ctrl+F) and submits `query`."""
    import pyautogui

    pyautogui.hotkey("ctrl", "f")
    time.sleep(0.4)
    # Ctrl+F only FOCUSES the search box, it doesn't select any text
    # already sitting in it -- without this Ctrl+A, a leftover query
    # from a previous search gets the new one appended onto it instead
    # of replaced. Verified live: without it, searching twice in a row
    # produced the literal query "blinding lightsBlinding Lights The
    # Weeknd".
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyautogui.write(query, interval=0.02)
    time.sleep(0.3)
    pyautogui.press("enter")
    time.sleep(APPLE_MUSIC_SEARCH_SETTLE_SECONDS)


# -----------------------------------------------------------------------------
# PC CONTROLLER
# -----------------------------------------------------------------------------
class PCController:
    """
    One instance manages all local-machine actions. Mirrors
    SmartRoomController's shape deliberately (methods return a small
    result dataclass, never raise for an expected failure) so
    jarvis_router.py's tool-dispatch error handling works identically
    for both controllers without needing special cases per subsystem.
    """

    def open_application(self, name: str) -> PCActionResult:
        key = name.strip().lower()
        launcher = APP_LAUNCHERS.get(key)
        if launcher is None:
            available = ", ".join(sorted(set(APP_LAUNCHERS.keys())))
            return PCActionResult(
                success=False, action="open_application",
                message=(
                    f"I don't know how to open '{name}' yet. I can currently "
                    f"open: {available}."
                ),
            )
        try:
            launcher()
            logger.info("Opened application: %s", name)
            return PCActionResult(
                success=True, action="open_application",
                message=f"Opened {name}.",
            )
        except Exception as exc:
            logger.error("Failed to open application %r: %s", name, exc)
            return PCActionResult(
                success=False, action="open_application",
                message=f"Found '{name}' but couldn't launch it: {exc}",
            )

    def open_path(self, path: str) -> PCActionResult:
        # A well-known folder NAME ("downloads") resolves to this actual
        # machine's real path — see _SPECIAL_FOLDERS's comment for why
        # this exists: the model has no way to know the real Windows
        # account name, so asking it for a full literal path there was a
        # real, observed failure, not a hypothetical one.
        key = path.strip().lower()
        if key in _SPECIAL_FOLDERS:
            path = _SPECIAL_FOLDERS[key]()

        # See this module's docstring for why existence is checked BEFORE
        # calling os.startfile — a real, on-disk path is the actual safety
        # boundary here, not just a nicer error message.
        if not os.path.exists(path):
            return PCActionResult(
                success=False, action="open_path",
                message=f"'{path}' doesn't exist on this computer, sir.",
            )
        try:
            os.startfile(path)
            is_dir = os.path.isdir(path)
            logger.info("Opened path: %s", path)
            return PCActionResult(
                success=True, action="open_path",
                message=f"Opened the {'folder' if is_dir else 'file'}.",
                data={"path": path, "is_directory": is_dir},
            )
        except OSError as exc:
            logger.error("Failed to open path %r: %s", path, exc)
            return PCActionResult(
                success=False, action="open_path",
                message=f"Couldn't open '{path}': {exc}",
            )

    def lock_screen(self) -> PCActionResult:
        # user32!LockWorkStation is the same call the Win+L shortcut and
        # the Start menu's "Lock" option use — fully standard, and
        # unlocking still needs the account's own password, so there's no
        # way this leaves the machine any less secure than locking it by
        # hand would.
        try:
            result = ctypes.windll.user32.LockWorkStation()
            if not result:
                raise ctypes.WinError()
            logger.info("Locked the screen.")
            return PCActionResult(
                success=True, action="lock_screen",
                message="Locked, sir.",
            )
        except Exception as exc:
            logger.error("Failed to lock screen: %s", exc)
            return PCActionResult(
                success=False, action="lock_screen",
                message=f"Couldn't lock the screen: {exc}",
            )

    def take_screenshot(self) -> PCActionResult:
        try:
            # Imported here, not at module level, so importing this file
            # for its dataclasses/whitelist doesn't require Pillow to
            # already be installed — same lazy-import reasoning
            # SemanticRouter uses for sentence-transformers.
            from PIL import ImageGrab

            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            filename = f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            full_path = os.path.join(SCREENSHOT_DIR, filename)
            ImageGrab.grab().save(full_path)
            logger.info("Saved screenshot: %s", full_path)
            return PCActionResult(
                success=True, action="take_screenshot",
                message="Got it — screenshot saved.",
                data={"path": full_path},
            )
        except ImportError:
            return PCActionResult(
                success=False, action="take_screenshot",
                message="Screenshot support needs Pillow installed (pip install Pillow).",
            )
        except Exception as exc:
            logger.error("Failed to take screenshot: %s", exc)
            return PCActionResult(
                success=False, action="take_screenshot",
                message=f"Couldn't take a screenshot: {exc}",
            )

    def run_command(self, command: str) -> PCActionResult:
        """
        Runs `command` through the Windows shell and returns its output.

        shell=True IS the point here, not an oversight — see this module's
        docstring for why this method exists as the deliberate exception
        to every other method's whitelist-only approach above. Every
        invocation is logged (command, exit code) so there's a record of
        what Citra actually ran, since this is real, unscoped command
        execution with no undo.
        """
        command = command.strip()
        if not command:
            return PCActionResult(
                success=False, action="run_command",
                message="No command given.",
            )
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=RUN_COMMAND_TIMEOUT_SECONDS,
                cwd=os.path.expanduser("~"),
            )
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            if len(output) > RUN_COMMAND_MAX_OUTPUT_CHARS:
                output = output[:RUN_COMMAND_MAX_OUTPUT_CHARS] + "\n... (truncated)"
            logger.info("Ran command: %r (exit code %d)", command, result.returncode)
            return PCActionResult(
                success=(result.returncode == 0),
                action="run_command",
                message=output or f"Command finished with exit code {result.returncode}.",
                data={"exit_code": result.returncode, "command": command},
            )
        except subprocess.TimeoutExpired:
            logger.error("Command timed out after %.0fs: %r", RUN_COMMAND_TIMEOUT_SECONDS, command)
            return PCActionResult(
                success=False, action="run_command",
                message=f"Command timed out after {RUN_COMMAND_TIMEOUT_SECONDS:.0f}s: {command}",
            )
        except Exception as exc:
            logger.error("Command failed to start: %r: %s", command, exc)
            return PCActionResult(
                success=False, action="run_command",
                message=f"Couldn't run '{command}': {exc}",
            )

    def open_layout(self, name: str) -> PCActionResult:
        """Opens every app in a named LAYOUT_PRESETS entry, in order."""
        key = name.strip().lower()
        apps = LAYOUT_PRESETS.get(key)
        if apps is None:
            available = ", ".join(sorted(LAYOUT_PRESETS.keys())) or "(none defined yet)"
            return PCActionResult(
                success=False, action="open_layout",
                message=f"I don't have a '{name}' layout. Defined layouts: {available}.",
            )
        failures = []
        for app in apps:
            result = self.open_application(app)
            if not result.success:
                failures.append(result.message)
        if failures:
            return PCActionResult(
                success=False, action="open_layout",
                message=f"Opened the '{name}' layout, but with problems: {'; '.join(failures)}",
            )
        logger.info("Opened layout '%s': %s", name, apps)
        return PCActionResult(
            success=True, action="open_layout",
            message=f"Opened your {name}.",
            data={"apps": apps},
        )

    def play_apple_music(self, query: str, device: str = None) -> PCActionResult:
        """
        Opens Apple Music, searches for `query`, and plays the top result
        -- one tool call end to end, not a multi-turn chain.

        DEVICE ROUTING: if `device` is omitted and another real playback
        device (e.g. Bluetooth headphones/speaker) is currently connected
        alongside the laptop's own speakers, this does NOT guess which
        one to use -- it returns a question (as a successful result, so
        the model just speaks it) listing the actual available devices
        by name, and expects a follow-up call with `device` set to
        whichever one the user picked. When `device` IS given (or there
        was never a choice to make -- only the laptop speakers are
        available), it switches Windows' system default playback device
        to match (see _set_default_playback_device's own comment for why
        that's the right lever) before doing anything else. This never
        touches Citra's OWN voice, which is pinned to the laptop speakers
        independently of the system default -- see
        jarvis_voice_assistant.py's LAPTOP_SPEAKERS_DEVICE_NAME.

        Composes things that already exist separately (launching the app,
        injecting keystrokes, and Windows UI Automation's own screen
        introspection) INSIDE this one method instead of leaving
        jarvis_router.py's Smart Path to chain the individual steps
        itself, turn by turn: that router caps a single request at
        MAX_TOOL_CALL_ROUNDS=3 round-trips (see its own comment for why),
        and this needs more steps than that (search, open the top result,
        find its Play button, click it) with none left over for the model
        to actually confirm back to the user afterward. Doing all of it
        here spends exactly one of those rounds instead.

        query is typed into Apple Music's own search box (Ctrl+F, same as
        a human would), not embedded in a URL -- verified live that the
        app's "music:" URI protocol accepts a ?term= query param but
        silently ignores it, landing on the plain search tab instead of
        real results.

        TARGETING: Windows UI Automation (comtypes), not the local vision
        model. A vision-model version was tried first and abandoned --
        verified live, the local model kept answering "50% across" for a
        search-result card regardless of where it actually was in a
        multi-column grid: plausible-sounding, wrong often enough to
        click blank space. UI Automation asks the OS directly for an
        element's real on-screen rectangle instead, which is exact, and
        far faster since there's no round trip to LM Studio at all. See
        _find_process_window/_find_content_play_button's own docstrings
        for the two more specific things verified live along the way:
        Tab reliably focuses the top search-result card, and that card's
        own click target does NOT start playback -- it navigates to that
        song/album's page, which always has one big "Play" button.

        FOCUS AND DPI: two more real, observed failure modes, both fixed
        before any keystrokes go out. (1) A freshly launched window can
        be fully visible without actually holding keyboard focus, so
        typed keystrokes silently go wherever focus really was instead
        -- see _focus_window's docstring. (2) This process wasn't
        per-monitor-DPI-aware (fixed once at module import time, see
        that comment) -- without it, Windows reports OTHER processes'
        window/element coordinates to us pre-scaled for this machine's
        display, while pyautogui's own clicks always target real
        physical pixels, so the two silently disagreed.

        THE RETRY ON SEARCH: verified live, on a genuinely cold Apple
        Music launch, pressing Enter in the search box can occasionally
        no-op if it lands before the app's search subsystem has finished
        initializing -- even though the window is already fully drawn
        and DOES accept the typed query text, just not the Enter that
        submits it. _looks_like_search_result catches this (Tab lands on
        a nav element instead of an actual result) and gives the page
        one more moment before giving up.
        """
        import pyautogui
        import comtypes.client as cc

        if device is None:
            other_devices = _list_active_playback_devices()
            if other_devices:
                device_names = ", ".join(name for name, _ in other_devices)
                return PCActionResult(
                    success=True, action="play_apple_music",
                    message=(
                        f"Where do you want to play '{query}' — your laptop "
                        f"speakers, or {device_names}?"
                    ),
                    data={"awaiting_device_choice": True, "query": query},
                )
        else:
            device_id = _find_playback_device_id(device)
            if device_id is None:
                return PCActionResult(
                    success=False, action="play_apple_music",
                    message=f"'{device}' isn't a playback device I can see right now.",
                )
            if not _set_default_playback_device(device_id):
                return PCActionResult(
                    success=False, action="play_apple_music",
                    message=f"Found '{device}' but couldn't switch playback to it.",
                )

        try:
            APP_LAUNCHERS["apple music"]()
        except Exception as exc:
            logger.error("play_apple_music couldn't launch Apple Music: %s", exc)
            return PCActionResult(
                success=False, action="play_apple_music",
                message=f"Couldn't open Apple Music: {exc}",
            )
        time.sleep(APPLE_MUSIC_LAUNCH_WAIT_SECONDS)

        hwnd = _find_process_window("AppleMusic.exe")
        if hwnd is None:
            return PCActionResult(
                success=False, action="play_apple_music",
                message="Couldn't find the Apple Music window after opening it.",
            )
        _focus_window(hwnd)
        time.sleep(0.6)  # let the foreground/maximize transition settle

        try:
            _search_apple_music_for(query)
        except Exception as exc:
            logger.error("play_apple_music couldn't search for %r: %s", query, exc)
            return PCActionResult(
                success=False, action="play_apple_music",
                message=f"Opened Apple Music but couldn't search for '{query}': {exc}",
            )

        # Imported here, not at module level -- same lazy-import reasoning
        # as take_screenshot's PIL import above. comtypes.gen's
        # UIAutomationClient wrapper is generated on first use (cached
        # after that), which is what CreateObject below needs.
        from comtypes.gen import UIAutomationClient as UIA

        try:
            uia = cc.CreateObject(UIA.CUIAutomation, interface=UIA.IUIAutomation)
            root = uia.ElementFromHandle(hwnd)

            # Tab walks keyboard focus forward through every focusable
            # element, eventually reaching the first 'Top Results' card
            # -- verified live, reliably, ONCE the window actually has
            # real focus (see FOCUS AND DPI above). How many presses
            # that takes varies a lot, though: verified live across
            # several runs, Tab's very first landing spot after Enter is
            # unpredictable (sometimes the search box itself, sometimes
            # a sidebar nav item, sometimes the account icon at the
            # bottom of the sidebar), so this keeps pressing and
            # checking instead of assuming a fixed number of presses --
            # cheap and safe since Tab alone can't trigger anything,
            # unlike re-sending Ctrl+F (verified live: that toggles the
            # search panel closed instead of re-focusing it).
            top_result = None
            for _tab_attempt in range(APPLE_MUSIC_MAX_TAB_ATTEMPTS):
                pyautogui.press("tab")
                time.sleep(0.3)
                candidate = uia.GetFocusedElement()
                if _looks_like_search_result(candidate, query):
                    top_result = candidate
                    break
            if top_result is None:
                return PCActionResult(
                    success=False, action="play_apple_music",
                    message=f"Couldn't get Apple Music to search for '{query}'.",
                )

            center = _uia_center(top_result)
            if center is None:
                return PCActionResult(
                    success=False, action="play_apple_music",
                    message=f"Found search results for '{query}' but couldn't locate the top result.",
                )
            pyautogui.click(*center)
            time.sleep(APPLE_MUSIC_DETAIL_PAGE_WAIT_SECONDS)

            # Prefer a specific matching track row over the page's big
            # Play button -- verified live that when the top result
            # routes to a parent ALBUM page (a song with no standalone
            # single release), Play starts the album from track 1, not
            # the song that was actually searched for. See
            # _find_track_row's docstring.
            track_row = _find_track_row(uia, root, query)
            if track_row is not None:
                center = _uia_center(track_row)
                if center is not None:
                    pyautogui.doubleClick(*center)
                else:
                    track_row = None

            if track_row is None:
                play_button = _find_content_play_button(uia, root)
                if play_button is None:
                    # The song/album page can take a beat longer to
                    # render than APPLE_MUSIC_DETAIL_PAGE_WAIT_SECONDS
                    # accounts for -- one more wait-and-look before
                    # giving up.
                    time.sleep(1.5)
                    play_button = _find_content_play_button(uia, root)
                if play_button is None:
                    return PCActionResult(
                        success=False, action="play_apple_music",
                        message=f"Opened '{query}' but couldn't find its Play button.",
                    )
                center = _uia_center(play_button)
                if center is None:
                    return PCActionResult(
                        success=False, action="play_apple_music",
                        message=f"Opened '{query}' but its Play button wasn't actually visible.",
                    )
                pyautogui.click(*center)

            # Verify playback actually started instead of trusting that
            # the click landed correctly -- verified live that it can
            # silently fail to (transport bar stayed on "Play", nothing
            # audibly happened) even when every coordinate along the way
            # checked out.
            if not _is_playing(uia, root):
                time.sleep(1.5)
            if not _is_playing(uia, root) and track_row is not None:
                # Safe to retry ONLY on the track-row path: double-
                # clicking a row starts/restarts that track, it doesn't
                # toggle it off, unlike the Play/Pause transport button
                # below. One extra attempt before giving up.
                center = _uia_center(track_row)
                if center is not None:
                    pyautogui.doubleClick(*center)
                    time.sleep(1.5)
            if not _is_playing(uia, root):
                return PCActionResult(
                    success=False, action="play_apple_music",
                    message=f"Opened '{query}' but it doesn't look like playback actually started.",
                )
        except Exception as exc:
            logger.error("play_apple_music failed for %r: %s", query, exc)
            return PCActionResult(
                success=False, action="play_apple_music",
                message=f"Found '{query}' but hit an error trying to play it: {exc}",
            )

        logger.info("Played Apple Music search result for: %s", query)
        return PCActionResult(
            success=True, action="play_apple_music",
            message=f"Playing {query} on Apple Music.",
            data={"query": query},
        )

    def force_quit_application(self, name: str) -> PCActionResult:
        """Force-terminates a process by name — 'kill it' for a frozen
        app. Deliberately NOT restricted to APP_LAUNCHERS's whitelist
        (unlike open_application): the whole point is handling whatever
        happens to be frozen, including apps this file never launched. A
        list-form subprocess call (not shell=True) means the process name
        can only ever be passed as taskkill's /IM argument, never
        interpreted as shell syntax."""
        image_name = name.strip()
        if not image_name:
            return PCActionResult(success=False, action="force_quit_application", message="No application name given.")
        if not image_name.lower().endswith(".exe"):
            image_name += ".exe"
        try:
            result = subprocess.run(
                ["taskkill", "/IM", image_name, "/F"],
                capture_output=True, text=True, timeout=10,
            )
            output = (result.stdout or result.stderr or "").strip()
            success = result.returncode == 0
            logger.info("force_quit_application(%r) -> exit %d: %s", name, result.returncode, output)
            return PCActionResult(
                success=success, action="force_quit_application",
                message=output or (f"Force-quit {name}." if success else f"Couldn't force-quit {name}."),
            )
        except Exception as exc:
            logger.error("force_quit_application(%r) failed: %s", name, exc)
            return PCActionResult(success=False, action="force_quit_application", message=f"Couldn't force-quit '{name}': {exc}")

    def install_application(self, name: str) -> PCActionResult:
        """Silently installs `name` via winget (Windows' built-in package
        manager) — winget's own signed-package verification against its
        trusted source repos is the real safety net here, not anything
        this method adds on top."""
        try:
            result = subprocess.run(
                ["winget", "install", "--name", name, "--silent",
                 "--accept-package-agreements", "--accept-source-agreements"],
                capture_output=True, text=True, timeout=WINGET_TIMEOUT_SECONDS,
            )
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            if len(output) > RUN_COMMAND_MAX_OUTPUT_CHARS:
                output = output[:RUN_COMMAND_MAX_OUTPUT_CHARS] + "\n... (truncated)"
            success = result.returncode == 0
            logger.info("install_application(%r) -> exit %d", name, result.returncode)
            return PCActionResult(
                success=success, action="install_application",
                message=output or (f"Installed {name}." if success else f"Install of {name} failed."),
            )
        except subprocess.TimeoutExpired:
            return PCActionResult(success=False, action="install_application", message=f"Install of '{name}' timed out after {WINGET_TIMEOUT_SECONDS:.0f}s.")
        except FileNotFoundError:
            return PCActionResult(success=False, action="install_application", message="winget isn't available on this machine.")
        except Exception as exc:
            logger.error("install_application(%r) failed: %s", name, exc)
            return PCActionResult(success=False, action="install_application", message=f"Couldn't install '{name}': {exc}")

    def uninstall_application(self, name: str) -> PCActionResult:
        """Silently uninstalls `name` via winget."""
        try:
            result = subprocess.run(
                ["winget", "uninstall", "--name", name, "--silent",
                 "--accept-source-agreements"],
                capture_output=True, text=True, timeout=WINGET_TIMEOUT_SECONDS,
            )
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            if len(output) > RUN_COMMAND_MAX_OUTPUT_CHARS:
                output = output[:RUN_COMMAND_MAX_OUTPUT_CHARS] + "\n... (truncated)"
            success = result.returncode == 0
            logger.info("uninstall_application(%r) -> exit %d", name, result.returncode)
            return PCActionResult(
                success=success, action="uninstall_application",
                message=output or (f"Uninstalled {name}." if success else f"Uninstall of {name} failed."),
            )
        except subprocess.TimeoutExpired:
            return PCActionResult(success=False, action="uninstall_application", message=f"Uninstall of '{name}' timed out after {WINGET_TIMEOUT_SECONDS:.0f}s.")
        except FileNotFoundError:
            return PCActionResult(success=False, action="uninstall_application", message="winget isn't available on this machine.")
        except Exception as exc:
            logger.error("uninstall_application(%r) failed: %s", name, exc)
            return PCActionResult(success=False, action="uninstall_application", message=f"Couldn't uninstall '{name}': {exc}")

    def search_files(self, query: str) -> PCActionResult:
        """Recursively searches FILE_SEARCH_ROOTS for filenames containing
        `query` (case-insensitive). Scoped to the user's home directory,
        not the whole drive — see FILE_SEARCH_ROOTS's comment for why."""
        query = query.strip().lower()
        if not query:
            return PCActionResult(success=False, action="search_files", message="No search term given.")

        start_time = time.time()
        matches = []
        for root_dir in FILE_SEARCH_ROOTS:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                dirnames[:] = [d for d in dirnames if d not in FILE_SEARCH_SKIP_DIRS and not d.startswith(".")]
                for filename in filenames:
                    if query in filename.lower():
                        matches.append(os.path.join(dirpath, filename))
                        if len(matches) >= FILE_SEARCH_MAX_RESULTS:
                            break
                if len(matches) >= FILE_SEARCH_MAX_RESULTS or time.time() - start_time > FILE_SEARCH_TIMEOUT_SECONDS:
                    break
            if len(matches) >= FILE_SEARCH_MAX_RESULTS or time.time() - start_time > FILE_SEARCH_TIMEOUT_SECONDS:
                break

        logger.info("search_files(%r) -> %d match(es)", query, len(matches))
        if not matches:
            return PCActionResult(success=False, action="search_files", message=f"No files matching '{query}' found.")
        return PCActionResult(
            success=True, action="search_files",
            message=f"Found {len(matches)} match(es): " + "; ".join(os.path.basename(m) for m in matches),
            data={"paths": matches},
        )

    def block_domain(self, domain: str) -> PCActionResult:
        """Blocks `domain` (and its www. subdomain) by redirecting it to
        127.0.0.1 in the Windows hosts file. Requires this process to be
        running elevated (admin) — the hosts file is admin-write-only."""
        domain = domain.strip().lower().removeprefix("http://").removeprefix("https://").removeprefix("www.").rstrip("/")
        if not domain:
            return PCActionResult(success=False, action="block_domain", message="No domain given.")
        try:
            with open(HOSTS_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            new_lines = _add_domain_block(lines, domain)
            with open(HOSTS_FILE_PATH, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            logger.info("Blocked domain: %s", domain)
            return PCActionResult(success=True, action="block_domain", message=f"Blocked {domain}.")
        except PermissionError:
            return PCActionResult(
                success=False, action="block_domain",
                message="Couldn't edit the hosts file — needs to run as Administrator.",
            )
        except Exception as exc:
            logger.error("block_domain(%r) failed: %s", domain, exc)
            return PCActionResult(success=False, action="block_domain", message=f"Couldn't block '{domain}': {exc}")

    def unblock_domain(self, domain: str) -> PCActionResult:
        """Removes a citra-managed hosts-file block for `domain` — see
        block_domain's docstring for the same admin-rights requirement."""
        domain = domain.strip().lower().removeprefix("http://").removeprefix("https://").removeprefix("www.").rstrip("/")
        if not domain:
            return PCActionResult(success=False, action="unblock_domain", message="No domain given.")
        try:
            with open(HOSTS_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            new_lines = _remove_domain_block(lines, domain)
            if len(new_lines) == len(lines):
                return PCActionResult(success=True, action="unblock_domain", message=f"{domain} wasn't blocked.")
            with open(HOSTS_FILE_PATH, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            logger.info("Unblocked domain: %s", domain)
            return PCActionResult(success=True, action="unblock_domain", message=f"Unblocked {domain}.")
        except PermissionError:
            return PCActionResult(
                success=False, action="unblock_domain",
                message="Couldn't edit the hosts file — needs to run as Administrator.",
            )
        except Exception as exc:
            logger.error("unblock_domain(%r) failed: %s", domain, exc)
            return PCActionResult(success=False, action="unblock_domain", message=f"Couldn't unblock '{domain}': {exc}")

    def set_theme(self, mode: str) -> PCActionResult:
        """Switches Windows between light and dark mode (both the system
        chrome and apps) via the same registry values Settings >
        Personalization > Colors itself writes to."""
        mode = mode.strip().lower()
        if mode not in ("dark", "light"):
            return PCActionResult(success=False, action="set_theme", message="Mode must be 'dark' or 'light'.")
        value = 0 if mode == "dark" else 1
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "AppsUseLightTheme", 0, winreg.REG_DWORD, value)
                winreg.SetValueEx(key, "SystemUsesLightTheme", 0, winreg.REG_DWORD, value)
            logger.info("Set theme: %s", mode)
            return PCActionResult(success=True, action="set_theme", message=f"Switched to {mode} mode.")
        except Exception as exc:
            logger.error("set_theme(%r) failed: %s", mode, exc)
            return PCActionResult(success=False, action="set_theme", message=f"Couldn't switch theme: {exc}")

    def set_brightness(self, percent: int) -> PCActionResult:
        """Sets screen brightness via WMI — works for most laptop
        built-in displays, not external monitors (most external monitors
        don't expose brightness control through this same WMI class)."""
        try:
            percent = max(0, min(100, int(percent)))
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{percent})"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                error = (result.stderr or "").strip()
                return PCActionResult(
                    success=False, action="set_brightness",
                    message=f"Couldn't set brightness (this may not be a built-in laptop display): {error}",
                )
            logger.info("Set brightness: %d%%", percent)
            return PCActionResult(success=True, action="set_brightness", message=f"Brightness set to {percent}%.")
        except Exception as exc:
            logger.error("set_brightness(%r) failed: %s", percent, exc)
            return PCActionResult(success=False, action="set_brightness", message=f"Couldn't set brightness: {exc}")


# -----------------------------------------------------------------------------
# TOOL SCHEMA — same OpenAI-style shape as jarvis_hardware_api.JARVIS_TOOL_SCHEMA
# -----------------------------------------------------------------------------
PC_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": (
                "Open an application on the user's computer by name. Only "
                "works for a known set of applications (notepad, calculator, "
                "file explorer, paint, task manager, control panel, command "
                "prompt, settings, camera, chrome, edge, apple music) — if "
                "the user asks for something else, say so rather than "
                "guessing. To play a specific song, use play_apple_music "
                "instead — it opens the app AND starts the music."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The application's common name, e.g. 'notepad' or 'chrome'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_path",
            "description": (
                "Open an existing file or folder on the user's computer in "
                "its default application (e.g. a folder in File Explorer, a "
                "document in its associated app). The path must already "
                "exist — use this for things the user references by a real "
                "location, not to create anything. For common folders "
                "(downloads, documents, desktop, pictures, music, videos, "
                "home), just pass that word as the path — e.g. 'downloads' "
                "— rather than guessing a full path, since you cannot know "
                "the actual Windows account name on this machine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full filesystem path to the file or folder, OR "
                            "one of: downloads, documents, desktop, pictures, "
                            "music, videos, home."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_screen",
            "description": "Lock the computer's screen — same as pressing Windows+L. Use for 'lock my pc', 'lock the screen', 'I'm stepping away'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the user's screen and save it. Use for 'take a screenshot', 'capture my screen'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run any command on the user's Windows computer via the shell "
                "(cmd.exe) and return its output. Prefer a more specific tool "
                "when one exists (force_quit_application, install_application, "
                "uninstall_application, search_files, block_domain, set_theme, "
                "set_brightness) — use this for everything else: creating, "
                "moving, deleting, or renaming files; running a script; "
                "checking system info; or any other PC automation. This is "
                "real, unscoped execution with no undo, so prefer a precise, "
                "minimal command over a broad one, and never invent a "
                "destructive command (deleting files, formatting, etc.) unless "
                "the user clearly asked for exactly that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The exact command to run, in Windows cmd.exe "
                            "syntax (e.g. 'dir C:\\Users', 'echo hello > "
                            "C:\\Users\\HP\\Desktop\\note.txt')."
                        ),
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_layout",
            "description": (
                "Open a named group of applications together, e.g. 'open my "
                "writing setup'. Only works for a known set of layouts — if "
                "the user asks for one that doesn't exist, say so rather than "
                "guessing which apps they mean."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The layout's name, e.g. 'writing setup' or 'coding setup'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_apple_music",
            "description": (
                "Play a specific song, artist, or album on Apple Music. "
                "Opens Apple Music, searches, and starts playing the top "
                "result — a complete action in one call, not something you "
                "need to follow up with analyze_screen or click_at "
                "yourself. Use for 'play <song>', 'play something by "
                "<artist>', 'put on <album>'.\n"
                "DEVICE: call this WITHOUT `device` first. If another "
                "playback device (e.g. Bluetooth headphones/speaker) is "
                "currently connected, this returns a question asking which "
                "device to use, naming the actual options — just speak that "
                "question to the user verbatim rather than picking for "
                "them, then call this again with `device` set to whatever "
                "they answered once they reply. If there's no other device "
                "connected, it just plays on the laptop speakers "
                "immediately — no question, no `device` needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for and play, e.g. 'Blinding Lights The Weeknd' or 'Taylor Swift'.",
                    },
                    "device": {
                        "type": "string",
                        "description": (
                            "Which output device to play through, e.g. "
                            "'laptop speakers' or the Bluetooth device's "
                            "name. Omit on the first call — only pass this "
                            "on a follow-up call, after asking the user "
                            "which device they want."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "force_quit_application",
            "description": "Force-terminate a frozen or unresponsive application by name. Use for 'kill it', 'force quit X', 'X is frozen'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The application's name, e.g. 'chrome' or 'notepad'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_application",
            "description": "Silently install an application via winget (Windows Package Manager). This downloads and runs a real installer — only use it when the user clearly asked to install something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The application's name, as you'd search for it, e.g. 'Visual Studio Code' or 'Spotify'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "uninstall_application",
            "description": "Silently uninstall an application via winget. Only use when the user clearly asked to uninstall/remove something — this deletes the application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The application's name, as it appears installed, e.g. 'Spotify'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search the user's home directory (Desktop, Documents, Downloads, etc.) for files whose name contains the given text, and return matching paths. Use for 'find my file called X', 'where is X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in filenames, e.g. 'resume' or 'invoice'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "block_domain",
            "description": "Block a website by domain so it can't be reached from this computer (e.g. 'block reddit.com'). Needs this process to be running as Administrator — if it fails, say so rather than pretending it worked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "The domain to block, e.g. 'reddit.com' (no https:// or www.).",
                    }
                },
                "required": ["domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unblock_domain",
            "description": "Remove a previously added website block. Needs this process to be running as Administrator, same as block_domain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "The domain to unblock, e.g. 'reddit.com'.",
                    }
                },
                "required": ["domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_theme",
            "description": "Switch Windows between dark and light mode. Use for 'switch to dark mode', 'make it easier on my eyes', 'turn on light mode'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["dark", "light"], "description": "Which mode to switch to."},
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_brightness",
            "description": "Set the built-in screen's brightness. Use for 'turn down the brightness', 'set brightness to 50', 'make the screen dimmer'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {"type": "integer", "description": "Brightness level, 0-100."},
                },
                "required": ["percent"],
            },
        },
    },
]


if __name__ == "__main__":
    import json
    print("\n=== PC Control Tool Schema ===\n")
    print(json.dumps(PC_TOOL_SCHEMA, indent=2))
