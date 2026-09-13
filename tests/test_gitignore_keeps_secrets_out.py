"""
The .gitignore is a security control here: it is what keeps real phone
numbers, the phone API token, TLS keys and WiFi passwords out of a
public repository. A trailing "# comment" on a pattern line silently
breaks that pattern (git treats the comment as part of the filename),
which is how citra_api_token.txt and citra_contacts.json were once NOT
ignored at all. This test asks git itself, so it cannot drift.
"""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_PATHS = [
    "citra_contacts.json",
    "citra_api_token.txt",
    "citra_cert.pem",
    "citra_key.pem",
    "citra_muted.json",
    "relay_server/wifi_secrets.h",
    "ac_ir_server/wifi_secrets.h",
    "call_recordings/anything.wav",
    "generated_code/anything.py",
    "citra.log",
]

TEMPLATE_PATHS = [
    "relay_server/wifi_secrets.example.h",
    "citra_devices.example.json",
    "hey_citra.onnx",
]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not os.path.isdir(os.path.join(ROOT, ".git")),
    reason="needs git and a checkout to ask",
)


def _is_ignored(path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=ROOT, capture_output=True, text=True,
    )
    # 0 = ignored, 1 = not ignored, 128 = error
    assert completed.returncode in (0, 1), completed.stderr
    return completed.returncode == 0


@pytest.mark.parametrize("path", SECRET_PATHS)
def test_secret_files_are_ignored(path):
    assert _is_ignored(path), f"{path} would be committed"


@pytest.mark.parametrize("path", TEMPLATE_PATHS)
def test_templates_and_the_wake_word_model_are_not_ignored(path):
    assert not _is_ignored(path), f"{path} is a shipped file and must stay tracked"


def test_no_gitignore_pattern_carries_a_trailing_comment():
    with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert "#" not in stripped, f".gitignore:{number}: '{stripped}' has an inline comment"
