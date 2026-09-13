"""
run_command is deliberately open - it is what makes "Citra, do X on my
computer" work for things nobody wrote a tool for. Its one guard rail is
a short deny-list of commands that no spoken request can legitimately
mean and whose damage is instant: wiping a disk, deleting a system
root, nuking the registry, or shutting the machine down from under the
voice pipeline. These tests pin that list from both sides - what is
refused, and what everyday commands must NOT be.

subprocess.run is patched so nothing here ever executes.
"""
import subprocess

import pytest

import jarvis_pc_control
from jarvis_pc_control import (
    RUN_COMMAND_ENV,
    RUN_COMMAND_MAX_LENGTH,
    PCController,
    run_command_refusal,
)


@pytest.fixture(autouse=True)
def never_actually_run(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ran\n", stderr="")

    monkeypatch.setattr(jarvis_pc_control.subprocess, "run", fake_run)
    monkeypatch.delenv(RUN_COMMAND_ENV, raising=False)
    return calls


@pytest.mark.parametrize("command", [
    "format c:",
    'FORMAT  "D:"',
    "diskpart",
    "bcdedit /deletevalue",
    "rd /s /q C:\\",
    "rmdir /S /Q c:",
    "del /s /q C:\\Windows\\System32",
    "erase /s c:\\users",
    "rm -rf /",
    "rm -fr \\",
    "Remove-Item -Recurse -Force C:\\",
    "reg delete HKLM\\Software\\Microsoft /f",
    "shutdown /s /t 0",
    "echo bye && shutdown /r",
    "Stop-Computer",
    "Restart-Computer -Force",
    "vssadmin delete shadows /all",
    "cipher /w:C:",
    "mkfs.ext4 /dev/sda",
    "dd if=/dev/zero of=/dev/sda",
])
def test_catastrophic_commands_are_refused(command, never_actually_run):
    reason = run_command_refusal(command)
    assert reason is not None, command
    assert "can't be undone" in reason

    result = PCController().run_command(command)
    assert result.success is False
    assert result.message == reason
    assert never_actually_run == []


@pytest.mark.parametrize("command", [
    "dir",
    "dir /s *.py",
    "del myfile.txt",
    "del /s /q build",
    "rd /s /q node_modules",
    "Remove-Item -Recurse .\\dist",
    "rm -rf build/",
    "git status",
    "python script.py",
    "winget list",
    "ipconfig /all",
    "Get-Process | Sort-Object CPU -Descending | Select-Object -First 5",
    "echo the format of this report is fine",
    "reg query HKCU\\Software",
    "tasklist",
    "start notepad",
])
def test_everyday_commands_are_allowed(command, never_actually_run):
    assert run_command_refusal(command) is None, command

    result = PCController().run_command(command)
    assert result.success is True
    assert result.message == "ran"
    assert never_actually_run == [command]


def test_empty_command_is_refused(never_actually_run):
    assert run_command_refusal("   ") == "No command given."
    assert PCController().run_command("").success is False
    assert never_actually_run == []


def test_absurdly_long_commands_are_refused(never_actually_run):
    command = "echo " + "a" * RUN_COMMAND_MAX_LENGTH
    reason = run_command_refusal(command)
    assert reason is not None
    assert str(RUN_COMMAND_MAX_LENGTH) in reason
    assert PCController().run_command(command).success is False
    assert never_actually_run == []


def test_the_tool_can_be_switched_off_for_a_household(monkeypatch, never_actually_run):
    monkeypatch.setenv(RUN_COMMAND_ENV, "off")
    result = PCController().run_command("dir")
    assert result.success is False
    assert "switched off" in result.message
    assert never_actually_run == []


def test_quotes_and_spacing_do_not_hide_a_refused_command():
    assert run_command_refusal('"format"   c:') is not None
    assert run_command_refusal("reg   'delete'  HKLM") is not None


def test_a_refused_command_is_logged_as_a_warning(caplog):
    with caplog.at_level("WARNING", logger="jarvis_pc_control"):
        PCController().run_command("format c:")
    assert any("Refused command" in record.getMessage() for record in caplog.records)
